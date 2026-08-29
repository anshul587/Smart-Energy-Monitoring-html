"""
tests/test_energy_saving.py
---------------------------
Deterministic tests for STAGE 11 (evidence-based energy-saving suggestions).

Uses synthetic fixtures only (no Firebase / no real data), mirroring the
existing Stage 9/10 test conventions. The 17 required scenarios are covered:
normal / peak / idle / pf / current / forecast / combined / per-PZEM / system /
savings / invalid / insufficient / deterministic / idempotent / firebase-failure /
priority / stage-1-10 regression.
"""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ai import energy_saving as es
from ai.energy_saving import (
    MeterEvidence,
    Recommendation,
    analyze_meter,
    build_energy_saving_payload,
    compute_anchor,
    generate_recommendations,
    set_firebase_ref_for_test,
    write_energy_saving,
)

BASE_TS = 1_700_000_000


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def make_frame(days=4, power_profile=None, pf=0.98, current_profile=None,
               base_current=0.5, n_per_day=288):
    n = days * n_per_day
    ts = np.arange(BASE_TS, BASE_TS + n * 300, 300)
    tod = (ts % 86400) // 60  # minutes since midnight
    if power_profile is None:
        power = np.full(n, 100.0)
    else:
        power = np.array([power_profile(int(t)) for t in tod], dtype=float)
    if current_profile is None:
        current = base_current * (power / 100.0)
    else:
        current = np.array([current_profile(int(t)) for t in tod], dtype=float)
    pf_arr = np.full(n, pf)
    df = pd.DataFrame({
        "timestamp": ts,
        "voltage": 230.0,
        "current": current,
        "power": power,
        "energy": np.cumsum(power * 300 / 3600.0),
        "frequency": 50.0,
        "pf": pf_arr,
    })
    return df


def peak_result(status="PEAK_FOUND", above=800.0, pk=1200.0, base=400.0):
    return SimpleNamespace(status=status, peak_above_baseline_w=above,
                          peak_power_w=pk, baseline_power_w=base)


def standby_profile(t, standby=80.0, operating=800.0, frac=0.3):
    # ~frac of 5-min samples are in the high "operating" mode, rest standby
    return operating if (t // 5) % 10 < int(frac * 10) else standby


def forecast_result(high_window=False):
    n = 288
    start = BASE_TS
    if high_window:
        power = np.where(
            ((np.arange(n) * 300 // 60) % 1440) >= 1080, 800.0, 100.0
        )
    else:
        power = np.full(n, 100.0)
    return SimpleNamespace(
        forecast_24h={
            "status": "FORECAST",
            "start_ts": start,
            "count": n,
            "confidence": "high",
            "forecast_power_w": power.tolist(),
        },
        forecast_7d={"status": "NO_FORECAST", "reason": "n/a"},
    )


# ---------------------------------------------------------------------------
# 1. normal operation -> no unnecessary recommendation
# ---------------------------------------------------------------------------

def test_normal_operation_no_recommendation():
    ev = MeterEvidence(pzem_number=1, feature_frame=make_frame())
    recs = generate_recommendations({1: ev})
    assert recs == []
    assert analyze_meter(ev, 0.0, BASE_TS) == []


# ---------------------------------------------------------------------------
# 2. repeated peak usage -> shift non-critical load
# ---------------------------------------------------------------------------

def test_repeated_peak_usage():
    def prof(t):
        return 800.0 if 1080 <= t < 1260 else 100.0  # 18:00-21:00
    ev = MeterEvidence(pzem_number=2, feature_frame=make_frame(power_profile=prof))
    recs = analyze_meter(ev, 0.0, BASE_TS)
    types = {r.recommendation_type for r in recs}
    assert "SHIFT_NON_CRITICAL_LOAD" in types
    r = next(x for x in recs if x.recommendation_type == "SHIFT_NON_CRITICAL_LOAD")
    assert r.priority in ("HIGH", "MEDIUM")
    assert r.evidence_window is not None


# ---------------------------------------------------------------------------
# 3. idle consumption
# ---------------------------------------------------------------------------

def test_idle_consumption():
    ev = MeterEvidence(pzem_number=3,
                       feature_frame=make_frame(power_profile=lambda t: standby_profile(t, 80.0, 800.0)))
    recs = analyze_meter(ev, 0.0, BASE_TS)
    r = next((x for x in recs if x.recommendation_type == "REDUCE_IDLE_CONSUMPTION"), None)
    assert r is not None
    assert r.potential_saving_kwh is not None
    assert r.estimated_percent_reduction is not None


# ---------------------------------------------------------------------------
# 4. poor power factor
# ---------------------------------------------------------------------------

def test_poor_power_factor():
    ev = MeterEvidence(pzem_number=4, feature_frame=make_frame(pf=0.75))
    recs = analyze_meter(ev, 0.0, BASE_TS)
    r = next((x for x in recs if x.recommendation_type == "IMPROVE_POWER_FACTOR"), None)
    assert r is not None
    # PF correction is not an active-energy (kWh) saving on a kWh tariff
    assert r.potential_saving_kwh is None


# ---------------------------------------------------------------------------
# 5. repeated high current
# ---------------------------------------------------------------------------

def test_repeated_high_current():
    def cur(t):
        return 5.0 if (t // 5) % 4 == 0 else 0.5  # ~25% of samples high
    ev = MeterEvidence(pzem_number=5,
                       feature_frame=make_frame(current_profile=cur, base_current=0.5))
    recs = analyze_meter(ev, 0.0, BASE_TS)
    r = next((x for x in recs if x.recommendation_type == "INVESTIGATE_HIGH_CURRENT"), None)
    assert r is not None
    assert r.supporting_metrics["max_current_a"] > r.supporting_metrics["median_current_a"]


# ---------------------------------------------------------------------------
# 6. forecasted high-load period
# ---------------------------------------------------------------------------

def test_forecasted_high_load():
    ev = MeterEvidence(pzem_number=6, feature_frame=make_frame(),
                       forecast_result=forecast_result(high_window=True))
    recs = analyze_meter(ev, 0.0, BASE_TS)
    r = next((x for x in recs if x.recommendation_type == "RESPOND_PREDICTABLE_HIGH_LOAD"), None)
    assert r is not None
    assert r.source_stages == ["stage9/forecast"]


# ---------------------------------------------------------------------------
# 7. combined evidence -> multiple recommendations, no double peak
# ---------------------------------------------------------------------------

def test_combined_evidence():
    def prof(t):
        return 800.0 if 1080 <= t < 1260 else 80.0
    ev = MeterEvidence(pzem_number=7, feature_frame=make_frame(power_profile=prof, pf=0.75))
    recs = analyze_meter(ev, 0.0, BASE_TS)
    types = {r.recommendation_type for r in recs}
    assert "REDUCE_IDLE_CONSUMPTION" in types
    assert "IMPROVE_POWER_FACTOR" in types
    assert "SHIFT_NON_CRITICAL_LOAD" in types
    # peak and recurring-peak must not both appear for the same meter
    assert not ({"REDUCE_PEAK_LOAD", "SHIFT_NON_CRITICAL_LOAD"} <= types)


# ---------------------------------------------------------------------------
# 8. per-PZEM recommendation
# ---------------------------------------------------------------------------

def test_per_pzem_recommendation():
    ev = MeterEvidence(pzem_number=4,
                       feature_frame=make_frame(power_profile=lambda t: standby_profile(t, 80.0, 800.0)))
    recs = generate_recommendations({4: ev})
    assert any(r.pzem_number == 4 for r in recs)
    # every per-PZEM rec is for the right meter; system recs (if any) are separate
    assert all(r.pzem_number == 4 for r in recs if r.pzem_number is not None)


# ---------------------------------------------------------------------------
# 9. system recommendation (aggregated valid PZEM data only)
# ---------------------------------------------------------------------------

def test_system_recommendation():
    sys_frame = make_frame(power_profile=lambda t: 800.0 if 1080 <= t < 1260 else 100.0)
    sys_ev = MeterEvidence(pzem_number=None, feature_frame=sys_frame)
    recs = analyze_meter(sys_ev, 0.0, BASE_TS)
    r = next((x for x in recs if x.recommendation_type == "SHIFT_NON_CRITICAL_LOAD"), None)
    assert r is not None
    assert r.pzem_number is None  # SYSTEM

    # also confirm generate() produces a system rec when per-PZEM meters sum to a peak
    def half(t):
        return 400.0 if 1080 <= t < 1260 else 100.0
    meters = {
        1: MeterEvidence(1, feature_frame=make_frame(power_profile=half)),
        2: MeterEvidence(2, feature_frame=make_frame(power_profile=half)),
    }
    recs2 = generate_recommendations(meters)
    assert any(r.pzem_number is None for r in recs2)


# ---------------------------------------------------------------------------
# 10. savings estimation (uses Stage 10 rate)
# ---------------------------------------------------------------------------

def test_savings_estimation():
    rate = 7.0
    ev = MeterEvidence(pzem_number=1,
                       feature_frame=make_frame(power_profile=lambda t: standby_profile(t, 80.0, 800.0)))
    recs = analyze_meter(ev, rate, BASE_TS)
    r = next(x for x in recs if x.recommendation_type == "REDUCE_IDLE_CONSUMPTION")
    assert r.potential_saving_kwh is not None
    assert r.potential_cost_saving is not None
    assert abs(r.potential_cost_saving - r.potential_saving_kwh * rate) < 1e-6


# ---------------------------------------------------------------------------
# 11. invalid / missing data -> graceful, no crash
# ---------------------------------------------------------------------------

def test_invalid_missing_data():
    assert generate_recommendations({1: MeterEvidence(pzem_number=1, feature_frame=None)}) == []
    bad = make_frame()
    bad["power"] = np.nan
    bad["pf"] = np.nan
    bad["current"] = np.nan
    ev = MeterEvidence(pzem_number=1, feature_frame=bad)
    assert analyze_meter(ev, 0.0, BASE_TS) == []


# ---------------------------------------------------------------------------
# 12. insufficient history -> no recommendation
# ---------------------------------------------------------------------------

def test_insufficient_history():
    n = 10
    ts = np.arange(BASE_TS, BASE_TS + n * 300, 300)
    df = pd.DataFrame({
        "timestamp": ts, "voltage": 230.0, "current": 5.0,
        "power": np.concatenate([np.full(5, 3000.0), np.full(5, 100.0)]),
        "energy": np.zeros(n), "frequency": 50.0, "pf": 0.99,
    })
    ev = MeterEvidence(pzem_number=1, feature_frame=df)
    assert analyze_meter(ev, 0.0, BASE_TS) == []
    assert generate_recommendations({1: ev}) == []


# ---------------------------------------------------------------------------
# 13. deterministic output
# ---------------------------------------------------------------------------

def test_deterministic_output():
    def prof(t):
        return 800.0 if 1080 <= t < 1260 else 80.0
    meters = {1: MeterEvidence(1, feature_frame=make_frame(power_profile=prof, pf=0.75))}
    a = generate_recommendations(meters)
    b = generate_recommendations(meters)
    assert json.dumps([asdict(r) for r in a], sort_keys=True) == \
        json.dumps([asdict(r) for r in b], sort_keys=True)


# ---------------------------------------------------------------------------
# 14. duplicate / idempotent persistence
# ---------------------------------------------------------------------------

class FakeRef:
    def __init__(self, store, path=()):
        self.store = store
        self.path = path

    def child(self, key):
        return FakeRef(self.store, self.path + (str(key),))

    def _k(self):
        return "/".join(self.path)

    def get(self):
        return self.store.get(self._k())

    def set(self, val):
        self.store[self._k()] = val
        return True


@pytest.fixture
def fake_firebase():
    store = {}
    set_firebase_ref_for_test(lambda path: FakeRef(store, (path,)))
    yield store
    set_firebase_ref_for_test(None)


def _some_recs():
    ev = MeterEvidence(pzem_number=1,
                       feature_frame=make_frame(power_profile=lambda t: standby_profile(t, 80.0, 800.0)))
    return generate_recommendations({1: ev})


def test_idempotent_persistence(fake_firebase):
    recs = _some_recs()
    anchor = 1_700_000_000
    r1 = write_energy_saving(recs, anchor)
    r2 = write_energy_saving(recs, anchor)
    r3 = write_energy_saving(recs, anchor + 1)
    assert r1["written"] is True
    assert r2["written"] is False and r2["reason"] == "exists"
    assert r3["written"] is True
    # stored payload is valid
    key = f"ai/energy_saving/{anchor}"
    assert fake_firebase[key]["recommendation_count"] == len(recs)


# ---------------------------------------------------------------------------
# 15. Firebase failure isolation
# ---------------------------------------------------------------------------

class FailRef:
    def child(self, key):
        return self

    def get(self):
        raise RuntimeError("firebase down")

    def set(self, val):
        raise RuntimeError("firebase down")


def test_firebase_failure_isolation():
    set_firebase_ref_for_test(lambda path: FailRef())
    try:
        res = write_energy_saving(_some_recs(), 1_700_000_000)
        assert res["written"] is False
        assert res["reason"] == "firebase_error"
    finally:
        set_firebase_ref_for_test(None)


# ---------------------------------------------------------------------------
# 16. priority classification
# ---------------------------------------------------------------------------

def test_priority_classification():
    # idle thresholds (standby floor low relative to operating level)
    lo = MeterEvidence(1, feature_frame=make_frame(power_profile=lambda t: standby_profile(t, 30.0, 1500.0)))
    hi = MeterEvidence(1, feature_frame=make_frame(power_profile=lambda t: standby_profile(t, 300.0, 1500.0)))
    lo_r = next(x for x in analyze_meter(lo, 0, BASE_TS)
                if x.recommendation_type == "REDUCE_IDLE_CONSUMPTION")
    hi_r = next(x for x in analyze_meter(hi, 0, BASE_TS)
                if x.recommendation_type == "REDUCE_IDLE_CONSUMPTION")
    assert lo_r.priority == "LOW"
    assert hi_r.priority == "HIGH"

    # pf thresholds
    med = MeterEvidence(1, feature_frame=make_frame(pf=0.85))
    crit = MeterEvidence(1, feature_frame=make_frame(pf=0.70))
    med_r = next(x for x in analyze_meter(med, 0, BASE_TS)
                 if x.recommendation_type == "IMPROVE_POWER_FACTOR")
    crit_r = next(x for x in analyze_meter(crit, 0, BASE_TS)
                  if x.recommendation_type == "IMPROVE_POWER_FACTOR")
    assert med_r.priority == "MEDIUM"
    assert crit_r.priority == "HIGH"

    # stage-7 peak thresholds
    pk_med = analyze_meter(MeterEvidence(1, feature_frame=make_frame(),
                                         peak_result=peak_result(above=600.0)), 0, BASE_TS)
    pk_hi = analyze_meter(MeterEvidence(1, feature_frame=make_frame(),
                                        peak_result=peak_result(above=1500.0)), 0, BASE_TS)
    pm = next((x for x in pk_med if x.recommendation_type == "REDUCE_PEAK_LOAD"), None)
    ph = next((x for x in pk_hi if x.recommendation_type == "REDUCE_PEAK_LOAD"), None)
    assert pm.priority == "MEDIUM"
    assert ph.priority == "HIGH"


# ---------------------------------------------------------------------------
# 17. Stage 1-10 regression (imports + no cross-stage breakage)
# ---------------------------------------------------------------------------

def test_stage_1_10_regression_imports():
    # Importing Stage 11 must not alter earlier stages' public APIs.
    import ai.preprocessing as pre
    import ai.peak_detection as pk
    import ai.maintenance_risk as mr
    import ai.forecast as fc
    import ai.bill_prediction as bp
    import ai.anomaly_detection as ad
    import ai.fault_diagnosis as fd

    for fn in ("run_preprocessing_pipeline",):
        assert callable(getattr(pre, fn))
    for fn in ("run_peak_detection_pipeline",):
        assert callable(getattr(pk, fn))
    for fn in ("run_maintenance_risk_pipeline",):
        assert callable(getattr(mr, fn))
    for fn in ("run_forecast_pipeline", "run_stage_9_pipeline"):
        assert callable(getattr(fc, fn))
    for fn in ("run_stage_10_pipeline", "write_bill_prediction"):
        assert callable(getattr(bp, fn))
    for fn in ("run_anomaly_detection_pipeline",):
        assert callable(getattr(ad, fn))
    for fn in ("run_fault_diagnosis_pipeline",):
        assert callable(getattr(fd, fn))

    # empty fleet -> empty, deterministic
    assert generate_recommendations({}) == []
    # payload builder works for the no-recommendation state
    payload = build_energy_saving_payload([], 1_700_000_000, rate=0.0)
    assert payload["status"] == "NO_RECOMMENDATION"
    assert payload["recommendation_count"] == 0


def test_compute_anchor_uses_latest_data():
    a = MeterEvidence(1, feature_frame=make_frame())
    b = MeterEvidence(2, feature_frame=make_frame())
    anchor = compute_anchor({1: a, 2: b})
    assert anchor == BASE_TS + (4 * 288 - 1) * 300
