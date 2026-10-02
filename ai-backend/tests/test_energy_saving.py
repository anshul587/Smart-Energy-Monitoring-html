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
    run_stage_11_pipeline,
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
# 4b. PF == 0 is a nil-load signature, never a poor-power-factor condition
#
# A PZEM reports pf == 0 exactly when no current is flowing. Measured over all
# 685 live rows, pf == 0 <=> power == 0 with zero exceptions, and on loaded rows
# pf is valid. So a median PF of 0 means "nothing was connected", not "poor
# power factor", and must not produce IMPROVE_POWER_FACTOR.
# ---------------------------------------------------------------------------

def _nil_load_frame(zero_fraction=0.8, days=4, n_per_day=288):
    """Real nil-load shape: pf is exactly 0 on every row where power is 0, and
    ~1.0 on the rows where the meter is actually loaded. With zero_fraction
    above 0.5 the median PF is 0 while the meter still has real load samples."""
    n = days * n_per_day
    ts = np.arange(BASE_TS, BASE_TS + n * 300, 300)
    dead = np.zeros(n, dtype=bool)
    dead[: int(n * zero_fraction)] = True
    power = np.where(dead, 0.0, 50.0)
    return pd.DataFrame({
        "timestamp": ts,
        "voltage": 230.0,
        "current": np.where(dead, 0.0, 0.22),
        "power": power,
        "energy": np.cumsum(power * 300 / 3600.0),
        "frequency": 50.0,
        "pf": np.where(dead, 0.0, 1.0),
    })


def test_detect_pf_skips_nil_load():
    frame = _nil_load_frame()
    # sanity: this frame really is the shape the audit measured
    assert float(frame["pf"].median()) == 0.0
    assert (frame.loc[frame["power"] == 0, "pf"] == 0).all()
    assert (frame.loc[frame["power"] > 0, "pf"] == 1.0).all()

    ev = MeterEvidence(pzem_number=1, feature_frame=frame)
    recs = analyze_meter(ev, 0.0, BASE_TS)
    assert not any(r.recommendation_type == "IMPROVE_POWER_FACTOR" for r in recs)
    # nil-load data is not an energy-saving condition at all
    assert recs == []


def test_detect_pf_skips_all_zero_pf_frame():
    frame = make_frame(power_profile=lambda t: 0.0, pf=0.0, base_current=0.0)
    assert (frame["pf"] == 0).all() and (frame["power"] == 0).all()

    ev = MeterEvidence(pzem_number=1, feature_frame=frame)
    assert es.detect_pf(ev, 0.0, BASE_TS) is None
    assert analyze_meter(ev, 0.0, BASE_TS) == []
    assert generate_recommendations({1: ev}) == []


def test_detect_pf_still_fires_on_genuine_low_pf():
    # Genuine poor PF on a loaded meter must still be reported, at the same
    # priorities as before the nil-load guard was added.
    med = MeterEvidence(1, feature_frame=make_frame(pf=0.85))
    crit = MeterEvidence(1, feature_frame=make_frame(pf=0.70))
    med_r = next(x for x in analyze_meter(med, 0, BASE_TS)
                 if x.recommendation_type == "IMPROVE_POWER_FACTOR")
    crit_r = next(x for x in analyze_meter(crit, 0, BASE_TS)
                  if x.recommendation_type == "IMPROVE_POWER_FACTOR")
    assert med_r.priority == "MEDIUM"
    assert crit_r.priority == "HIGH"
    # threshold itself is unchanged
    assert es.PF_POOR_THRESHOLD == 0.90
    assert es.PF_CRITICAL_THRESHOLD == 0.80


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


# ---------------------------------------------------------------------------
# 18. Stage 11 records the "ran, found nothing" state, not silence
#
# The /ai/energy_saving node must exist with status NO_RECOMMENDATION whenever
# the pipeline legitimately finds nothing, so the dashboard/API can tell
# "pipeline ran, no recommendation" apart from "pipeline never ran".
# ---------------------------------------------------------------------------

def test_stage11_persists_no_recommendation_record(fake_firebase):
    # Nil-load fleet: the only thing that could have fired is the PF=0 false
    # positive, which is now guarded, so this is a genuine no-recommendation run.
    meters = {n: MeterEvidence(n, feature_frame=_nil_load_frame())
              for n in range(1, 4)}
    res = run_stage_11_pipeline(meters, rate=5.0, force=True)
    anchor = res["anchor_timestamp"]

    assert res["recommendations"] == []
    assert res["persist"]["written"] is True

    payload = fake_firebase[f"ai/energy_saving/{anchor}"]
    assert payload["status"] == "NO_RECOMMENDATION"
    assert payload["recommendation_count"] == 0
    assert payload["recommendations"] == []
    assert payload["timestamp"] == anchor
    # the record must still name where it came from
    assert payload["source_stages"]

    # an empty fleet must also leave a truthful record behind
    res_empty = run_stage_11_pipeline({}, rate=0.0, force=True,
                                     anchor_ts=1_700_000_001)
    assert res_empty["recommendations"] == []
    empty_payload = fake_firebase["ai/energy_saving/1700000001"]
    assert empty_payload["status"] == "NO_RECOMMENDATION"
    assert empty_payload["recommendation_count"] == 0


# ---------------------------------------------------------------------------
# 19. Recurring-window detection needs a dense-enough series
#
# _recurring_high_window bins the day into 30-minute slots and compares per-bin
# medians. It refuses to claim a window when there are too few samples overall
# (fewer than MIN_EVIDENCE_SAMPLES), when the series has no positive level, or
# when the above-ratio bins hold fewer than MIN_SAMPLES_PER_BIN samples. The
# last one matters because a bin with a single reading yields that reading as
# its "median", which fabricates a recurring window on sparse/bursty data.
# ---------------------------------------------------------------------------

def _bin_counts(frame, col="power"):
    ts = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
    hod = ts.dt.hour * 60 + ts.dt.minute
    return frame.groupby((hod // es.BIN_MINUTES).astype(int))[col].size()


def test_recurring_window_requires_dense_bins():
    # Too few samples overall -> no window, no SHIFT_NON_CRITICAL_LOAD.
    sparse = make_frame(days=5, n_per_day=3,
                        power_profile=lambda t: 900.0 if t < 30 else 40.0)
    assert len(sparse) == 15 < es.MIN_EVIDENCE_SAMPLES
    assert es._recurring_high_window(sparse, "power") is None
    assert es.detect_recurring_peak(
        MeterEvidence(1, feature_frame=sparse), 0.0, BASE_TS) is None

    # All-zero / nil-load series has no meaningful level to recur above.
    flat_zero = make_frame(power_profile=lambda t: 0.0, pf=0.0, base_current=0.0)
    assert es._recurring_high_window(flat_zero, "power") is None

    # A genuinely dense series with a real recurring window must still fire,
    # otherwise this test would be passing for the wrong reason.
    def prof(t):
        return 800.0 if 1080 <= t < 1260 else 100.0
    dense = make_frame(days=4, power_profile=prof)
    assert es._recurring_high_window(dense, "power") is not None
    assert es.detect_recurring_peak(
        MeterEvidence(1, feature_frame=dense), 0.0, BASE_TS) is not None


def _sparse_bursty_frame(rows=105, days=30, n_spikes=30, spike_w=900.0, base_w=30.0):
    """The audited failure shape: a 30-day window holding only `rows` readings,
    so the bursts land in half-hour bins that each hold a SINGLE sample. Every
    spike sits in its own distinct bin (one per day), and the low-power filler
    is concentrated in separate bins, so no bin ever shows a recurring pattern.
    """
    rows_out = []
    for i in range(n_spikes):                       # bins 8..37, one sample each
        rows_out.append((BASE_TS + i * 86400 + (8 + i) * 30 * 60, spike_w))
    for i in range(rows - n_spikes):                # bins 4..6, low power
        d = i % days
        rows_out.append((BASE_TS + d * 86400 + (4 + (i // days)) * 30 * 60, base_w))
    return pd.DataFrame(sorted(rows_out), columns=["timestamp", "power"])


def test_recurring_window_rejects_sparse_bursty_single_sample_bins():
    frame = _sparse_bursty_frame()
    assert len(frame) == 105
    assert len(frame) >= es.MIN_EVIDENCE_SAMPLES  # the total-sample gate PASSES

    counts = _bin_counts(frame)
    # the failing premise: the above-ratio (spike) bins hold only one reading
    assert counts.min() >= 1
    spike_bins = frame.loc[frame["power"] > 500.0, "timestamp"]
    spike_ts = pd.to_datetime(spike_bins, unit="s", utc=True)
    spike_bin_ids = ((spike_ts.dt.hour * 60 + spike_ts.dt.minute) // es.BIN_MINUTES)
    for b in spike_bin_ids:
        assert counts[b] == 1, f"bin {b} should hold a single sample"

    # single-sample bins are not a recurring pattern
    assert es._recurring_high_window(frame, "power") is None
    rec = es.detect_recurring_peak(
        MeterEvidence(1, feature_frame=frame), 0.0, BASE_TS)
    assert rec is None, f"sparse bursty data fired: {rec}"

    types = {r.recommendation_type for r in
             analyze_meter(MeterEvidence(1, feature_frame=frame), 0.0, BASE_TS)}
    assert "SHIFT_NON_CRITICAL_LOAD" not in types


def test_recurring_window_preserved_for_dense_repeated_half_hour():
    # A genuine repeated half-hour window: every day 18:00-19:00 is high at a
    # 5-min cadence, so the 18:00 bin holds many samples. Must still fire.
    def prof(t):
        return 800.0 if 1080 <= t < 1140 else 60.0
    frame = make_frame(days=4, power_profile=prof)
    counts = _bin_counts(frame)
    assert counts.max() >= es.MIN_SAMPLES_PER_BIN

    win = es._recurring_high_window(frame, "power")
    assert win is not None
    assert win["peak_median"] == 800.0
    assert win["overall_median"] < win["peak_median"]

    rec = es.detect_recurring_peak(
        MeterEvidence(1, feature_frame=frame), 0.0, BASE_TS)
    assert rec is not None
    assert rec.recommendation_type == "SHIFT_NON_CRITICAL_LOAD"
    # unchanged by this fix: 800 W is 4x the 200 W typical -> HIGH
    assert rec.priority == "HIGH"


def test_recurring_window_min_samples_per_bin_boundary():
    """Boundary of MIN_SAMPLES_PER_BIN: a bin holding exactly the minimum
    number of samples still yields a window; one fewer yields None."""
    assert es.MIN_SAMPLES_PER_BIN == 2

    def frame_with_bin_samples(n_in_peak_bin, days=30):
        # exactly n_in_peak_bin readings in bin 36 (18:00) across the WHOLE window
        rows = []
        for i in range(n_in_peak_bin):
            rows.append((BASE_TS + i * 86400 + 36 * 30 * 60, 800.0))
        # low-power filler in other bins so the total clears MIN_EVIDENCE_SAMPLES
        for i in range(days):
            for k in range(6):
                rows.append((BASE_TS + i * 86400 + (4 + k) * 30 * 60, 60.0))
        return pd.DataFrame(sorted(rows), columns=["timestamp", "power"])

    # exactly MIN_SAMPLES_PER_BIN samples in the peak bin -> accepted
    at_min = frame_with_bin_samples(es.MIN_SAMPLES_PER_BIN)
    c = _bin_counts(at_min)
    ts = pd.to_datetime(at_min["timestamp"], unit="s", utc=True)
    peak_bin = ((ts.dt.hour * 60 + ts.dt.minute) // es.BIN_MINUTES)[at_min["power"] > 500.0].iloc[0]
    assert c[peak_bin] == es.MIN_SAMPLES_PER_BIN
    assert es._recurring_high_window(at_min, "power") is not None

    # one fewer -> the same single-sample spike is rejected
    below = frame_with_bin_samples(es.MIN_SAMPLES_PER_BIN - 1)
    c2 = _bin_counts(below)
    assert c2[peak_bin] == es.MIN_SAMPLES_PER_BIN - 1
    assert es._recurring_high_window(below, "power") is None
    assert es.detect_recurring_peak(
        MeterEvidence(1, feature_frame=below), 0.0, BASE_TS) is None
