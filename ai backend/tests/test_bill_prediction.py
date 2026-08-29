"""
tests/test_bill_prediction.py
-----------------------------
STAGE 10: AI Bill Prediction tests.

All fixtures are DETERMINISTIC SYNTHETIC data (clearly labeled test data, never
persisted anywhere real). Firebase is fully mocked.

Covers the 18 required scenarios:
  1.  normal bill prediction
  2.  sufficient forecast
  3.  insufficient forecast
  4.  zero consumption
  5.  multiple PZEM aggregation
  6.  rate calculation
  7.  invalid rate
  8.  missing energy
  9.  NaN / null data
 10.  W -> kW -> kWh conversion
 11.  actual + forecast combination
 12.  predicted bill calculation
 13.  predicted difference
 14.  confidence propagation
 15.  deterministic output
 16.  idempotent Firebase persistence
 17.  Firebase failure handling
 18.  Stage 1-9 regression (integration with Stage 9 output)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ai import bill_prediction as bp
from tests.test_anomaly_detection import _settings
from tests.test_forecast import _daily_series, _pre, START_TS, SLOT, BIN


# ---------------------------------------------------------------------------
# Firebase mock (in-memory) + forecast-record helpers
# ---------------------------------------------------------------------------

class FakeRef:
    """In-memory stand-in for a firebase_admin db reference."""

    def __init__(self, store: dict, path: str):
        self._store, self._path = store, path

    def child(self, key: str) -> "FakeRef":
        return type(self)(self._store, f"{self._path}/{key}")

    def get(self):
        return self._store.get(self._path)

    def set(self, value) -> None:
        self._store[self._path] = value


@pytest.fixture
def fake_db(monkeypatch) -> dict:
    store: dict = {}
    monkeypatch.setattr(bp, "_db_ref", lambda path: FakeRef(store, path))
    return store


def _horizon(status="FORECAST", powers=None, confidence="medium"):
    powers = powers or []
    return {
        "status": status,
        "confidence": confidence,
        "reason": None if status == "FORECAST" else "insufficient_data",
        "start_ts": 0, "end_ts": 0, "count": len(powers),
        "timestamps": [],
        "forecast_power_w": list(powers),
        "lower_bound": [], "upper_bound": [],
    }


def _system_record(horizon_24, horizon_7, anchor=1_700_000_000, confidence="medium"):
    """Mimics a persisted Stage 9 system forecast record (dict form)."""
    h24 = _horizon(powers=horizon_24, confidence=confidence)
    h7 = _horizon(powers=horizon_7, confidence=confidence)
    return {
        "anchor_timestamp": anchor,
        "confidence": confidence,
        "forecast_24h": h24,
        "forecast_7d": h7,
    }


# ---------------------------------------------------------------------------
# 1. normal bill prediction
# ---------------------------------------------------------------------------

def test_normal_bill_prediction():
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN))
    r = bp.predict_bill_from_record(100.0, rec, horizon="forecast_24h", rate=8.0, billing_period="30d")
    assert r["status"] == "OK"
    # 288 * 1000 W over 5-min slots = 24 kWh
    assert r["forecast_energy_kwh"] == 24.0
    assert r["estimated_total_energy_kwh"] == 124.0
    assert r["estimated_bill"] == pytest.approx(124.0 * 8.0)
    assert r["predicted_difference"] == pytest.approx(24.0 * 8.0)
    assert r["forecast_confidence"] == "medium"


# ---------------------------------------------------------------------------
# 2. sufficient forecast
# ---------------------------------------------------------------------------

def test_sufficient_forecast_is_ok():
    rec = _system_record([500.0] * BIN, [500.0] * (7 * BIN))
    r = bp.predict_bill_from_record(10.0, rec, horizon="forecast_24h", rate=5.0)
    assert r["status"] == "OK"


# ---------------------------------------------------------------------------
# 3. insufficient forecast
# ---------------------------------------------------------------------------

def test_insufficient_forecast_withheld():
    rec = _system_record(None, None)
    rec["forecast_24h"] = _horizon(status="NO_FORECAST")
    rec["forecast_7d"] = _horizon(status="NO_FORECAST")
    r = bp.predict_bill_from_record(10.0, rec, horizon="forecast_24h", rate=5.0)
    assert r["status"] == "INSUFFICIENT"
    assert r["estimated_bill"] is None
    assert r["forecast_energy_kwh"] is None
    assert "forecast" in (r["reason"] or "").lower()


# ---------------------------------------------------------------------------
# 4. zero consumption
# ---------------------------------------------------------------------------

def test_zero_actual_consumption():
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN))
    r = bp.predict_bill_from_record(0.0, rec, horizon="forecast_24h", rate=8.0)
    assert r["status"] == "OK"
    assert r["actual_energy_kwh"] == 0.0
    assert r["estimated_total_energy_kwh"] == 24.0
    assert r["estimated_bill"] == pytest.approx(24.0 * 8.0)


# ---------------------------------------------------------------------------
# 5. multiple PZEM aggregation (system forecast is the pointwise sum)
# ---------------------------------------------------------------------------

def test_multiple_pzem_aggregation():
    # System forecast = PZEM1 (1000 W) + PZEM2 (1000 W) = 2000 W flat
    p1 = [1000.0] * BIN
    p2 = [1000.0] * BIN
    system_pw = [a + b for a, b in zip(p1, p2)]
    rec = _system_record(system_pw, system_pw)
    r = bp.predict_bill_from_record(0.0, rec, horizon="forecast_24h", rate=1.0)
    # 288 * 2000 W = 48 kWh
    assert r["forecast_energy_kwh"] == 48.0
    assert r["estimated_bill"] == pytest.approx(48.0)


# ---------------------------------------------------------------------------
# 6. rate calculation
# ---------------------------------------------------------------------------

def test_rate_calculation():
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN))
    r = bp.predict_bill_from_record(50.0, rec, horizon="forecast_24h", rate=10.0)
    assert r["estimated_bill"] == pytest.approx((50.0 + 24.0) * 10.0)


# ---------------------------------------------------------------------------
# 7. invalid rate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_rate", [0.0, -5.0, float("nan"), "x"])
def test_invalid_rate_no_bill_but_energy_kept(bad_rate):
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN))
    r = bp.predict_bill_from_record(50.0, rec, horizon="forecast_24h", rate=bad_rate)
    assert r["status"] == "OK"                       # energy still computed
    assert r["actual_energy_kwh"] == 50.0
    assert r["forecast_energy_kwh"] == 24.0
    assert r["estimated_bill"] is None               # no fake bill
    assert r["predicted_difference"] is None


# ---------------------------------------------------------------------------
# 8. missing energy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_actual", [None, float("nan")])
def test_missing_actual_energy_insufficient(bad_actual):
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN))
    r = bp.predict_bill_from_record(bad_actual, rec, horizon="forecast_24h", rate=8.0)
    assert r["status"] == "INSUFFICIENT"
    assert "actual" in (r["reason"] or "").lower()


def test_negative_actual_energy_insufficient():
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN))
    r = bp.predict_bill_from_record(-10.0, rec, horizon="forecast_24h", rate=8.0)
    assert r["status"] == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# 9. NaN / null data in forecast power
# ---------------------------------------------------------------------------

def test_nan_null_power_dropped():
    powers = [1000.0] * BIN
    powers[5] = None
    powers[50] = float("nan")
    powers[100] = float("inf")
    rec = _system_record(powers, powers)
    r = bp.predict_bill_from_record(0.0, rec, horizon="forecast_24h", rate=1.0)
    assert r["status"] == "OK"
    # 285 valid points * 1000 W = 23.75 kWh (no NaN/Inf leaked)
    assert r["forecast_energy_kwh"] == pytest.approx(285 * 1000 / 12000)
    assert r["estimated_total_energy_kwh"] >= 0


# ---------------------------------------------------------------------------
# 10. W -> kW -> kWh conversion
# ---------------------------------------------------------------------------

def test_w_to_kwh_conversion_single_slot():
    # 1000 W over a single 5-min slot = 1000/1000 kW * (5/60) h = 0.08333 kWh
    rec = {"forecast_24h": _horizon(powers=[1000.0]), "forecast_7d": _horizon(powers=[]), "anchor_timestamp": 1}
    kwh = bp.forecast_energy_kwh(rec["forecast_24h"])
    assert kwh == pytest.approx(1000.0 / 12000.0)


def test_w_to_kwh_conversion_288_slots():
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN))
    r = bp.predict_bill_from_record(0.0, rec, horizon="forecast_24h", rate=1.0)
    assert r["forecast_energy_kwh"] == pytest.approx(24.0)   # 288 * 1000 / 12000


# ---------------------------------------------------------------------------
# 11. actual + forecast combination
# ---------------------------------------------------------------------------

def test_actual_plus_forecast_combination():
    rec = _system_record([2000.0] * BIN, [2000.0] * (7 * BIN))  # 48 kWh forecast
    r = bp.predict_bill_from_record(52.0, rec, horizon="forecast_24h", rate=1.0)
    assert r["estimated_total_energy_kwh"] == pytest.approx(100.0)
    assert r["forecast_energy_kwh"] == pytest.approx(48.0)


# ---------------------------------------------------------------------------
# 12. predicted bill calculation (explicit)
# ---------------------------------------------------------------------------

def test_predicted_bill_calculation():
    rec = _system_record([500.0] * BIN, [500.0] * (7 * BIN))   # 12 kWh forecast
    r = bp.predict_bill_from_record(88.0, rec, horizon="forecast_24h", rate=6.5)
    assert r["estimated_bill"] == pytest.approx(100.0 * 6.5)


# ---------------------------------------------------------------------------
# 13. predicted difference
# ---------------------------------------------------------------------------

def test_predicted_difference():
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN))  # 24 kWh forecast
    r = bp.predict_bill_from_record(100.0, rec, horizon="forecast_24h", rate=8.0)
    # difference = forecast energy * rate = 24 * 8
    assert r["predicted_difference"] == pytest.approx(24.0 * 8.0)
    assert r["predicted_difference"] == pytest.approx(r["estimated_bill"] - 100.0 * 8.0)


# ---------------------------------------------------------------------------
# 14. confidence propagation
# ---------------------------------------------------------------------------

def test_confidence_propagation():
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN), confidence="high")
    r = bp.predict_bill_from_record(10.0, rec, horizon="forecast_24h", rate=1.0)
    assert r["forecast_confidence"] == "high"
    rec_low = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN), confidence="low")
    r2 = bp.predict_bill_from_record(10.0, rec_low, horizon="forecast_24h", rate=1.0)
    assert r2["forecast_confidence"] == "low"


# ---------------------------------------------------------------------------
# 15. deterministic output
# ---------------------------------------------------------------------------

def test_deterministic_output():
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN))
    a = bp.predict_bill_from_record(10.0, rec, horizon="forecast_24h", rate=8.0)
    b = bp.predict_bill_from_record(10.0, rec, horizon="forecast_24h", rate=8.0)
    for k in ("actual_energy_kwh", "forecast_energy_kwh", "estimated_total_energy_kwh",
              "estimated_bill", "predicted_difference"):
        assert a[k] == b[k]


# ---------------------------------------------------------------------------
# 16. idempotent Firebase persistence
# ---------------------------------------------------------------------------

def test_idempotent_rerun_no_duplicates(fake_db: dict):
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN), anchor=123)
    r = bp.predict_bill_from_record(10.0, rec, horizon="forecast_24h", rate=8.0)
    first = bp.write_bill_prediction(r, 123)
    stored_before = fake_db["ai/bill_prediction/123"]["estimated_bill"]
    second = bp.write_bill_prediction(r, 123)
    assert len([k for k in fake_db if k.startswith("ai/bill_prediction/")]) == 1
    assert fake_db["ai/bill_prediction/123"]["estimated_bill"] == stored_before
    assert first and second


def test_persistence_payload_shape(fake_db: dict):
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN), anchor=123)
    r = bp.predict_bill_from_record(10.0, rec, horizon="forecast_24h", rate=8.0)
    bp.write_bill_prediction(r, 123)
    p = fake_db["ai/bill_prediction/123"]
    for f in ("timestamp", "actual_energy_kwh", "forecast_energy_kwh",
              "estimated_total_energy_kwh", "rate", "estimated_bill",
              "predicted_difference", "forecast_confidence", "billing_period",
              "source_stage"):
        assert f in p
    assert p["source_stage"] == "stage10/bill_prediction"
    assert p["timestamp"] == 123


# ---------------------------------------------------------------------------
# 17. Firebase failure handling
# ---------------------------------------------------------------------------

def test_firebase_failure_returns_false(monkeypatch):
    class BoomRef:
        def child(self, key): return self
        def get(self): return None
        def set(self, value): raise RuntimeError("simulated Firebase outage")

    monkeypatch.setattr(bp, "_db_ref", lambda path: BoomRef())
    rec = _system_record([1000.0] * BIN, [1000.0] * (7 * BIN), anchor=9)
    r = bp.predict_bill_from_record(10.0, rec, horizon="forecast_24h", rate=8.0)
    assert bp.write_bill_prediction(r, 9) is False


# ---------------------------------------------------------------------------
# 18. Stage 1-9 regression: consume real Stage 9 output, and actual-energy
#     aggregation across PZEMs via the existing Stage 1 loader
# ---------------------------------------------------------------------------

def test_bill_prediction_consumes_stage9_system_forecast(tmp_path: Path):
    """Prove Stage 10 slots into the existing Stage 9 pipeline: build a real
    Stage 9 fleet forecast, then feed its SYSTEM record straight into Stage 10
    (no bespoke adapter format)."""
    settings = _settings(tmp_path)
    frames = {n: _daily_series(days=3) for n in range(1, settings.pzem_count + 1)}
    pres = {n: _pre(n, frames[n], available_days=3.0) for n in frames}
    _results, system = __import__("ai.forecast", fromlist=["forecast"]).run_forecast_pipeline(
        settings=settings, preprocess_results=pres
    )
    # actual energy supplied directly (real run derives it from data_loader)
    r = bp.predict_bill_from_record(100.0, system, horizon="forecast_24h", rate=7.0)
    assert r["status"] == "OK"
    assert r["forecast_energy_kwh"] > 0
    assert r["forecast_confidence"] in ("low", "medium", "high")


def test_actual_energy_aggregation_via_stage1_loader(tmp_path: Path, monkeypatch):
    """Actual energy must come from the EXISTING Stage 1 history loader, summed
    across valid PZEMs with negative (reset) counters clamped."""
    from ai.data_loader import HistoryLoadResult

    def fake_fetch(settings):
        def mk(energy_series):
            n = len(energy_series)
            return HistoryLoadResult(
                pzem_number=0,
                frame=pd.DataFrame({
                    "timestamp": [START_TS + i * SLOT for i in range(n)],
                    "voltage": [230.0] * n, "current": [1.0] * n,
                    "power": [100.0] * n, "energy": energy_series,
                    "frequency": [50.0] * n, "pf": [0.9] * n,
                }),
                available_days=3.0, requested_days=60,
                served_from_cache_only=False, dropped_rows=0, duplicate_keys_collapsed=0,
            )
        return {
            1: mk([100.0, 200.0]),   # +100 kWh
            2: mk([50.0, 150.0]),    # +100 kWh
            3: mk([300.0, 250.0]),   # counter decreased -> clamped to 0
        }

    monkeypatch.setattr("ai.data_loader.fetch_all_history", fake_fetch)
    total = bp.compute_actual_energy_from_history(settings=_settings(tmp_path))
    assert total == pytest.approx(200.0)
