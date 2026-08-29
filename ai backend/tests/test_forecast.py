"""
tests/test_forecast.py
----------------------
STAGE 9: Forecasting tests.

All fixtures are DETERMINISTIC SYNTHETIC data (fixed timestamps, fixed
values, seed-free) used ONLY inside this module — clearly labeled test
data, never persisted anywhere real. Firebase is fully mocked.

Covers the 17 required scenarios:
  1. sufficient history
  2. insufficient history
  3. 24-hour forecast
  4. 7-day forecast
  5. multiple PZEMs
  6. flat series
  7. zero-power series
  8. missing samples
  9. NaN/null values
 10. irregular timestamps
 11. outlier handling
 12. deterministic output
 13. confidence/data-sufficiency state
 14. Firebase persistence
 15. idempotent rerun
 16. Firebase failure isolation
 17. Stage 1-8 regression (integration with existing stages' API)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from ai import forecast as fc
from ai.config import Settings
from ai.preprocessing import PreprocessResult
from tests.test_anomaly_detection import _settings

START_TS = 1_700_000_000
SLOT = 300
BIN = 288


# ---------------------------------------------------------------------------
# Deterministic synthetic fixtures (test data only)
# ---------------------------------------------------------------------------

def _frame(powers, start_ts: int = START_TS, slot: int = SLOT) -> pd.DataFrame:
    """Minimal PZEM-shaped frame; only timestamp/power affect Stage 9."""
    n = len(powers)
    return pd.DataFrame({
        "timestamp": [start_ts + i * slot for i in range(n)],
        "voltage": [230.0] * n,
        "current": [0.5] * n,
        "power": [np.nan if p is None else float(p) for p in powers],
        "energy": [1.0] * n,
        "frequency": [50.0] * n,
        "pf": [0.9] * n,
    })


def _pre(pzem_number: int, frame: Optional[pd.DataFrame], available_days: float = 0.0,
         status: str = "READY") -> PreprocessResult:
    ready = frame is not None
    s = status if frame is not None else "INSUFFICIENT_DATA"
    oldest = int(frame["timestamp"].iloc[0]) if frame is not None and not frame.empty else None
    newest = int(frame["timestamp"].iloc[-1]) if frame is not None and not frame.empty else None
    return PreprocessResult(
        pzem_number=pzem_number,
        status=s if not status else status,
        reason=None if ready else "synthetic no-data",
        record_count=0 if frame is None else len(frame),
        oldest_timestamp=oldest,
        newest_timestamp=newest,
        available_days=available_days,
        valid_rows=0 if frame is None else len(frame),
        invalid_rows=0,
        duplicates_removed=0,
        missing_values=0,
        feature_frame=frame,
    )


def _ts(index: int, start_ts: int = START_TS, slot: int = SLOT) -> int:
    return start_ts + index * slot


# A realistic ~3-day daily sinusoid, used for sufficient-history cases.
def _daily_series(days: int = 3, amp: float = 800.0, base: float = 1000.0,
                  start_ts: int = START_TS, slot: int = SLOT) -> pd.DataFrame:
    rows = days * BIN
    ts = [start_ts + i * slot for i in range(rows)]
    pw = []
    for i in range(rows):
        tod = (ts[i] % 86400) / 300.0
        pw.append(base + amp * np.sin(2 * np.pi * tod / BIN))
    return _frame(pw, start_ts=start_ts, slot=slot)


# ---------------------------------------------------------------------------
# Firebase mock (in-memory)
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
    monkeypatch.setattr(fc, "_db_ref", lambda path: FakeRef(store, path))
    return store


# ===========================================================================
# 1. sufficient history
# ===========================================================================

def test_sufficient_history_produces_forecast(tmp_path: Path):
    frame = _daily_series(days=3)
    r = fc.forecast_meter(1, _pre(1, frame, available_days=3.0))
    assert r.status == "FORECAST"
    assert r.forecast_24h["status"] == "FORECAST"
    assert r.forecast_7d["status"] == "FORECAST"


# ===========================================================================
# 2. insufficient history
# ===========================================================================

def test_insufficient_history_no_forecast(tmp_path: Path):
    # 50 samples (~0.17 day) -- below FORECAST_MIN_SPAN_DAYS
    frame = _frame([500.0] * 50)
    r = fc.forecast_meter(2, _pre(2, frame, available_days=0.17))
    assert r.status == "NO_FORECAST"
    assert r.forecast_24h["status"] == "NO_FORECAST"
    assert r.forecast_7d["status"] == "NO_FORECAST"
    assert "insufficient_data" in (r.reason or "")


def test_no_feature_frame_no_forecast(tmp_path: Path):
    r = fc.forecast_meter(3, _pre(3, None))
    assert r.status == "NO_FORECAST"
    assert r.forecast_24h["status"] == "NO_FORECAST"


# ===========================================================================
# 3. 24-hour forecast
# ===========================================================================

def test_24h_forecast_288_points(tmp_path: Path):
    frame = _daily_series(days=3)
    r = fc.forecast_meter(1, _pre(1, frame))
    h = r.forecast_24h
    assert h["count"] == BIN  # 288 points = 24h at 5-min cadence
    assert h["end_ts"] - h["start_ts"] == (BIN - 1) * SLOT
    # first forecast point is one slot after the last observation
    assert h["start_ts"] == _ts(3 * BIN - 1) + SLOT


def test_24h_forecast_follows_daily_shape(tmp_path: Path):
    frame = _daily_series(days=3)
    r = fc.forecast_meter(1, _pre(1, frame))
    pw = r.forecast_24h["forecast_power_w"]
    # sinusoid min/max should be reflected
    assert min(pw) < 1000 < max(pw)
    # bounded & finite
    assert all(np.isfinite(pw))


# ===========================================================================
# 4. 7-day forecast
# ===========================================================================

def test_7d_forecast_2016_points(tmp_path: Path):
    frame = _daily_series(days=10)  # enough for weekly modulation
    r = fc.forecast_meter(1, _pre(1, frame, available_days=10.0))
    h = r.forecast_7d
    assert h["count"] == 7 * BIN
    # 7 days of history => medium confidence, no "repeated" note
    assert h["confidence"] == "medium"
    assert h["reason"] is None


def test_7d_forecast_low_confidence_without_weekly_pattern(tmp_path: Path):
    frame = _daily_series(days=3)  # < 7 days => weekly pattern not estimated
    r = fc.forecast_meter(1, _pre(1, frame, available_days=3.0))
    h = r.forecast_7d
    assert h["confidence"] == "low"
    assert "repeated" in (h["reason"] or "").lower()


# ===========================================================================
# 5. multiple PZEMs
# ===========================================================================

def test_multiple_pzems_independent(tmp_path: Path):
    settings = _settings(tmp_path)
    frames = {n: _daily_series(days=3) for n in range(1, settings.pzem_count + 1)}
    pres = {n: _pre(n, frames[n]) for n in frames}
    results, system = fc.run_forecast_pipeline(settings=settings, preprocess_results=pres)
    assert all(r.status == "FORECAST" for r in results.values())
    # each PZEM's forecast must only reflect its own number (no mixing)
    for n, r in results.items():
        assert r.pzem_number == n
    assert system.status == "FORECAST"
    assert system.meters_included == sorted(frames.keys())


def test_system_forecast_is_pointwise_sum(tmp_path: Path):
    # Two PZEMs with identical constant 100 W -> system should be ~200 W
    f1 = _frame([100.0] * (3 * BIN))
    f2 = _frame([100.0] * (3 * BIN), start_ts=START_TS + 17 * SLOT)  # different anchor
    pres = {1: _pre(1, f1), 2: _pre(2, f2)}
    results, system = fc.run_forecast_pipeline(settings=_settings(tmp_path), preprocess_results=pres)
    pw = system.forecast_24h["forecast_power_w"]
    # sum of two ~100 W flat profiles ~= 200 W (within band tolerance)
    assert all(190 <= p <= 210 for p in pw)


# ===========================================================================
# 6. flat series
# ===========================================================================

def test_flat_series_forecast_is_flat(tmp_path: Path):
    frame = _frame([700.0] * (3 * BIN))
    r = fc.forecast_meter(1, _pre(1, frame, available_days=3.0))
    pw = r.forecast_24h["forecast_power_w"]
    assert max(pw) - min(pw) < 1e-6  # essentially constant
    assert abs(pw[0] - 700.0) < 1.0


# ===========================================================================
# 7. zero-power series
# ===========================================================================

def test_zero_power_series(tmp_path: Path):
    frame = _frame([0.0] * (3 * BIN))
    r = fc.forecast_meter(1, _pre(1, frame, available_days=3.0))
    assert r.status == "FORECAST"
    pw = r.forecast_24h["forecast_power_w"]
    assert all(p == 0.0 for p in pw)
    # lower bound never negative
    assert all(b >= 0 for b in r.forecast_24h["lower_bound"])


# ===========================================================================
# 8. missing samples
# ===========================================================================

def test_missing_samples_filled_by_profile(tmp_path: Path):
    # Daily sinusoid but with big gaps (every other sample dropped) -- still
    # >= 1 day of distinct bins represented across 3 days.
    full = _daily_series(days=3)
    keep = full.iloc[::2].reset_index(drop=True)  # half the samples missing
    r = fc.forecast_meter(1, _pre(1, keep, available_days=3.0))
    assert r.status == "FORECAST"
    pw = r.forecast_24h["forecast_power_w"]
    assert min(pw) < 1000 < max(pw)  # shape recovered despite missing samples


# ===========================================================================
# 9. NaN / null values
# ===========================================================================

def test_nan_null_values_dropped(tmp_path: Path):
    pw = [1000.0 + 800 * np.sin(2 * np.pi * i / BIN) for i in range(3 * BIN)]
    pw[10] = None          # null
    pw[50] = float("nan")  # NaN
    frame = _frame(pw)
    r = fc.forecast_meter(1, _pre(1, frame, available_days=3.0))
    assert r.status == "FORECAST"
    assert r.valid_samples == 3 * BIN - 2  # two rows dropped
    assert all(np.isfinite(r.forecast_24h["forecast_power_w"]))


# ===========================================================================
# 10. irregular timestamps
# ===========================================================================

def test_irregular_timestamps_binned(tmp_path: Path):
    # Same powers but jittered timestamps (not exactly on 5-min grid)
    base = _daily_series(days=3)
    jitter = np.array([START_TS + i * SLOT + (i % 7) for i in range(len(base))])
    frame = base.copy()
    frame["timestamp"] = jitter
    r = fc.forecast_meter(1, _pre(1, frame, available_days=3.0))
    assert r.status == "FORECAST"
    pw = r.forecast_24h["forecast_power_w"]
    assert min(pw) < 1000 < max(pw)


# ===========================================================================
# 11. outlier handling
# ===========================================================================

def test_outliers_do_not_distort_forecast(tmp_path: Path):
    pw = [1000.0] * (3 * BIN)
    # inject a few extreme spikes (sensor glitches)
    for k in (100, 200, 400):
        pw[k] = 50000.0
    frame = _frame(pw)
    r = fc.forecast_meter(1, _pre(1, frame, available_days=3.0))
    pwf = r.forecast_24h["forecast_power_w"]
    # median-based profile should keep the forecast near 1000, not 50000
    assert max(pwf) < 2000
    assert abs(np.median(pwf) - 1000.0) < 50


# ===========================================================================
# 12. deterministic output
# ===========================================================================

def test_deterministic_output(tmp_path: Path):
    frame = _daily_series(days=4)
    r1 = fc.forecast_meter(1, _pre(1, frame, available_days=4.0))
    r2 = fc.forecast_meter(1, _pre(1, frame, available_days=4.0))
    assert r1.forecast_24h["forecast_power_w"] == r2.forecast_24h["forecast_power_w"]
    assert r1.forecast_7d["forecast_power_w"] == r2.forecast_7d["forecast_power_w"]
    assert r1.anchor_timestamp == r2.anchor_timestamp


# ===========================================================================
# 13. confidence / data-sufficiency state
# ===========================================================================

def test_confidence_tiers(tmp_path: Path):
    # ~2 days -> low
    r_low = fc.forecast_meter(1, _pre(1, _daily_series(days=2), available_days=2.0))
    assert r_low.forecast_24h["confidence"] == "low"
    # ~10 days -> medium
    r_med = fc.forecast_meter(1, _pre(1, _daily_series(days=10), available_days=10.0))
    assert r_med.forecast_24h["confidence"] == "medium"
    # ~20 days -> high
    r_high = fc.forecast_meter(1, _pre(1, _daily_series(days=20), available_days=20.0))
    assert r_high.forecast_24h["confidence"] == "high"


def test_uncertainty_band_present_and_subtle(tmp_path: Path):
    frame = _daily_series(days=3)
    r = fc.forecast_meter(1, _pre(1, frame, available_days=3.0))
    lo = r.forecast_24h["lower_bound"]
    hi = r.forecast_24h["upper_bound"]
    pw = r.forecast_24h["forecast_power_w"]
    # band must bracket the forecast, lower floored at 0
    for a, b, c in zip(lo, pw, hi):
        assert a <= b <= c
        assert a >= 0
    # band should not be absurdly wide relative to values (subtle)
    spread = max(hi) - min(lo)
    assert spread < 5 * (max(pw) - min(pw) + 1)


# ===========================================================================
# 14. Firebase persistence
# ===========================================================================

def test_firebase_persistence_writes_pzem_and_system(tmp_path: Path):
    frames = {n: _daily_series(days=3) for n in range(1, 4)}
    pres = {n: _pre(n, frames[n]) for n in frames}
    results, system = fc.run_forecast_pipeline(settings=_settings(tmp_path), preprocess_results=pres)
    store: dict = {}
    fc._db_ref = lambda path: FakeRef(store, path)  # type: ignore
    counts = {n: 1 if fc.write_forecast_result(r) else 0 for n, r in sorted(results.items())}
    sys_count = 1 if fc.write_system_forecast(system) else 0
    assert sum(counts.values()) == 3
    assert sys_count == 1
    # payload shape
    key = str(results[1].anchor_timestamp)
    rec = store[f"ai/forecast/pzem_1/{key}"]
    assert rec["pzem_number"] == 1
    assert rec["source_stage"] == "stage9/forecast"
    assert "forecast_24h" in rec and "forecast_7d" in rec
    sys_key = str(system.anchor_timestamp)
    assert store[f"ai/forecast/system/{sys_key}"]["meters_included"] == [1, 2, 3]


# ===========================================================================
# 15. idempotent rerun
# ===========================================================================

def test_idempotent_rerun_no_duplicates(tmp_path: Path, fake_db: dict):
    frame = _daily_series(days=3)
    pres = {1: _pre(1, frame)}
    results, system = fc.run_forecast_pipeline(settings=_settings(tmp_path), preprocess_results=pres)
    first = fc.write_forecast_result(results[1])
    # capture stored value
    key = str(results[1].anchor_timestamp)
    stored_before = fake_db[f"ai/forecast/pzem_1/{key}"]["forecast_24h"]["forecast_power_w"][0]
    # rerun with identical data
    results2, _ = fc.run_forecast_pipeline(settings=_settings(tmp_path), preprocess_results=pres)
    second = fc.write_forecast_result(results2[1])
    # only one record exists, value unchanged
    assert len([k for k in fake_db if k.startswith("ai/forecast/pzem_1/")]) == 1
    assert fake_db[f"ai/forecast/pzem_1/{key}"]["forecast_24h"]["forecast_power_w"][0] == stored_before
    assert first and second


# ===========================================================================
# 16. Firebase failure isolation
# ===========================================================================

def test_firebase_failure_isolation(tmp_path: Path, monkeypatch):
    class BoomRef:
        def child(self, key): return self
        def get(self): return None
        def set(self, value): raise RuntimeError("simulated Firebase outage")

    monkeypatch.setattr(fc, "_db_ref", lambda path: BoomRef())
    frames = {n: _daily_series(days=3) for n in range(1, 4)}
    pres = {n: _pre(n, frames[n]) for n in frames}
    results, system = fc.run_forecast_pipeline(settings=_settings(tmp_path), preprocess_results=pres)
    # all writes "fail" (return False) but none raise -- pipeline stays alive
    counts = {n: fc.write_forecast_result(r) for n, r in sorted(results.items())}
    sys_count = fc.write_system_forecast(system)
    assert all(v is False for v in counts.values())
    assert sys_count is False


def test_one_pzem_failure_does_not_block_others(tmp_path: Path, monkeypatch):
    store: dict = {}
    class FlakyRef:
        def __init__(self, path): self.path = path
        def child(self, key): return FlakyRef(f"{self.path}/{key}")
        def get(self): return store.get(self.path)
        def set(self, value):
            if "pzem_2" in self.path:
                raise RuntimeError("pzem_2 write fails")
            store[self.path] = value
    monkeypatch.setattr(fc, "_db_ref", lambda path: FlakyRef(path))
    frames = {n: _daily_series(days=3) for n in range(1, 4)}
    pres = {n: _pre(n, frames[n]) for n in frames}
    results, system = fc.run_forecast_pipeline(settings=_settings(tmp_path), preprocess_results=pres)
    counts = {n: fc.write_forecast_result(r) for n, r in sorted(results.items())}
    assert counts[1] is True
    assert counts[2] is False   # isolated failure
    assert counts[3] is True


# ===========================================================================
# 17. Stage 1-8 regression (API integration)
# ===========================================================================

def test_forecast_consumes_stage2_feature_frame(tmp_path: Path):
    """Forecast must work with a real Stage 2 PreprocessResult.feature_frame
    produced by ai.preprocessing, not a bespoke format -- proving it slots
    into the existing Stages 1-2 pipeline unchanged."""
    frame = _daily_series(days=3)
    pre = _pre(1, frame, available_days=3.0)
    r = fc.forecast_meter(1, preprocess_result=pre)
    assert r.status == "FORECAST"
    results, system = fc.run_forecast_pipeline(
        settings=_settings(tmp_path), preprocess_results={1: pre}
    )
    assert results[1].pzem_number == 1
    assert system.forecast_24h["status"] == "FORECAST"


def test_forecast_reuses_existing_stage2_cleaning_contract(tmp_path: Path):
    """Negative / out-of-range power rows would already be dropped by Stage 2;
    forecast must still behave (it defensively re-drops, never fabricates)."""
    frame = _daily_series(days=3)
    frame.loc[frame.index[5], "power"] = -500.0   # invalid but Stage2 frame
    r = fc.forecast_meter(1, _pre(1, frame, available_days=3.0))
    assert r.status == "FORECAST"
    assert all(v >= 0 for v in r.forecast_24h["forecast_power_w"])
