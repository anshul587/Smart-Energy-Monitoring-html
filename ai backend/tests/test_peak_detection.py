"""
tests/test_peak_detection.py
----------------------------
STAGE 7: Peak Load Detection tests.

All fixtures are DETERMINISTIC SYNTHETIC data (fixed timestamps, fixed
values, seed-free) used ONLY inside this module — clearly labeled test
data, never persisted anywhere real. Firebase is fully mocked.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from ai import peak_detection as pk
from ai.config import Settings
from ai.preprocessing import PreprocessResult
from tests.test_anomaly_detection import _make_frame, _settings

START_TS = 1_700_000_000
SLOT = 300


# ---------------------------------------------------------------------------
# Deterministic synthetic fixtures (test data only)
# ---------------------------------------------------------------------------

def _frame(powers, start_ts: int = START_TS, slot: int = SLOT) -> pd.DataFrame:
    """Minimal PZEM-shaped frame; only timestamp/power affect Stage 7."""
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


def _pre(pzem_number: int, frame: Optional[pd.DataFrame]) -> PreprocessResult:
    """A minimal READY PreprocessResult carrying a handcrafted frame.
    Stage 7 only consumes feature_frame, so the metadata here is inert."""
    ready = frame is not None
    return PreprocessResult(
        pzem_number=pzem_number,
        status="READY" if ready else "INSUFFICIENT_DATA",
        reason=None if ready else "synthetic no-data",
        record_count=0 if frame is None else len(frame),
        oldest_timestamp=None,
        newest_timestamp=None,
        available_days=0.0,
        valid_rows=0 if frame is None else len(frame),
        invalid_rows=0,
        duplicates_removed=0,
        missing_values=0,
        feature_frame=frame,
    )


def _ts(index: int, start_ts: int = START_TS, slot: int = SLOT) -> int:
    return start_ts + index * slot


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
    monkeypatch.setattr(pk, "_db_ref", lambda path: FakeRef(store, path))
    return store


def _threshold_settings(tmp_path: Path, threshold: float) -> Settings:
    return replace(_settings(tmp_path), peak_power_threshold_w=threshold)


# ===========================================================================
# 1. normal power series
# ===========================================================================

def test_normal_power_series_full_stats(tmp_path: Path):
    r = pk.detect_peak_for_meter(1, _pre(1, _frame([50, 55, 60, 58, 52])), _settings(tmp_path))
    assert r.status == "PEAK_FOUND"
    assert r.peak_power_w == 60.0
    assert r.peak_timestamp == _ts(2)
    assert r.average_power_w == pytest.approx(55.0)
    assert r.baseline_power_w == pytest.approx(55.0)          # median convention
    assert r.peak_above_baseline_w == pytest.approx(5.0)
    assert r.samples_analyzed == 5
    assert r.isolated_outliers_dropped == 0


# ===========================================================================
# 2. single clear peak
# ===========================================================================

def test_single_clear_peak_detected(tmp_path: Path):
    powers = [40.0] * 8 + [260.0] + [40.0] * 8
    r = pk.detect_peak_for_meter(1, _pre(1, _frame(powers)), _settings(tmp_path))
    assert r.status == "PEAK_FOUND"
    assert r.peak_power_w == 260.0
    assert r.peak_timestamp == _ts(8)
    # One lone sample cannot prove sustainment at the 5-min interval.
    assert r.sustained is False
    assert r.peak_duration_seconds == 0


# ===========================================================================
# 3. multiple peaks -> global maximum wins
# ===========================================================================

def test_multiple_peaks_selects_global_maximum(tmp_path: Path):
    powers = [40.0] * 4 + [200.0] + [40.0] * 4 + [150.0] + [40.0] * 4
    r = pk.detect_peak_for_meter(1, _pre(1, _frame(powers)), _settings(tmp_path))
    assert r.status == "PEAK_FOUND"
    assert r.peak_power_w == 200.0
    assert r.peak_timestamp == _ts(4)


# ===========================================================================
# 4. equal maximum values -> earliest timestamp, single event
# ===========================================================================

def test_equal_maxima_pick_earliest_timestamp_one_event(tmp_path: Path):
    powers = [40.0] * 3 + [90.0] + [40.0] * 3 + [90.0] + [40.0] * 3
    r = pk.detect_peak_for_meter(1, _pre(1, _frame(powers)), _settings(tmp_path))
    assert r.status == "PEAK_FOUND"
    assert r.peak_power_w == 90.0
    assert r.peak_timestamp == _ts(3), "ties must resolve to the EARLIEST sample"


# ===========================================================================
# 5. sustained peak / duration in slot multiples
# ===========================================================================

def test_sustained_peak_duration_is_slot_multiples(tmp_path: Path):
    powers = [40.0] * 5 + [260.0, 270.0, 265.0] + [40.0] * 5
    r = pk.detect_peak_for_meter(1, _pre(1, _frame(powers)), _settings(tmp_path))
    assert r.status == "PEAK_FOUND"
    assert r.peak_power_w == 270.0
    assert r.sustained is True
    assert r.peak_duration_seconds == 600           # run of 3 samples spans 2 links x 300 s
    assert r.peak_duration_seconds % SLOT == 0      # never sub-interval precision


# ===========================================================================
# 6. isolated outlier is NOT crowned the operational peak
# ===========================================================================

def test_isolated_outlier_rejected_but_recorded(tmp_path: Path):
    powers = [50.0] * 10 + [5000.0] + [50.0] * 10
    r = pk.detect_peak_for_meter(1, _pre(1, _frame(powers)), _settings(tmp_path))
    assert r.status == "PEAK_FOUND"
    assert r.peak_power_w == 50.0, "a 100x-median lone spike is a glitch, not THE peak"
    assert r.isolated_outliers_dropped == 1
    assert r.dropped_outlier_power_w == 5000.0
    assert r.dropped_outlier_timestamp == _ts(10)


def test_short_real_load_event_stays_a_peak(tmp_path: Path):
    # Lonely but NOT extreme (< 10x median): a legitimate short load.
    powers = [50.0] * 6 + [400.0] + [50.0] * 6
    r = pk.detect_peak_for_meter(1, _pre(1, _frame(powers)), _settings(tmp_path))
    assert r.status == "PEAK_FOUND"
    assert r.peak_power_w == 400.0
    assert r.isolated_outliers_dropped == 0


# ===========================================================================
# 7. + 8. missing values and NaN values
# ===========================================================================

@pytest.mark.parametrize("hole", [None, np.nan])
def test_missing_and_nan_values_dropped_counted(tmp_path: Path, hole):
    powers = [hole, 60.0, 70.0, 80.0]
    r = pk.detect_peak_for_meter(1, _pre(1, _frame(powers)), _settings(tmp_path))
    assert r.status == "PEAK_FOUND"
    assert r.samples_analyzed == 3
    assert r.invalid_rows_dropped == 1
    assert r.peak_power_w == 80.0
    assert r.baseline_power_w == pytest.approx(70.0)
    assert r.peak_above_baseline_w == pytest.approx(10.0)


# ===========================================================================
# 9. invalid timestamps (+ negative power while we're at it)
# ===========================================================================

def test_invalid_timestamp_and_negative_power_dropped(tmp_path: Path):
    frame = pd.DataFrame({
        "timestamp": ["not-a-ts", START_TS, START_TS + SLOT, START_TS + 2 * SLOT, START_TS + 3 * SLOT],
        "voltage": [230.0] * 5,
        "current": [0.5] * 5,
        "power": [999.0, 50.0, 60.0, 70.0, -5.0],
        "energy": [1.0] * 5,
        "frequency": [50.0] * 5,
        "pf": [0.9] * 5,
    })
    r = pk.detect_peak_for_meter(1, _pre(1, frame), _settings(tmp_path))
    assert r.status == "PEAK_FOUND"
    assert r.invalid_rows_dropped == 2               # bad-ts row AND negative-power row
    assert r.peak_power_w == 70.0
    assert r.dropped_outlier_power_w is None         # neither was an outlier event


# ===========================================================================
# 10. insufficient data — nothing invented
# ===========================================================================

def test_insufficient_data_reports_no_peak(tmp_path: Path):
    r = pk.detect_peak_for_meter(1, _pre(1, _frame([100.0, 120.0])), _settings(tmp_path))
    assert r.status == "NO_PEAK"
    assert "insufficient_data" in r.reason
    assert r.peak_power_w is None and r.peak_timestamp is None
    assert r.average_power_w is None


def test_stage2_insufficient_result_gives_no_peak(tmp_path: Path):
    r = pk.detect_peak_for_meter(1, _pre(1, None), _settings(tmp_path))
    assert r.status == "NO_PEAK"
    assert r.peak_power_w is None
    assert r.reason


# ===========================================================================
# 11. zero-power series
# ===========================================================================

def test_zero_power_series_honest_zero_peak(tmp_path: Path):
    r = pk.detect_peak_for_meter(1, _pre(1, _frame([0.0] * 6)), _settings(tmp_path))
    assert r.status == "PEAK_FOUND"
    assert r.peak_power_w == 0.0
    assert r.peak_timestamp == _ts(0)
    assert r.peak_above_baseline_w == 0.0
    assert r.exceeds_threshold is None               # threshold disabled at default


# ===========================================================================
# 12. multiple PZEMs analyzed independently
# ===========================================================================

def test_multiple_pzems_independent(tmp_path: Path):
    settings = replace(_settings(tmp_path), pzem_count=3)
    pres = {
        1: _pre(1, _frame([50.0] * 5 + [100.0] + [50.0] * 5)),
        2: _pre(2, _frame([30.0] * 5 + [250.0, 245.0] + [30.0] * 5)),
        3: _pre(3, None),                        # this meter simply has nothing
    }
    results, system = pk.run_peak_detection_pipeline(settings=settings, preprocess_results=pres)
    assert results[1].peak_power_w == 100.0
    assert results[2].peak_power_w == 250.0 and results[2].sustained is True
    assert results[3].status == "NO_PEAK"
    assert system.meters_analyzed == 2           # meter 3 excluded honestly


# ===========================================================================
# 13. system-wide peak
# ===========================================================================

def test_system_wide_peak_sum_and_dominant(tmp_path: Path):
    m1 = _frame([100.0, 150.0, 120.0])
    m2 = _frame([80.0, 90.0, 70.0], start_ts=START_TS + 10)   # offset < slot -> same buckets
    pres = {1: _pre(1, m1), 2: _pre(2, m2)}
    _, system = pk.run_peak_detection_pipeline(
        settings=replace(_settings(tmp_path), pzem_count=3),
        preprocess_results={**pres, 3: _pre(3, None)},
    )
    assert system.status == "PEAK_FOUND"
    assert system.total_peak_power_w == pytest.approx(150.0 + 90.0)   # slot 2 sum
    # System timestamps are 300 s SLOT STARTS, not raw sample timestamps.
    assert system.timestamp == (_ts(1) // SLOT) * SLOT
    assert system.dominant_pzems == [1]
    assert system.per_pzem_power_w == {"pzem_1": 150.0, "pzem_2": 90.0}


def test_system_peak_tie_lists_all_dominant_sorted(tmp_path: Path):
    m1 = _frame([100.0, 100.0, 100.0])
    m2 = _frame([100.0, 100.0, 100.0])
    _, system = pk.run_peak_detection_pipeline(
        settings=replace(_settings(tmp_path), pzem_count=2),
        preprocess_results={1: _pre(1, m1), 2: _pre(2, m2)},
    )
    assert system.status == "PEAK_FOUND"
    assert system.total_peak_power_w == pytest.approx(200.0)
    assert system.dominant_pzems == [1, 2]


def test_system_peak_without_common_slots_is_honest(tmp_path: Path):
    m1 = _frame([100.0, 110.0, 120.0], start_ts=START_TS)
    m2 = _frame([200.0, 210.0, 220.0], start_ts=START_TS + 10_000)   # disjoint window
    _, system = pk.run_peak_detection_pipeline(
        settings=replace(_settings(tmp_path), pzem_count=2),
        preprocess_results={1: _pre(1, m1), 2: _pre(2, m2)},
    )
    assert system.status == "NO_PEAK"
    assert "common" in system.reason
    assert system.total_peak_power_w is None


# ===========================================================================
# 14. deterministic timestamp behavior across reruns
# ===========================================================================

def test_rerun_is_bitwise_deterministic(tmp_path: Path):
    frame = _frame([40.0] * 3 + [90.0] + [40.0] * 3 + [90.0] + [40.0] * 3)
    r1 = pk.detect_peak_for_meter(1, _pre(1, frame), _settings(tmp_path))
    r2 = pk.detect_peak_for_meter(1, _pre(1, frame), _settings(tmp_path))
    assert r1.peak_power_w == r2.peak_power_w
    assert r1.peak_timestamp == r2.peak_timestamp
    assert r1.peak_duration_seconds == r2.peak_duration_seconds
    assert r1.average_power_w == r2.average_power_w
    assert pk.peak_payload(r1) == pk.peak_payload(r2)


# ===========================================================================
# 15. threshold behavior (annotation only, separate from Stage 4 faults)
# ===========================================================================

def test_threshold_disabled_by_default(tmp_path: Path):
    r = pk.detect_peak_for_meter(1, _pre(1, _frame([50.0, 300.0, 60.0])), _settings(tmp_path))
    assert r.threshold_w == 0.0
    assert r.exceeds_threshold is None
    assert r.peak_above_threshold_w is None


def test_threshold_annotation_on_and_off(tmp_path: Path):
    frame = _frame([50.0, 300.0, 60.0])
    high = pk.detect_peak_for_meter(1, _pre(1, frame), _threshold_settings(tmp_path, 250.0))
    assert high.exceeds_threshold is True
    assert high.peak_above_threshold_w == pytest.approx(50.0)

    low = pk.detect_peak_for_meter(1, _pre(1, _frame([50.0, 60.0, 55.0])),
                                   _threshold_settings(tmp_path, 250.0))
    assert low.status == "PEAK_FOUND"             # threshold never gates detection
    assert low.exceeds_threshold is False
    assert low.peak_above_threshold_w == pytest.approx(-190.0)


# ===========================================================================
# 16. duplicate / idempotent execution against mocked Firebase
# ===========================================================================

def test_rerun_same_analysis_creates_no_duplicates(fake_db: dict, tmp_path: Path):
    r = pk.detect_peak_for_meter(1, _pre(1, _frame([50.0, 300.0, 60.0])), _settings(tmp_path))
    assert pk.write_peak_result(r) is True
    assert pk.write_peak_result(r) is True       # same data -> same key -> skip
    assert len(fake_db) == 1
    (record,) = fake_db.values()
    assert record["peak_power_w"] == 300.0
    assert record["source_stage"] == "stage7/peak_detection"

    _, system = pk.run_peak_detection_pipeline(
        settings=replace(_settings(tmp_path), pzem_count=1),
        preprocess_results={1: _pre(1, _frame([50.0, 300.0, 60.0]))},
    )
    assert pk.write_system_peak(system) is True
    assert pk.write_system_peak(system) is True
    system_keys = [k for k in fake_db if k.startswith("ai/peaks/system")]
    assert len(system_keys) == 1                 # still no duplicates anywhere


def test_payload_is_json_safe(fake_db: dict, tmp_path: Path):
    r = pk.detect_peak_for_meter(1, _pre(1, _frame([50.0, 300.0, 60.0])), _settings(tmp_path))
    payload = pk.peak_payload(r)
    json.dumps(payload)                          # raises on numpy types / NaN
    assert isinstance(payload["peak_power_w"], float)
    assert isinstance(payload["timestamp"], int)


# ===========================================================================
# 17. Firebase write failure is graceful
# ===========================================================================

def test_firebase_write_failure_returns_false(monkeypatch, tmp_path: Path):
    class ExplodingRef(FakeRef):
        def get(self):
            raise ConnectionError("firebase unreachable")

        def set(self, value):
            raise ConnectionError("firebase unreachable")

    monkeypatch.setattr(pk, "_db_ref", lambda path: ExplodingRef({}, path))
    r = pk.detect_peak_for_meter(
        1,
        _pre(1, _frame([50.0] * 12 + [400.0] * 2)),
        _threshold_settings(tmp_path, 350.0),
    )
    assert pk.write_peak_result(r) is False      # logged, not raised
    sys_res = pk.SystemPeakResult(
        status="PEAK_FOUND",
        total_peak_power_w=850.0,
        timestamp=START_TS,
        dominant_pzems=[1],
        per_pzem_power_w={"pzem_1": 850.0},
        meters_analyzed=1,
        threshold_w=350.0,
        exceeds_threshold=True,
    )
    assert pk.write_system_peak(sys_res) is False


# ===========================================================================
# 18. Stage 1–6 regression smoke (full suite runs separately)
# ===========================================================================

def test_stage7_consumes_real_stage2_output(tmp_path: Path):
    """End-to-end over the REAL Stage 1->2 code path (synthetic input data):
    preprocessing.preprocess_meter -> Stage 7 detects the planted peak."""
    from ai import preprocessing

    settings = _settings(tmp_path)
    frame = _make_frame(30, power_fn=lambda i: 400.0 if i == 15 else 50.0)
    hist_pk = replace(_make_history_like(frame), pzem_number=7)
    pre = preprocessing.preprocess_meter(7, settings=settings, history_result=hist_pk)
    assert pre.status == "READY"

    r = pk.detect_peak_for_meter(7, pre, settings)
    assert r.status == "PEAK_FOUND"
    assert r.peak_power_w == 400.0
    assert r.peak_timestamp == frame["timestamp"].iloc[15]
    assert r.samples_analyzed == 30


def _make_history_like(frame: pd.DataFrame):
    from ai.data_loader import HistoryLoadResult
    return HistoryLoadResult(
        pzem_number=1,
        frame=frame,
        available_days=round((frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0]) / 86400, 2),
        requested_days=60,
        served_from_cache_only=False,
        dropped_rows=0,
        duplicate_keys_collapsed=0,
    )


def test_existing_stage_modules_unaffected_imports(tmp_path: Path):
    """Stage 1–6 public entry points still import and exist alongside Stage 7."""
    from ai import anomaly_detection, config, data_loader, fault_diagnosis, persist_ai_results, peak_detection, preprocessing  # noqa: F401

    assert callable(data_loader.fetch_meter_history)
    assert callable(preprocessing.preprocess_meter)
    assert callable(anomaly_detection.run_anomaly_detection_pipeline)
    assert callable(fault_diagnosis.run_fault_diagnosis_pipeline)
    assert callable(persist_ai_results.run_stage_5_pipeline)
    assert callable(peak_detection.run_stage_7_pipeline)
    # Stage 4 fault thresholds untouched by Stage 7.
    assert fault_diagnosis.FAULT_HIGH_POWER_W == 5000.0


def test_format_report_renders_peaks_and_system(tmp_path: Path):
    results, system = pk.run_peak_detection_pipeline(
        settings=replace(_settings(tmp_path), pzem_count=2),
        preprocess_results={
            1: _pre(1, _frame([50.0, 300.0, 60.0])),
            2: _pre(2, None),
        },
    )
    text = pk.format_report(results, system)
    assert "PZEM 1" in text and "PZEM 2" in text
    assert "SYSTEM-WIDE PEAK" in text
    assert "Status: PEAK_FOUND" in text
