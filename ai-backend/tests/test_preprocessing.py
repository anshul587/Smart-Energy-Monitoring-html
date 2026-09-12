"""
Stage 2 tests for ai/preprocessing.py.

None of these touch Firebase or the network: preprocess_meter() accepts a
HistoryLoadResult directly (the same shape ai.data_loader produces), so
tests build synthetic ones instead of mocking the Firebase SDK. That keeps
these tests fast, deterministic, and independent of whatever is currently
in the real project's Firebase project.

Run with:  pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai import preprocessing as pp
from ai.config import Settings
from ai.data_loader import READING_FIELDS, HistoryLoadResult


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        firebase_service_account_path="unused-in-tests.json",
        firebase_database_url="https://example-not-real.firebasedatabase.app",
        pzem_count=9,
        history_retention_days=60,
        cache_dir=tmp_path / "cache",
    )


def make_frame(n_rows: int, start_ts: int, step_seconds: int = 300, **overrides) -> pd.DataFrame:
    """Builds n_rows of plausible, evenly-spaced readings. Any column in
    `overrides` is passed as a full-length array/scalar to override the
    default values for specific rows (see individual tests)."""
    timestamps = [start_ts + i * step_seconds for i in range(n_rows)]
    data = {
        "timestamp": timestamps,
        "voltage": np.full(n_rows, 230.0),
        "current": np.full(n_rows, 1.5),
        "power": np.full(n_rows, 300.0),
        "energy": np.linspace(10.0, 10.0 + n_rows * 0.01, n_rows),
        "frequency": np.full(n_rows, 50.0),
        "pf": np.full(n_rows, 0.95),
    }
    data.update(overrides)
    return pd.DataFrame(data, columns=["timestamp", *READING_FIELDS])


def wrap(frame: pd.DataFrame, pzem_number: int = 1) -> HistoryLoadResult:
    return HistoryLoadResult(
        pzem_number=pzem_number,
        frame=frame,
        available_days=0.0,  # not used by preprocessing; loader-level field
        requested_days=60,
        served_from_cache_only=False,
        dropped_rows=0,
        duplicate_keys_collapsed=0,
    )


# ---------------------------------------------------------------------------
# 1. Normal data
# ---------------------------------------------------------------------------

def test_normal_data_is_ready_with_full_features(settings):
    now = int(time.time())
    frame = make_frame(400, now - 400 * 300)  # ~1.4 days of clean 5-min data
    result = pp.preprocess_meter(1, settings=settings, history_result=wrap(frame))

    assert result.status == "READY"
    assert result.record_count == 400
    assert result.valid_rows == 400
    assert result.invalid_rows == 0
    assert result.duplicates_removed == 0
    assert result.missing_values == 0
    assert result.feature_frame is not None
    for col in (
        "hour_of_day", "day_of_week", "is_weekend",
        "rolling_mean_power_1h", "rolling_std_power_1h",
        "rolling_mean_power_1d", "rolling_trend_power_1d",
        "baseline_power", "deviation_power", "pct_deviation_power",
    ):
        assert col in result.feature_frame.columns
    # Baseline is the median of THIS meter's own data, not a fixed number.
    assert result.feature_frame["baseline_power"].iloc[0] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# 2. Duplicate timestamps
# ---------------------------------------------------------------------------

def test_duplicate_timestamps_are_removed_and_counted(settings):
    now = int(time.time())
    frame = make_frame(50, now - 50 * 300)
    # Three extra rows: two more copies of ts[5], one more copy of ts[20].
    dup_rows = frame.iloc[[5, 5, 20]].copy()
    frame_with_dupes = pd.concat([frame, dup_rows], ignore_index=True)

    result = pp.preprocess_meter(1, settings=settings, history_result=wrap(frame_with_dupes))
    assert result.duplicates_removed == 3  # 3 extra rows collapsed away
    assert result.valid_rows == 50  # collapsed back down to the original 50


# ---------------------------------------------------------------------------
# 3. Missing values
# ---------------------------------------------------------------------------

def test_missing_values_are_dropped_and_counted(settings):
    now = int(time.time())
    frame = make_frame(30, now - 30 * 300)
    frame.loc[3, "voltage"] = np.nan
    frame.loc[7, "pf"] = None

    result = pp.preprocess_meter(1, settings=settings, history_result=wrap(frame))
    assert result.missing_values == 2
    assert result.valid_rows == 28  # 2 rows dropped
    assert result.invalid_rows == 0


# ---------------------------------------------------------------------------
# 4. Invalid values
# ---------------------------------------------------------------------------

def test_invalid_out_of_range_values_are_dropped_and_counted(settings):
    now = int(time.time())
    frame = make_frame(30, now - 30 * 300)
    frame.loc[2, "current"] = -5.0     # negative current: implausible
    frame.loc[9, "pf"] = 1.5           # power factor must be within [-1, 1]
    frame.loc[15, "frequency"] = 5.0   # far outside 40-70 Hz sanity range

    result = pp.preprocess_meter(1, settings=settings, history_result=wrap(frame))
    assert result.invalid_rows == 3
    assert result.valid_rows == 27
    assert result.missing_values == 0


# ---------------------------------------------------------------------------
# 5. Insufficient data (some data, but below MIN_VALID_ROWS)
# ---------------------------------------------------------------------------

def test_insufficient_data_below_minimum_rows(settings):
    now = int(time.time())
    frame = make_frame(3, now - 3 * 300)  # well under MIN_VALID_ROWS (12)

    result = pp.preprocess_meter(1, settings=settings, history_result=wrap(frame))
    assert result.status == "INSUFFICIENT_DATA"
    assert result.feature_frame is None
    assert result.valid_rows == 3
    assert "at least" in result.reason.lower()


def test_ready_but_reduced_features_below_long_window_minimum(settings):
    now = int(time.time())
    # Between MIN_VALID_ROWS (12) and MIN_ROWS_FOR_LONG_WINDOW (24): READY,
    # but the ~1d rolling columns should be NaN, not fabricated.
    frame = make_frame(18, now - 18 * 300)
    result = pp.preprocess_meter(1, settings=settings, history_result=wrap(frame))
    assert result.status == "READY"
    assert result.feature_frame["rolling_mean_power_1d"].isna().all()
    assert result.reason is not None and "reduced feature set" in result.reason


# ---------------------------------------------------------------------------
# 6. One PZEM with no history at all
# ---------------------------------------------------------------------------

def test_no_history_reports_insufficient_data_with_exact_reason(settings):
    frame = pd.DataFrame(columns=["timestamp", *READING_FIELDS])
    result = pp.preprocess_meter(4, settings=settings, history_result=wrap(frame, pzem_number=4))
    assert result.status == "INSUFFICIENT_DATA"
    assert result.reason == "No usable historical data available."
    assert result.record_count == 0
    assert result.feature_frame is None


# ---------------------------------------------------------------------------
# 7. Multiple PZEMs with different amounts of history
# ---------------------------------------------------------------------------

def test_multiple_meters_with_different_history_amounts_are_independent(settings):
    now = int(time.time())
    plenty = wrap(make_frame(500, now - 500 * 300), pzem_number=1)
    a_little = wrap(make_frame(5, now - 5 * 300), pzem_number=2)
    none_at_all = wrap(pd.DataFrame(columns=["timestamp", *READING_FIELDS]), pzem_number=3)

    r1 = pp.preprocess_meter(1, settings=settings, history_result=plenty)
    r2 = pp.preprocess_meter(2, settings=settings, history_result=a_little)
    r3 = pp.preprocess_meter(3, settings=settings, history_result=none_at_all)

    assert r1.status == "READY"
    assert r2.status == "INSUFFICIENT_DATA" and r2.record_count == 5
    assert r3.status == "INSUFFICIENT_DATA" and r3.record_count == 0


# ---------------------------------------------------------------------------
# 8. All 9 PZEMs, dynamically discovered — deliberately asymmetric
#    (not 1/2/3/7 vs 4/5/6/8/9, or any other assumed pattern) to prove
#    nothing in the pipeline hardcodes which meters have data.
# ---------------------------------------------------------------------------

def test_full_fleet_dynamic_discovery_no_hardcoded_assumptions(settings, monkeypatch):
    now = int(time.time())
    # PZEM 2, 5, 9 have plenty of data; PZEM 6 has a little (insufficient);
    # everyone else has none. This mapping is intentionally different from
    # any pattern mentioned earlier in the project.
    rich = {2, 5, 9}
    sparse = {6}

    def fake_fetch_meter_history(pzem_number, settings=None, force_full_refresh=False):
        if pzem_number in rich:
            frame = make_frame(500, now - 500 * 300)
        elif pzem_number in sparse:
            frame = make_frame(4, now - 4 * 300)
        else:
            frame = pd.DataFrame(columns=["timestamp", *READING_FIELDS])
        return wrap(frame, pzem_number=pzem_number)

    monkeypatch.setattr(pp.data_loader, "fetch_meter_history", fake_fetch_meter_history)

    results = pp.run_preprocessing_pipeline(settings=settings)

    assert set(results.keys()) == set(range(1, 10))
    for n in range(1, 10):
        if n in rich:
            assert results[n].status == "READY", f"PZEM {n} should be READY"
        elif n in sparse:
            assert results[n].status == "INSUFFICIENT_DATA"
            assert results[n].record_count == 4
        else:
            assert results[n].status == "INSUFFICIENT_DATA"
            assert results[n].record_count == 0
            assert results[n].reason == "No usable historical data available."

    # The report renders all 9, in order, without assuming which have data.
    report = pp.format_report(results)
    for n in range(1, 10):
        assert f"PZEM {n}\n" in report


def test_one_meter_raising_unexpectedly_does_not_block_the_others(settings, monkeypatch):
    now = int(time.time())
    good_frame = make_frame(500, now - 500 * 300)

    def flaky_fetch(pzem_number, settings=None, force_full_refresh=False):
        if pzem_number == 7:
            raise RuntimeError("simulated unexpected failure")
        return wrap(good_frame, pzem_number=pzem_number)

    monkeypatch.setattr(pp.data_loader, "fetch_meter_history", flaky_fetch)
    results = pp.run_preprocessing_pipeline(settings=settings)

    assert results[7].status == "INSUFFICIENT_DATA"
    assert "Data loading failed" in results[7].reason
    for n in range(1, 10):
        if n != 7:
            assert results[n].status == "READY"


# ---------------------------------------------------------------------------
# Regression: the real-Firebase bug.
#
# ai.data_loader.fetch_meter_history() concatenates a meter's on-disk cache
# with freshly-fetched rows via pd.concat(). The FIRST time a meter is ever
# cached, the "cached" side of that concat is an empty
# pd.DataFrame(columns=[...]) — object dtype, since there's no data yet to
# infer float64 from. Concatenating that with a real float64 frame silently
# downgrades the WHOLE combined frame to object dtype (values are untouched
# — still real floats — but the dtype metadata is wrong). That object-dtype
# frame is exactly what gets cached to disk and handed to preprocessing.
#
# np.isfinite() refuses to run on an object-dtype array even when every
# element is a plain float, which is exactly the real error seen against
# production Firebase data:
#   ufunc 'isfinite' not supported for the input types
# ---------------------------------------------------------------------------

def test_object_dtype_frame_from_upstream_concat_is_handled(settings):
    """Reproduces the exact upstream scenario (empty object-dtype cache
    concatenated with real float64 data) and confirms preprocessing no
    longer raises, still finds the data valid, and still reports the
    correct record/valid counts."""
    now = int(time.time())

    empty_cached = pd.DataFrame(columns=["timestamp", *READING_FIELDS])  # object dtype
    real_rows = [
        {"timestamp": now - (30 - i) * 300, "voltage": 230.0 + i * 0.1, "current": 1.2,
         "power": 276.0, "energy": 10.0 + i * 0.01, "frequency": 50.0, "pf": 0.95}
        for i in range(30)
    ]
    real_frame = pd.DataFrame(real_rows, columns=["timestamp", *READING_FIELDS])  # float64

    combined = pd.concat([empty_cached, real_frame], ignore_index=True)
    assert combined["voltage"].dtype == object  # confirms this actually reproduces the bug
    assert isinstance(combined["voltage"].iloc[0], float)  # but the values are still real floats

    result = pp.preprocess_meter(1, settings=settings, history_result=wrap(combined))

    assert result.status == "READY"  # must not raise, and must recognize the real data
    assert result.record_count == 30
    assert result.valid_rows == 30
    assert result.invalid_rows == 0
    assert result.feature_frame is not None
    assert result.feature_frame["baseline_power"].iloc[0] == pytest.approx(276.0)


def test_object_dtype_with_genuinely_bad_values_counts_them_as_invalid_not_fabricated(settings):
    """Same object-dtype scenario, but one row has a value that truly
    isn't numeric (e.g. a corrupted/legacy write) mixed in among real
    floats. That row must be dropped and counted as invalid — never
    silently coerced to 0 or any other fabricated number, and never
    allowed to crash the rest of the meter's processing."""
    now = int(time.time())
    rows = [
        {"timestamp": now - (20 - i) * 300, "voltage": 230.0, "current": 1.2,
         "power": 276.0, "energy": 10.0 + i * 0.01, "frequency": 50.0, "pf": 0.95}
        for i in range(20)
    ]
    frame = pd.DataFrame(rows, columns=["timestamp", *READING_FIELDS])
    # Simulate a corrupted/legacy field value slipping through as a
    # non-numeric string inside an otherwise-numeric object-dtype column.
    frame["voltage"] = frame["voltage"].astype(object)
    frame.loc[5, "voltage"] = "ERR"

    empty_cached = pd.DataFrame(columns=["timestamp", *READING_FIELDS])
    combined = pd.concat([empty_cached, frame], ignore_index=True)

    result = pp.preprocess_meter(1, settings=settings, history_result=wrap(combined))

    assert result.status == "READY"
    assert result.record_count == 20
    assert result.invalid_rows == 1       # the "ERR" row, counted as invalid
    assert result.valid_rows == 19        # everything else survived
    assert result.missing_values == 0     # this was a type problem, not a missing value
    # The dropped row's real value must never appear anywhere as a
    # fabricated substitute (e.g. 0.0) in the surviving valid data.
    assert "ERR" not in result.feature_frame["voltage"].astype(str).values


def test_exception_during_cleaning_preserves_real_record_count(settings, monkeypatch):
    """Regression for the reporting half of the bug: if something inside
    preprocessing still manages to raise, the result must report the ACTUAL
    number of records the loader returned — never silently fall back to
    "Records: 0" and imply the meter had no data when it actually did."""
    now = int(time.time())
    frame = make_frame(42, now - 42 * 300)

    def broken_clean(frame):
        raise TypeError("simulated: ufunc 'isfinite' not supported for the input types")

    monkeypatch.setattr(pp, "_clean_frame", broken_clean)
    result = pp.preprocess_meter(1, settings=settings, history_result=wrap(frame))

    assert result.status == "INSUFFICIENT_DATA"
    assert result.record_count == 42  # NOT 0 — the loader really did return 42 records
    assert "Preprocessing error while cleaning data" in result.reason
    assert result.debug_traceback is not None
    assert "isfinite" in result.debug_traceback


def test_exception_during_feature_building_preserves_full_context(settings, monkeypatch):
    """Same idea, but for a failure after cleaning succeeds — record_count,
    valid_rows, invalid_rows, duplicates_removed, and missing_values (all
    already known at that point) must all still be reported accurately."""
    now = int(time.time())
    frame = make_frame(50, now - 50 * 300)

    def broken_features(cleaned):
        raise ValueError("simulated feature-engineering failure")

    monkeypatch.setattr(pp, "_build_features", broken_features)
    result = pp.preprocess_meter(1, settings=settings, history_result=wrap(frame))

    assert result.status == "INSUFFICIENT_DATA"
    assert result.record_count == 50
    assert result.valid_rows == 50
    assert result.invalid_rows == 0
    assert result.duplicates_removed == 0
    assert result.missing_values == 0
    assert "Preprocessing error while building features" in result.reason
    assert result.debug_traceback is not None
