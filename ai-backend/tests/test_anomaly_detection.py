# Implement Stage 9: Forecasting + Dashboard Integration.
#
# IMPORTANT:
# - Stage 1–8 are already implemented and working.
# - Preserve all existing functionality.
# - Do not rewrite previous stages.
# - Do not modify ESP32 firmware.
# - Do not modify existing Firebase data contracts:
#   /meters
#   /history
#   /alerts
#   /ai/anomalies
#   /ai/faults
#   /ai/peaks
#   /ai/maintenance
# - Do not modify credentials, authentication, or security settings.
# - Do not redesign the dashboard.
# - Keep the existing dashboard layout, cards, colors, graphs, themes, selector, zoom/pan, alerts, and responsive behavior.
# - First inspect the current AI pipeline and dashboard before making changes.
#
# ==================================================
# 1. FORECASTING GOAL
# ==================================================
#
# Build Stage 9 forecasting using the existing 5-minute historical power data.
#
# Provide:
#
# - per-PZEM 24-hour power forecast
# - per-PZEM 7-day power forecast
# - system-level forecast where sufficient data exists
# - forecast confidence / data sufficiency state
#
# Use the existing history/data-loading architecture.
#
# Do not create a second Firebase data-loading pipeline.
#
# ==================================================
# 2. FORECASTING METHOD
# ==================================================
#
# Use an explainable time-series forecasting approach appropriate for the available data.
#
# Prefer the existing project plan's forecasting approach where already specified.
#
# The forecast must use real historical observations.
#
# Do not generate fake production forecasts.
#
# Handle:
# - missing samples
# - irregular timestamps
# - NaN/null values
# - insufficient history
# - zero/near-zero series
# - flat series
# - outliers
#
# Do not claim high confidence when data is insufficient.
#
# ==================================================
# 3. DATA SUFFICIENCY
# ==================================================
#
# The project does not yet have the planned 30-day real dataset.
#
# Therefore:
#
# - implement the complete forecasting pipeline now
# - support deterministic synthetic fixtures for testing only
# - clearly distinguish test data from real production data
# - provide NO_FORECAST / INSUFFICIENT_DATA when history is insufficient
# - do not pretend synthetic test accuracy is production accuracy
#
# Document the minimum data required for:
# - 24-hour forecast
# - 7-day forecast
# - high-confidence forecast
#
# Do not invent unexplained requirements.
#
# ==================================================
# 4. FORECAST OUTPUT
# ==================================================
#
# For each PZEM, produce forecast points containing:
#
# - pzem_number
# - timestamp
# - forecast_power_w
# - lower_bound if supported
# - upper_bound if supported
# - confidence/status
# - source_stage
#
# Use Unix-second timestamps consistent with the existing project.
#
# Preserve historical timestamps and units.
#
# ==================================================
# 5. FIREBASE PERSISTENCE
# ==================================================
#
# Store Stage 9 output under a dedicated namespace:
#
# /ai/forecast/pzem_N/<forecast_timestamp>
#
# /ai/forecast/system/<forecast_timestamp>
#
# Do not modify previous AI namespaces.
#
# Use deterministic keys and idempotent writes.
#
# Rerunning the same historical input must not create uncontrolled duplicates.
#
# Handle Firebase failures safely.
#
# One PZEM failure must not stop all other PZEM forecasts.
#
# ==================================================
# 6. DASHBOARD INTEGRATION
# ==================================================
#
# Integrate forecasting into the EXISTING dashboard without redesigning it.
#
# Add a compact Forecast section/panel using the existing visual language.
#
# Show:
#
# - Forecast horizon selector:
#   24h
#   7d
#
# - forecast chart
# - actual historical data + forecast continuation
# - PZEM selector where appropriate
# - confidence/data-sufficiency state
#
# For insufficient data:
#
# - show "Forecast unavailable" / "Insufficient data"
# - do not show fake forecast lines
# - explain briefly why the forecast is unavailable
#
# Do not replace existing graphs.
#
# Do not disturb:
# - Real-Time Power graph
# - Common Frequency graph
# - PZEM modal graph
# - AI anomaly/fault indicators
# - existing alerts
#
# ==================================================
# 7. FORECAST CHART
# ==================================================
#
# Use the existing Chart.js ecosystem.
#
# Show clearly:
#
# Historical actual data
# ───────────────
#                   ╲
#                    ╲ forecast
#
# Differentiate actual vs forecast clearly without changing the dashboard's overall color language.
#
# If confidence bounds are supported:
# - display a subtle uncertainty band
# - do not exaggerate precision
#
# Preserve:
# - zoom
# - pan
# - reset
# - dynamic axis precision
#
# ==================================================
# 8. PZEM ASSOCIATION
# ==================================================
#
# Ensure:
#
# pzem_1 → PZEM 1
# ...
# pzem_9 → PZEM 9
#
# Never mix forecasts between meters.
#
# System forecast must use the correct system aggregation method and only include valid available meters.
#
# ==================================================
# 9. REAL-TIME / HISTORICAL SEPARATION
# ==================================================
#
# Do not mix:
# - 10-second live readings
# - 5-minute historical samples
# - forecast points
#
# Forecasting should be based on the historical series and clearly separated from live actual values.
#
# Do not alter:
# - 10-second live cycle
# - 5-minute history cycle
# - 60-day retention
#
# ==================================================
# 10. TESTS
# ==================================================
#
# Add deterministic tests for:
#
# 1. sufficient history
# 2. insufficient history
# 3. 24-hour forecast
# 4. 7-day forecast
# 5. multiple PZEMs
# 6. flat series
# 7. zero-power series
# 8. missing samples
# 9. NaN/null values
# 10. irregular timestamps
# 11. outlier handling
# 12. deterministic output
# 13. confidence/data-sufficiency state
# 14. Firebase persistence
# 15. idempotent rerun
# 16. Firebase failure isolation
# 17. Stage 1–8 regression
#
# Run the complete AI test suite.
#
# ==================================================
# 11. DASHBOARD LIVE VERIFICATION
# ==================================================
#
# After implementation, use:
#
# - Playwright MCP
# - Chrome DevTools MCP
#
# Open the actual dashboard over HTTP.
#
# Verify:
#
# - dashboard loads normally
# - existing 9 PZEM cards still work
# - existing graphs still work
# - new Forecast section renders
# - 24h selector works
# - 7d selector works
# - PZEM selection works
# - forecast chart renders when valid forecast data exists
# - insufficient-data state renders correctly
# - no fake forecast is shown
# - zoom works
# - pan works
# - reset works
# - dark theme works
# - light theme works
# - responsive layout works
#
# ==================================================
# 12. CHROME DEVTOOLS CHECK
# ==================================================
#
# Check the actual browser for:
#
# - SyntaxError
# - ReferenceError
# - TypeError
# - Chart.js errors
# - Firebase errors
# - failed network requests
# - duplicate listeners
# - runtime errors
# - layout/overflow problems
#
# There must be no new uncaught errors.
#
# ==================================================
# 13. REGRESSION CHECK
# ==================================================
#
# Confirm Stage 1–8 remain intact:
#
# - data loading
# - preprocessing
# - anomaly detection
# - fault diagnosis
# - AI persistence
# - dashboard AI integration
# - peak detection
# - maintenance risk
# - ESP32 functionality
# - 10-second live update
# - 5-minute history
# - 60-day retention
# - alerts
# - buzzer/red-blink behavior
# - existing graph behavior
#
# ==================================================
# 14. BUG-FIX RULE
# ==================================================
#
# If a bug is found:
#
# 1. identify root cause
# 2. fix it
# 3. rerun affected tests
# 4. rerun regression tests
# 5. rerun Playwright
# 6. rerun Chrome DevTools checks
#
# Do not stop at reporting a fixable bug.
#
# Do not modify unrelated functionality.
#
# ==================================================
# 15. FINAL REPORT
# ==================================================
#
# Return:
#
# 1. Files modified
# 2. Forecasting method
# 3. Data sufficiency requirements
# 4. Forecast schema
# 5. Firebase paths
# 6. Idempotency strategy
# 7. Dashboard changes
# 8. Tests passed/failed
# 9. Stage 1–8 regression result
# 10. Playwright result
# 11. Chrome DevTools result
# 12. Current limitations due to lack of real 30-day data
# 13. Parameters requiring real-data validation
#
# IMPORTANT:
# Do not claim final forecast accuracy.
# Real 30-day data must later be used for validation and tuning."""
# tests/test_anomaly_detection.py
# --------------------------------
# STAGE 3 tests (revised for data-driven operating-state detection). All
# data here is SYNTHETIC and only ever used inside this test module —
# nothing here touches Firebase or writes synthetic data anywhere real.
#
# Strategy: build a synthetic ai.data_loader.HistoryLoadResult (raw
# readings) with realistic ACTIVE/IDLE power patterns across one or more
# CALENDAR DAYS at VARYING, NON-FIXED start/stop times (mirroring a
# classroom with no fixed operating schedule), run it through the REAL
# ai.preprocessing.preprocess_meter() to get a REAL Stage 2 feature frame,
# then run the REAL ai.anomaly_detection functions on that.
# """

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from ai import anomaly_detection as ad
from ai import preprocessing
from ai.config import Settings
from ai.data_loader import READING_FIELDS, HistoryLoadResult

START_TS = 1_700_000_000  # arbitrary fixed epoch anchor, deterministic
SLOT_SECONDS = 300        # matches the firmware's 5-minute history cadence
DAY_ROWS = 288             # 5-minute slots in a day — the METER reports every
                            # slot regardless of whether the classroom is
                            # active; "day" here is just how many readings
                            # accumulate per 24h, NOT an assumed schedule.


def _settings(tmp_path: Path, pzem_count: int = 9) -> Settings:
    """A Settings instance that never touches real env vars / Firebase —
    every field is supplied explicitly, bypassing the _require() lookups."""
    return Settings(
        firebase_service_account_path="unused-in-tests.json",
        firebase_database_url="https://unused-in-tests.example/",
        pzem_count=pzem_count,
        history_retention_days=60,
        cache_dir=tmp_path,
        anthropic_api_key="",
    )


def _history_result(pzem_number: int, frame: pd.DataFrame) -> HistoryLoadResult:
    return HistoryLoadResult(
        pzem_number=pzem_number,
        frame=frame,
        available_days=round(
            ((frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0]) / 86400)
            if not frame.empty else 0.0,
            2,
        ),
        requested_days=60,
        served_from_cache_only=False,
        dropped_rows=0,
        duplicate_keys_collapsed=0,
    )


def _make_frame(n_rows: int, power_fn=None, seed: int = 0) -> pd.DataFrame:
    """Original Stage-1/2-shaped fixture generator: n_rows of plausible,
    physically-valid PZEM readings at the firmware's 5-minute cadence,
    power shaped by power_fn(i). Kept for tests about the ML pipeline
    itself (feature exclusion, reproducibility, non-finite handling)
    where the operating-state/active-day mechanics aren't the point."""
    rng = np.random.default_rng(seed)
    timestamps = [START_TS + i * SLOT_SECONDS for i in range(n_rows)]

    if power_fn is None:
        power = np.array([
            50.0 + 20.0 * np.sin(2 * np.pi * (i % 288) / 288.0) for i in range(n_rows)
        ]) + rng.normal(0, 1.0, n_rows)
    else:
        power = np.array([power_fn(i) for i in range(n_rows)], dtype="float64")

    power = np.clip(power, 1.0, None)
    voltage = 230.0 + rng.normal(0, 0.5, n_rows)
    pf = np.clip(0.9 + rng.normal(0, 0.02, n_rows), 0.5, 1.0)
    current = power / (voltage * pf)
    frequency = 50.0 + rng.normal(0, 0.05, n_rows)
    energy = np.cumsum(power) / 1000.0 / 12.0

    return pd.DataFrame({
        "timestamp": timestamps,
        "voltage": voltage,
        "current": current,
        "power": power,
        "energy": energy,
        "frequency": frequency,
        "pf": pf,
    })


def _make_classroom_frame(
    day_active_windows: list[list[tuple[int, int]]],
    idle_power: float = 5.0,
    active_power: float = 60.0,
    idle_noise: float = 0.5,
    active_noise: float = 3.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Builds a multi-day frame representing a classroom with NO fixed
    operating schedule. `day_active_windows` is one entry per calendar
    day: a list of (start_row, end_row) tuples, row indices LOCAL to
    that day (0..287, i.e. that day's own 5-minute slots), marking
    windows where the classroom is genuinely drawing active load. Rows
    outside every window for a day are idle/standby (idle_power > 0 —
    never exactly zero, to exercise "standby load exists" explicitly).
    An empty window list for a day means that day is entirely unused.

    Different days may start/stop active windows at completely different
    row offsets, have zero, one, or several windows, or none at all —
    exactly the "no fixed clock schedule" scenario this revision targets.
    """
    rng = np.random.default_rng(seed)
    rows = []
    ts = START_TS
    for day_windows in day_active_windows:
        for i in range(DAY_ROWS):
            is_active = any(s <= i < e for (s, e) in day_windows)
            if is_active:
                power = max(active_power + rng.normal(0, active_noise), 0.5)
            else:
                power = max(idle_power + rng.normal(0, idle_noise), 0.1)
            voltage = 230.0 + rng.normal(0, 0.5)
            pf = float(np.clip(0.9 + rng.normal(0, 0.02), 0.5, 1.0))
            current = power / (voltage * pf)
            frequency = 50.0 + rng.normal(0, 0.05)
            rows.append({
                "timestamp": ts,
                "voltage": voltage,
                "current": current,
                "power": power,
                "frequency": frequency,
                "pf": pf,
            })
            ts += SLOT_SECONDS
    frame = pd.DataFrame(rows)
    frame["energy"] = np.cumsum(frame["power"].to_numpy()) / 1000.0 / 12.0
    return frame[["timestamp", "voltage", "current", "power", "energy", "frequency", "pf"]]


def _preprocess(pzem_number: int, frame: pd.DataFrame, settings: Settings):
    hist = _history_result(pzem_number, frame)
    return preprocessing.preprocess_meter(pzem_number, settings=settings, history_result=hist)


# ===========================================================================
# 1-6, 9: operating-state detection under variable/no-fixed-schedule conditions
# ===========================================================================

def test_variable_operating_hours_are_detected_as_active(tmp_path):
    """Day 1: 09:00-13:00 active (rows 108-156). No fixed clock schedule
    is assumed anywhere in the detector — it must find this window from
    the electrical behavior alone."""
    settings = _settings(tmp_path)
    frame = _make_classroom_frame([[(108, 156)]], seed=1)
    pre = _preprocess(1, frame, settings)
    assert pre.status == "READY"

    state = ad.detect_operating_state(pre.feature_frame)
    labels = state.labels
    # The middle of the active window should be ACTIVE...
    assert labels[130] == "ACTIVE"
    # ...and well outside it (e.g. before 09:00 or after 13:00) should be INACTIVE.
    assert labels[10] == "INACTIVE"
    assert labels[250] == "INACTIVE"


def test_different_operating_periods_on_different_days(tmp_path):
    """Day 1: 09:00-13:00. Day 2: 10:00-18:00. Day 3: 08:30-12:00 — the
    exact example from the project spec. No two days share a schedule."""
    settings = _settings(tmp_path)
    day1 = [(108, 156)]        # 09:00-13:00
    day2 = [(120, 216)]        # 10:00-18:00
    day3 = [(102, 144)]        # 08:30-12:00
    frame = _make_classroom_frame([day1, day2, day3], seed=2)
    pre = _preprocess(2, frame, settings)
    assert pre.status == "READY"

    state = ad.detect_operating_state(pre.feature_frame)
    labels = state.labels
    # Day 1 (rows 0-287): active window is 108-156.
    assert labels[130] == "ACTIVE"
    assert labels[20] == "INACTIVE"
    # Day 2 (rows 288-575): active window is 288+120=408 .. 288+216=504.
    assert labels[288 + 150] == "ACTIVE"
    assert labels[288 + 20] == "INACTIVE"
    # Day 3 (rows 576-863): active window is 576+102=678 .. 576+144=720.
    assert labels[576 + 110] == "ACTIVE"
    assert labels[576 + 250] == "INACTIVE"


def test_multiple_active_periods_in_one_day_with_a_break(tmp_path):
    """A single day with two separate class blocks and a break between
    them: 09:00-11:00 and 13:00-15:00 (rows 108-132 and 156-180)."""
    settings = _settings(tmp_path)
    frame = _make_classroom_frame([[(108, 132), (156, 180)]], seed=3)
    pre = _preprocess(3, frame, settings)
    assert pre.status == "READY"

    state = ad.detect_operating_state(pre.feature_frame)
    labels = state.labels
    assert labels[115] == "ACTIVE"    # first block
    assert labels[145] == "INACTIVE"  # the break between blocks
    assert labels[165] == "ACTIVE"    # second block


def test_inactive_periods_between_active_periods(tmp_path):
    """Explicit check that the gap between two active blocks is
    classified INACTIVE, not simply left ambiguous or defaulted ACTIVE."""
    settings = _settings(tmp_path)
    frame = _make_classroom_frame([[(50, 100), (200, 250)]], seed=4)
    pre = _preprocess(4, frame, settings)
    state = ad.detect_operating_state(pre.feature_frame)
    gap = state.labels[120:180]
    assert (gap == "INACTIVE").all()


def test_unused_day_is_entirely_inactive(tmp_path):
    """A day with no active window at all (classroom unused that day)
    must not be forced into ACTIVE just because the clock says so."""
    settings = _settings(tmp_path)
    frame = _make_classroom_frame([[(100, 160)], []], seed=5)  # day 2 unused
    pre = _preprocess(5, frame, settings)
    state = ad.detect_operating_state(pre.feature_frame)
    day2_labels = state.labels[288:576]
    assert (day2_labels == "INACTIVE").all()


def test_adaptive_per_pzem_operating_state_detection(tmp_path):
    """Two PZEMs with completely different active/idle power levels (a
    small appliance circuit vs. a heavy one) must each be classified
    using their OWN power distribution, not a shared/fixed threshold."""
    settings = _settings(tmp_path)
    small_frame = _make_classroom_frame(
        [[(108, 156)], [(120, 216)], [(102, 144)]],
        idle_power=2.0, active_power=15.0, active_noise=1.0, seed=6,
    )
    heavy_frame = _make_classroom_frame(
        [[(108, 156)], [(120, 216)], [(102, 144)]],
        idle_power=40.0, active_power=800.0, active_noise=20.0, seed=7,
    )
    small_pre = _preprocess(6, small_frame, settings)
    heavy_pre = _preprocess(7, heavy_frame, settings)

    small_state = ad.detect_operating_state(small_pre.feature_frame)
    heavy_state = ad.detect_operating_state(heavy_pre.feature_frame)

    # Same relative active windows -> both meters should find roughly the
    # same PROPORTION active, despite wildly different absolute power
    # levels. A single fixed-watt threshold could not get both right.
    small_active_frac = small_state.active_rows / len(small_state.labels)
    heavy_active_frac = heavy_state.active_rows / len(heavy_state.labels)
    assert abs(small_active_frac - heavy_active_frac) < 0.15
    # And a fixed threshold tuned for the heavy meter (e.g. 200W) would
    # call the ENTIRE small meter inactive -- confirm that's not what
    # happened here.
    assert small_state.active_rows > 0


def test_operating_state_does_not_use_timestamp(tmp_path):
    """Shifting every timestamp by a large constant (while leaving the
    electrical readings identical) must not change the operating-state
    labels at all -- state is derived purely from power/current."""
    settings = _settings(tmp_path)
    frame_a = _make_classroom_frame([[(108, 156)], [(50, 250)]], seed=8)
    frame_b = frame_a.copy()
    frame_b["timestamp"] = frame_b["timestamp"] + 30 * 86400  # shift a month

    pre_a = _preprocess(1, frame_a, settings)
    pre_b = _preprocess(2, frame_b, settings)

    state_a = ad.detect_operating_state(pre_a.feature_frame)
    state_b = ad.detect_operating_state(pre_b.feature_frame)

    assert list(state_a.labels) == list(state_b.labels)
    assert "timestamp" not in ad.STATE_FEATURES
    assert "hour_of_day" not in ad.STATE_FEATURES
    assert "day_of_week" not in ad.STATE_FEATURES


# ===========================================================================
# Persistence / debounce
# ===========================================================================

def test_persistence_filters_a_single_noisy_sample():
    """A lone one-row spike surrounded by idle rows must not register as
    a full state transition when persistence_samples > 1."""
    raw = np.array(["INACTIVE"] * 10 + ["ACTIVE"] + ["INACTIVE"] * 10, dtype=object)
    debounced = ad._apply_persistence(raw, min_persistence=2)
    assert (debounced == "INACTIVE").all()


def test_persistence_keeps_a_run_that_meets_the_threshold():
    raw = np.array(["INACTIVE"] * 10 + ["ACTIVE"] * 5 + ["INACTIVE"] * 10, dtype=object)
    debounced = ad._apply_persistence(raw, min_persistence=2)
    assert (debounced[10:15] == "ACTIVE").all()


def test_persistence_disabled_when_min_persistence_is_one():
    raw = np.array(["INACTIVE", "ACTIVE", "INACTIVE"], dtype=object)
    debounced = ad._apply_persistence(raw, min_persistence=1)
    assert list(debounced) == list(raw)


# ===========================================================================
# 7, 8: minimum active training data (rows AND day-diversity)
# ===========================================================================

def test_insufficient_active_historical_data_too_few_rows_and_days(tmp_path):
    """Only 2 days of a couple hours' activity each: well under both
    MIN_ACTIVE_TRAINING_ROWS and MIN_ACTIVE_TRAINING_DAYS."""
    settings = _settings(tmp_path)
    day1 = [(108, 128)]   # 20 rows active
    day2 = [(120, 140)]   # 20 rows active
    frame = _make_classroom_frame([day1, day2], seed=9)
    pre = _preprocess(1, frame, settings)
    assert pre.status == "READY"

    result = ad.detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)
    assert result.model_status == "INSUFFICIENT_DATA"
    assert result.result_frame is None
    assert result.active_days_represented < ad.MIN_ACTIVE_TRAINING_DAYS
    assert result.training_rows < ad.MIN_ACTIVE_TRAINING_ROWS
    assert str(ad.MIN_ACTIVE_TRAINING_ROWS) in result.reason


def test_insufficient_active_data_enough_rows_but_too_few_days(tmp_path):
    """Enough total ACTIVE rows (>= 256) but concentrated into only 2
    distinct days -- must still be INSUFFICIENT_DATA on day-diversity
    grounds, proving 256 active rows from one/two long days isn't
    accepted as a substitute for genuine day-to-day variation."""
    settings = _settings(tmp_path)
    day1 = [(0, 160)]    # 160 active rows
    day2 = [(0, 160)]    # 160 active rows -> 320 total, only 2 days
    frame = _make_classroom_frame([day1, day2], seed=10)
    pre = _preprocess(1, frame, settings)
    result = ad.detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)

    assert result.training_rows >= ad.MIN_ACTIVE_TRAINING_ROWS
    assert result.active_days_represented == 2
    assert result.model_status == "INSUFFICIENT_DATA"


def test_insufficient_active_data_enough_days_but_too_few_rows(tmp_path):
    """3 distinct days each contribute active rows (clearing the
    day-diversity bar and the per-day minimum) but the total is still
    well under MIN_ACTIVE_TRAINING_ROWS. Windows are offset away from
    row 0 of the whole frame so they don't collide with Stage 2's own
    ~23-row rolling-window warm-up NaNs at the very start of the series."""
    settings = _settings(tmp_path)
    days = [[(100, 115)], [(100, 115)], [(100, 115)]]  # 15 active rows/day * 3 = 45
    frame = _make_classroom_frame(days, seed=11)
    pre = _preprocess(1, frame, settings)
    result = ad.detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)

    assert result.active_days_represented == 3
    assert result.training_rows < ad.MIN_ACTIVE_TRAINING_ROWS
    assert result.model_status == "INSUFFICIENT_DATA"


def test_enough_active_historical_data_trains_successfully(tmp_path):
    """5 distinct days, each with several hours of active operation at
    varying start/stop times -- clears both the row-count and
    day-diversity requirements with real day-to-day schedule variation."""
    settings = _settings(tmp_path)
    days = [
        [(100, 160)],   # ~5h
        [(60, 130)],    # ~5.8h, different start
        [(150, 220)],   # ~5.8h, later start
        [(90, 150), (180, 210)],  # split day with a break
        [(0, 70)],      # very early start
    ]
    frame = _make_classroom_frame(days, seed=12)
    pre = _preprocess(1, frame, settings)
    assert pre.status == "READY"

    result = ad.detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)
    assert result.model_status == "READY"
    assert result.training_rows >= ad.MIN_ACTIVE_TRAINING_ROWS
    assert result.active_days_represented >= ad.MIN_ACTIVE_TRAINING_DAYS
    assert result.result_frame is not None
    # INACTIVE rows must be present and explicitly NOT_SCORED, never
    # given a fabricated anomaly label.
    inactive_rows = result.result_frame[result.result_frame["operating_state"] == "INACTIVE"]
    assert not inactive_rows.empty
    assert (inactive_rows["anomaly_label"] == "NOT_SCORED").all()
    assert inactive_rows["anomaly_score"].isna().all()
    # ACTIVE, feature-complete rows should mostly be scored NORMAL.
    active_scored = result.result_frame[result.result_frame["anomaly_label"] != "NOT_SCORED"]
    assert not active_scored.empty
    label_counts = active_scored["anomaly_label"].value_counts()
    assert label_counts.get("NORMAL", 0) > label_counts.get("ANOMALY", 0)


def test_different_pzems_different_operating_patterns_same_fleet_run(tmp_path):
    """Fleet run where different PZEMs have genuinely different
    day-to-day active schedules; each must be modeled independently."""
    settings = _settings(tmp_path, pzem_count=2)
    busy_days = [[(50, 200)], [(60, 220)], [(40, 190)], [(70, 230)], [(30, 180)]]
    quiet_days = [[(120, 132)], [(130, 140)]]  # short, too few days/rows

    busy_frame = _make_classroom_frame(busy_days, seed=13)
    quiet_frame = _make_classroom_frame(quiet_days, seed=14)

    preprocess_results = {
        1: _preprocess(1, busy_frame, settings),
        2: _preprocess(2, quiet_frame, settings),
    }
    results = ad.run_anomaly_detection_pipeline(settings=settings, preprocess_results=preprocess_results)

    assert results[1].model_status == "READY"
    assert results[2].model_status == "INSUFFICIENT_DATA"


# ===========================================================================
# 10, 11: no fake anomalies / results for insufficient data
# ===========================================================================

def test_no_fake_anomaly_for_insufficient_active_data(tmp_path):
    settings = _settings(tmp_path)
    frame = _make_classroom_frame([[(108, 112)]], seed=15)  # trivially small
    pre = _preprocess(1, frame, settings)
    result = ad.detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)

    assert result.model_status == "INSUFFICIENT_DATA"
    assert result.result_frame is None
    assert not hasattr(result, "anomaly_label")


def test_stage2_insufficient_data_short_history_passes_through(tmp_path):
    """Stage 2 itself refuses (too few rows overall) -- Stage 3 must
    surface that, not attempt operating-state detection on nothing."""
    settings = _settings(tmp_path)
    frame = _make_frame(5)
    pre = _preprocess(4, frame, settings)
    assert pre.status == "INSUFFICIENT_DATA"

    result = ad.detect_anomalies_for_meter(4, settings=settings, preprocess_result=pre)
    assert result.model_status == "INSUFFICIENT_DATA"
    assert result.result_frame is None
    assert result.training_rows == 0
    assert result.reason


# ===========================================================================
# 12: one PZEM failure does not stop others
# ===========================================================================

def test_one_meter_failure_does_not_break_fleet(tmp_path):
    settings = _settings(tmp_path)
    good_days = [[(100, 160)], [(60, 130)], [(150, 220)], [(90, 180)], [(0, 70)]]
    good_pre = _preprocess(1, _make_classroom_frame(good_days, seed=16), settings)

    broken_pre = _preprocess(2, _make_classroom_frame(good_days, seed=17), settings)
    broken_pre.feature_frame = broken_pre.feature_frame.drop(columns=["timestamp"])

    settings2 = _settings(tmp_path, pzem_count=2)
    results = ad.run_anomaly_detection_pipeline(
        settings=settings2,
        preprocess_results={1: good_pre, 2: broken_pre},
    )

    assert results[1].model_status == "READY"
    assert results[2].model_status == "INSUFFICIENT_DATA"
    assert results[2].debug_traceback is not None


# ===========================================================================
# 13: reproducibility
# ===========================================================================

def test_reproducible_with_fixed_random_state(tmp_path):
    settings = _settings(tmp_path)
    days = [[(100, 160)], [(60, 130)], [(150, 220)], [(90, 180)], [(0, 70)]]
    frame = _make_classroom_frame(days, seed=18)
    pre = _preprocess(1, frame, settings)

    result_a = ad.detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre, random_state=123)
    result_b = ad.detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre, random_state=123)

    scored_a = result_a.result_frame[result_a.result_frame["anomaly_label"] != "NOT_SCORED"]
    scored_b = result_b.result_frame[result_b.result_frame["anomaly_label"] != "NOT_SCORED"]
    pd.testing.assert_series_equal(
        scored_a["anomaly_score"].reset_index(drop=True),
        scored_b["anomaly_score"].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        scored_a["anomaly_label"].reset_index(drop=True),
        scored_b["anomaly_label"].reset_index(drop=True),
    )
    # Operating-state detection itself is also reproducible.
    assert list(result_a.result_frame["operating_state"]) == list(result_b.result_frame["operating_state"])


# ===========================================================================
# 14: existing Stage 3 behavior remains compatible
# ===========================================================================

def test_timestamp_excluded_from_ml_features(tmp_path):
    settings = _settings(tmp_path)
    days = [[(100, 160)], [(60, 130)], [(150, 220)], [(90, 180)], [(0, 70)]]
    pre = _preprocess(1, _make_classroom_frame(days, seed=19), settings)
    result = ad.detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)

    assert "timestamp" not in result.features_used
    assert "datetime_utc" not in result.features_used
    assert "timestamp" in ad.EXCLUDED_STAGE2_COLUMNS


def test_anomaly_score_is_not_a_probability(tmp_path):
    settings = _settings(tmp_path)
    days = [[(100, 160)], [(60, 130)], [(150, 220)], [(90, 180)], [(0, 70)]]
    pre = _preprocess(1, _make_classroom_frame(days, seed=20), settings)
    result = ad.detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)

    assert "anomaly_score" in result.result_frame.columns
    assert "anomaly_score_normalized" in result.result_frame.columns
    assert "anomaly_probability" not in result.result_frame.columns
    assert "probability" not in "".join(result.result_frame.columns).lower()

    scored = result.result_frame[result.result_frame["anomaly_label"] != "NOT_SCORED"]
    raw = scored["anomaly_score"]
    assert raw.min() < 0 or raw.max() > 1


def test_non_finite_feature_values_are_dropped_defensively(tmp_path):
    settings = _settings(tmp_path)
    days = [[(100, 160)], [(60, 130)], [(150, 220)], [(90, 180)], [(0, 70)]]
    pre = _preprocess(1, _make_classroom_frame(days, seed=21), settings)

    # Simulate a violated upstream guarantee: inject an inf into an
    # ACTIVE row of the feature frame directly.
    corrupted = pre.feature_frame.copy()
    active_idx = corrupted.index[corrupted["power"] > 30][5]
    corrupted.loc[active_idx, "power"] = np.inf
    pre.feature_frame = corrupted

    result = ad.detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)
    assert result.model_status == "READY"
    scored = result.result_frame[result.result_frame["anomaly_label"] != "NOT_SCORED"]
    assert np.isfinite(scored["anomaly_score"].to_numpy()).all()
    bad_ts = corrupted.loc[active_idx, "timestamp"]
    assert bad_ts not in scored["timestamp"].to_numpy()


def test_constant_power_meter_does_not_crash_and_trains(tmp_path):
    """A meter with essentially no variability (always-on, near-constant
    load) must not be forced into INACTIVE by the fallback -- see
    FALLBACK_MIN_COEFFICIENT_OF_VARIATION -- and must still train."""
    settings = _settings(tmp_path)
    frame = _make_frame(2000, power_fn=lambda i: 100.0, seed=22)
    pre = _preprocess(1, frame, settings)
    assert pre.status == "READY"

    result = ad.detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)
    assert result.model_status == "READY"
    assert result.operating_state_method == "fallback_no_variability"
    assert result.active_rows == len(pre.feature_frame)
    assert result.result_frame is not None
    scored = result.result_frame[result.result_frame["anomaly_label"] != "NOT_SCORED"]
    assert not scored[["anomaly_score", "anomaly_score_normalized"]].isna().any().any()


def test_fleet_summary_counts_only_scored_rows(tmp_path):
    settings = _settings(tmp_path, pzem_count=1)
    days = [[(100, 160)], [(60, 130)], [(150, 220)], [(90, 180)], [(0, 70)]]
    pre = _preprocess(1, _make_classroom_frame(days, seed=23), settings)
    results = {1: ad.detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)}
    summary = ad.summarize_fleet(results)
    assert summary.analyzed == 1
    assert summary.insufficient_data == 0
    assert summary.anomalous_now + summary.normal_now == 1
