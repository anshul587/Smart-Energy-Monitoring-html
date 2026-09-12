"""
ai/preprocessing.py
--------------------
STAGE 2: Data preprocessing.

Takes what ai.data_loader already fetched from Firebase (via
fetch_meter_history / fetch_all_history) and turns it into a clean,
feature-rich DataFrame per PZEM for Stage 3 (anomaly detection).

This module NEVER decides in advance which PZEMs have data. Every run
independently asks the data loader for each of settings.pzem_count
meters, looks at what actually comes back, and classifies READY vs
INSUFFICIENT_DATA from that — nothing about meter numbers is hardcoded.
If a PZEM has no history today, it reports INSUFFICIENT_DATA today; if it
starts reporting next week, the very next run picks it up automatically
because this always re-queries the loader instead of caching a "which
meters have data" list anywhere.

What "clean" means here, concretely:
  - sorted by timestamp, duplicate timestamps collapsed (defensively —
    the loader already dedups, but this stage doesn't assume that and
    dedups again so it's correct even if handed raw data directly)
  - rows with missing (NaN/None) readings dropped, counted, not filled
  - rows with numerically-present-but-physically-implausible readings
    (negative voltage, power factor outside [-1, 1], etc.) dropped,
    counted separately from missing values
  - nothing here EVER fabricates a value to fill a gap

Feature engineering, computed only for PZEMs with enough valid rows to
make it meaningful (see MIN_VALID_ROWS below):
  - hour_of_day, day_of_week, is_weekend
  - rolling mean/std of power, current, pf over two windows (~1h, ~1d
    worth of the firmware's 5-minute history slots)
  - baseline power/current/pf (median of that meter's own available
    history — a robust "what's normal for THIS meter" reference, not a
    fixed global threshold)
  - deviation and percent-deviation from that baseline
  - a rolling trend (slope of power/current over the ~1-day window,
    in units per hour) for Stage 8's predictive-maintenance trend checks
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from . import data_loader
from .config import Settings, get_settings
from .data_loader import READING_FIELDS, HistoryLoadResult

logger = logging.getLogger("ai.preprocessing")

# ---------------------------------------------------------------------------
# Tunable constants — all documented, none of them silently invented at
# call time. Changing these does not require touching the logic below.
# ---------------------------------------------------------------------------

# The firmware writes one history slot every 300s (config.h
# HISTORY_SLOT_SECONDS). These window sizes are expressed as a row COUNT
# assuming that cadence, used as the pandas rolling `window`. Real data can
# have gaps (a meter offline for a while), so this is "up to ~1h / ~1d of
# slots if they're all present", not a hard time guarantee — that's why
# min_periods below is deliberately looser than the full window.
HISTORY_SLOT_SECONDS = 300
SHORT_WINDOW = 12          # ~1 hour of 5-minute slots
LONG_WINDOW = 288          # ~1 day of 5-minute slots

# Minimum valid (post-cleaning) rows required before this stage will
# compute rolling/baseline features at all. Below this, statistics like a
# rolling std or a median baseline aren't meaningful yet.
MIN_VALID_ROWS = 12  # ~1 hour of slots at the firmware's cadence

# Minimum rows required for the LONG_WINDOW (~1 day) rolling features
# specifically. Below this, only SHORT_WINDOW features are computed and
# the long-window columns are left as NaN rather than computed over a
# window that's mostly padding.
MIN_ROWS_FOR_LONG_WINDOW = 24  # ~2 hours

# Sanity bounds used to flag "invalid" readings. These are deliberately
# generous, documented ranges to catch obviously corrupt values (sensor
# glitches, a legacy/malformed row that slipped past the loader's own
# type check) — they are NOT a precise electrical specification for any
# particular PZEM-004T variant or installation, and should be adjusted if
# your circuits legitimately run outside them.
VOLTAGE_RANGE = (0.0, 500.0)      # volts
CURRENT_RANGE = (0.0, 1000.0)     # amps — generous for CT-extended PZEMs
POWER_RANGE = (0.0, 1_000_000.0)  # watts — PZEM power is unsigned/real-power only
ENERGY_MIN = 0.0                  # cumulative kWh counter, can't be negative
FREQUENCY_RANGE = (40.0, 70.0)    # Hz — covers 50/60Hz grids with margin
PF_RANGE = (-1.0, 1.0)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    """Everything the data-discovery report and Stage 3 both need for one
    PZEM. feature_frame is None whenever status != READY — Stage 3 must
    check status before touching it."""

    pzem_number: int
    status: str                      # "READY" or "INSUFFICIENT_DATA"
    reason: Optional[str]

    record_count: int                # rows returned by the data loader, pre-cleaning
    oldest_timestamp: Optional[int]  # of the CLEANED/usable data, unix seconds
    newest_timestamp: Optional[int]
    available_days: float            # span of the cleaned/usable data

    valid_rows: int
    invalid_rows: int
    duplicates_removed: int
    missing_values: int

    feature_frame: Optional[pd.DataFrame] = field(default=None, repr=False)
    # Populated only when an unexpected exception was caught while
    # processing this meter, so a real failure can be diagnosed instead of
    # silently reported as "no data". Not shown in the normal report by
    # default — see print_debug_tracebacks().
    debug_traceback: Optional[str] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _within(value: pd.Series, bounds: tuple[float, float]) -> pd.Series:
    lo, hi = bounds
    return (value >= lo) & (value <= hi)


def _coerce_numeric(working: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Forces every column in READING_FIELDS to a real numeric (float64)
    dtype before any NumPy ufunc (np.isfinite, comparisons, rolling, etc.)
    touches them.

    Why this is needed: ai.data_loader.fetch_meter_history() concatenates
    each meter's on-disk cache with freshly-fetched Firebase rows via
    pd.concat(). The FIRST time a meter is ever cached, the "cached" side
    of that concat is an empty pd.DataFrame(columns=[...]) — which pandas
    gives object dtype (there's no data yet to infer float64 from). When
    pandas concatenates an object-dtype frame with a float64 frame, the
    combined result is silently downgraded to object dtype for ALL rows,
    including the real, valid float readings — pandas doesn't raise or
    warn about this. That object-dtype frame is what gets cached to disk
    and returned to every caller from then on.

    The values themselves are untouched (still genuine Python floats
    inside an object-dtype Series) — this is purely a dtype problem, not
    data corruption — but np.isfinite() specifically refuses to run on an
    object-dtype array even when every element is a plain float, which is
    exactly the
        ufunc 'isfinite' not supported for the input types
    error this function exists to prevent.

    pd.to_numeric(..., errors="coerce") is used rather than a plain
    .astype(float): a genuinely bad value (a stray string, a dict from a
    malformed write, etc.) becomes NaN instead of raising and killing the
    whole meter's preprocessing — and every such coercion is counted
    below so it's reported as an invalid value, never silently dropped
    without a trace and never replaced with a fabricated number.

    Returns (working_with_numeric_dtypes, missing_before_coercion,
    type_coercion_failures) — the two counts are kept separate so the
    report can distinguish "field was genuinely absent" from "field had a
    non-numeric value present".
    """
    # Captured BEFORE coercion: pd.isna() works fine on object-dtype data
    # (it's just checking for None/NaN/NaT, not doing numeric work), so
    # this accurately reflects what was truly missing to begin with.
    missing_before = int(working[READING_FIELDS].isna().sum().sum())

    for column in READING_FIELDS:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    missing_after = int(working[READING_FIELDS].isna().sum().sum())
    # Cells that were NOT missing before but became NaN after coercion
    # were present with a non-numeric value — that's a type/invalid-value
    # problem, distinct from genuinely absent data.
    type_coercion_failures = max(0, missing_after - missing_before)

    return working, missing_before, type_coercion_failures


def _clean_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    """Sorts, dedups, and validates one meter's readings.

    Returns (cleaned_frame, duplicates_removed, missing_values, invalid_rows).
    cleaned_frame only contains rows that are both complete (no NaN in any
    reading field) and within the physically-plausible sanity ranges above.

    invalid_rows counts BOTH kinds of "present but unusable" data: values
    that were the wrong type (non-numeric, coerced to NaN and dropped —
    see _coerce_numeric) and values that were numeric but outside the
    physically-plausible sanity ranges (negative current, PF outside
    [-1, 1], etc.). missing_values counts only fields that were genuinely
    absent (None/NaN) to begin with. Neither count is ever used to
    fabricate a replacement value — both describe rows that get dropped.
    """
    if frame.empty:
        return frame.copy(), 0, 0, 0

    working = frame.copy()

    # Sort first so "duplicate timestamp" dedup below has deterministic
    # behavior (keep the last-seen value, matching data_loader's own rule).
    working = working.sort_values("timestamp").reset_index(drop=True)

    before = len(working)
    working = working.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)
    duplicates_removed = before - len(working)

    # Force numeric dtype BEFORE any NaN-based dropping or NumPy ufunc
    # touches these columns — see _coerce_numeric's docstring for exactly
    # why this is required (object-dtype frames from data_loader's
    # empty-cache concat, which np.isfinite() can't operate on even
    # though the underlying values are real floats).
    working, missing_values, type_coercion_failures = _coerce_numeric(working)

    # Missing values: count NaN cells across the reading fields BEFORE
    # dropping, so the report reflects what was actually found in the
    # data, not just how many rows got discarded because of it. (Cells
    # that were present but non-numeric were already separated out into
    # type_coercion_failures above, and are folded into invalid_rows
    # below instead of missing_values — "missing" means genuinely absent.)
    working = working.dropna(subset=READING_FIELDS).reset_index(drop=True)

    # Invalid (present but implausible, or present but wrong-typed) values
    # — range-checked only on rows that already passed the missing-value
    # filter. working's READING_FIELDS columns are now guaranteed float64
    # by _coerce_numeric, so np.isfinite() below is safe to call.
    if working.empty:
        return working, duplicates_removed, missing_values, type_coercion_failures

    valid_mask = (
        _within(working["voltage"], VOLTAGE_RANGE)
        & _within(working["current"], CURRENT_RANGE)
        & _within(working["power"], POWER_RANGE)
        & (working["energy"] >= ENERGY_MIN)
        & _within(working["frequency"], FREQUENCY_RANGE)
        & _within(working["pf"], PF_RANGE)
        & np.isfinite(working[READING_FIELDS].to_numpy(dtype="float64")).all(axis=1)
    )
    range_invalid_rows = int((~valid_mask).sum())
    invalid_rows = type_coercion_failures + range_invalid_rows
    cleaned = working[valid_mask].reset_index(drop=True)

    return cleaned, duplicates_removed, missing_values, invalid_rows


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _rolling_stats(series: pd.Series, window: int, min_periods: int) -> tuple[pd.Series, pd.Series]:
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    return mean, std


def _rolling_slope(series: pd.Series, hours: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rolling linear-trend slope of `series` against elapsed hours, using
    pandas' built-in rolling covariance/variance (vectorized — avoids a
    much slower per-window polyfit over potentially tens of thousands of
    rows). Units: series-per-hour. NaN wherever the window's x-variance is
    zero (e.g. all timestamps identical) or there isn't enough data yet.
    """
    cov = series.rolling(window=window, min_periods=min_periods).cov(hours)
    var = hours.rolling(window=window, min_periods=min_periods).var()
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = cov / var
    return slope.replace([np.inf, -np.inf], np.nan)


def _safe_pct_deviation(value: pd.Series, baseline: float) -> pd.Series:
    """(value - baseline) / baseline * 100, guarded against a zero or
    NaN baseline instead of raising or silently producing inf."""
    if baseline is None or not np.isfinite(baseline) or baseline == 0:
        return pd.Series(np.nan, index=value.index)
    return (value - baseline) / baseline * 100.0


def _build_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """cleaned must already be sorted, deduped, and fully valid (output of
    _clean_frame with at least MIN_VALID_ROWS rows) — this function does
    not re-validate."""
    df = cleaned.copy()

    dt_utc = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["datetime_utc"] = dt_utc
    # NOTE: the firmware syncs time via configTime(0, 0, ...) — both
    # gmtOffset_sec and daylightOffset_sec are 0, so history/pzem_N
    # timestamps are true UTC, not local time. hour_of_day and
    # day_of_week below are therefore UTC-based. If you want them in a
    # local timezone (e.g. IST, UTC+5:30) for time-of-day pattern
    # analysis, convert dt_utc with .dt.tz_convert() before deriving
    # these two columns — deliberately not assumed here.
    df["hour_of_day"] = dt_utc.dt.hour
    df["day_of_week"] = dt_utc.dt.dayofweek  # Monday=0 .. Sunday=6
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    elapsed_hours = (df["timestamp"] - df["timestamp"].iloc[0]) / 3600.0
    n = len(df)
    short_min_periods = min(3, n)
    long_min_periods = min(MIN_ROWS_FOR_LONG_WINDOW, n) if n >= MIN_ROWS_FOR_LONG_WINDOW else None

    for column in ("power", "current", "pf"):
        mean_short, std_short = _rolling_stats(df[column], SHORT_WINDOW, short_min_periods)
        df[f"rolling_mean_{column}_1h"] = mean_short
        df[f"rolling_std_{column}_1h"] = std_short

        if long_min_periods is not None:
            mean_long, std_long = _rolling_stats(df[column], LONG_WINDOW, long_min_periods)
            df[f"rolling_mean_{column}_1d"] = mean_long
            df[f"rolling_std_{column}_1d"] = std_long
            df[f"rolling_trend_{column}_1d"] = _rolling_slope(
                df[column], elapsed_hours, LONG_WINDOW, long_min_periods
            )
        else:
            df[f"rolling_mean_{column}_1d"] = np.nan
            df[f"rolling_std_{column}_1d"] = np.nan
            df[f"rolling_trend_{column}_1d"] = np.nan

        # Baseline: median of ALL available valid history for this meter,
        # not the mean — robust to the anomalies Stage 3 is specifically
        # looking for pulling the "normal" reference away from normal.
        # This is a single scalar (this meter's own baseline), broadcast
        # across every row, not a rolling value.
        baseline = float(df[column].median())
        df[f"baseline_{column}"] = baseline
        df[f"deviation_{column}"] = df[column] - baseline
        df[f"pct_deviation_{column}"] = _safe_pct_deviation(df[column], baseline)

    return df


# ---------------------------------------------------------------------------
# Per-PZEM entry point
# ---------------------------------------------------------------------------

def _insufficient_data_result(
    pzem_number: int,
    reason: str,
    record_count: int = 0,
    oldest_timestamp: Optional[int] = None,
    newest_timestamp: Optional[int] = None,
    available_days: float = 0.0,
    valid_rows: int = 0,
    invalid_rows: int = 0,
    duplicates_removed: int = 0,
    missing_values: int = 0,
    debug_traceback: Optional[str] = None,
) -> PreprocessResult:
    """Shared constructor for every INSUFFICIENT_DATA / error return path,
    so record_count and the other already-known stats are never dropped
    just because a later step failed — this is the fix for the bug where
    an exception during cleaning/feature-building was reported as
    "Records: 0" even though the loader had genuinely returned records."""
    return PreprocessResult(
        pzem_number=pzem_number,
        status="INSUFFICIENT_DATA",
        reason=reason,
        record_count=record_count,
        oldest_timestamp=oldest_timestamp,
        newest_timestamp=newest_timestamp,
        available_days=available_days,
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
        duplicates_removed=duplicates_removed,
        missing_values=missing_values,
        feature_frame=None,
        debug_traceback=debug_traceback,
    )


def preprocess_meter(
    pzem_number: int,
    settings: Optional[Settings] = None,
    history_result: Optional[HistoryLoadResult] = None,
) -> PreprocessResult:
    """Preprocesses one PZEM.

    If history_result is omitted, this calls ai.data_loader.fetch_meter_history()
    itself (the normal path — hits cache/Firebase). Tests pass history_result
    directly to exercise cleaning/feature logic without any Firebase
    dependency at all — see tests/test_preprocessing.py.

    record_count is captured immediately after the fetch and is included
    in EVERY return path below, including ones triggered by an unexpected
    exception during cleaning or feature-building — a bug (fixed after a
    real Firebase run surfaced it) had exceptions there fall through to a
    generic "Records: 0" result, hiding the fact that real records had
    actually been loaded.
    """
    settings = settings or get_settings()
    if history_result is None:
        history_result = data_loader.fetch_meter_history(pzem_number, settings=settings)

    record_count = len(history_result.frame)

    if history_result.frame.empty:
        return _insufficient_data_result(
            pzem_number, "No usable historical data available.", record_count=0
        )

    try:
        cleaned, duplicates_removed, missing_values, invalid_rows = _clean_frame(history_result.frame)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: this is
        # exactly the boundary a dtype surprise from upstream can hit, and
        # record_count must survive it (see the bug this fixes, above).
        tb = traceback.format_exc()
        logger.error(
            "Preprocessing failed while CLEANING PZEM %d (record_count=%d, "
            "so this was NOT a zero-record meter):\n%s",
            pzem_number, record_count, tb,
        )
        return _insufficient_data_result(
            pzem_number,
            reason=f"Preprocessing error while cleaning data: {exc}",
            record_count=record_count,
            debug_traceback=tb,
        )

    valid_rows = len(cleaned)

    if valid_rows == 0:
        return _insufficient_data_result(
            pzem_number,
            reason=(
                f"{record_count} record(s) were loaded but none passed cleaning "
                f"({missing_values} missing value(s), {invalid_rows} invalid "
                f"value(s), {duplicates_removed} duplicate timestamp(s) removed)."
            ),
            record_count=record_count,
            invalid_rows=invalid_rows,
            duplicates_removed=duplicates_removed,
            missing_values=missing_values,
        )

    oldest_ts = int(cleaned["timestamp"].iloc[0])
    newest_ts = int(cleaned["timestamp"].iloc[-1])
    available_days = round((newest_ts - oldest_ts) / 86400, 2)

    if valid_rows < MIN_VALID_ROWS:
        return _insufficient_data_result(
            pzem_number,
            reason=(
                f"Only {valid_rows} valid record(s) after cleaning; at least "
                f"{MIN_VALID_ROWS} are needed (~1 hour at the firmware's "
                f"5-minute history interval) before rolling/baseline features "
                f"can be computed reliably. Record count and duplicates/"
                f"invalid/missing counts above are still real."
            ),
            record_count=record_count,
            oldest_timestamp=oldest_ts,
            newest_timestamp=newest_ts,
            available_days=available_days,
            valid_rows=valid_rows,
            invalid_rows=invalid_rows,
            duplicates_removed=duplicates_removed,
            missing_values=missing_values,
        )

    try:
        feature_frame = _build_features(cleaned)
    except Exception as exc:  # noqa: BLE001 - same rationale as the
        # cleaning try/except above: preserve every stat already computed.
        tb = traceback.format_exc()
        logger.error(
            "Preprocessing failed while BUILDING FEATURES for PZEM %d "
            "(record_count=%d, valid_rows=%d, so this was NOT a zero-record "
            "meter):\n%s",
            pzem_number, record_count, valid_rows, tb,
        )
        return _insufficient_data_result(
            pzem_number,
            reason=f"Preprocessing error while building features: {exc}",
            record_count=record_count,
            oldest_timestamp=oldest_ts,
            newest_timestamp=newest_ts,
            available_days=available_days,
            valid_rows=valid_rows,
            invalid_rows=invalid_rows,
            duplicates_removed=duplicates_removed,
            missing_values=missing_values,
            debug_traceback=tb,
        )

    reason = None
    if valid_rows < MIN_ROWS_FOR_LONG_WINDOW:
        reason = (
            f"READY with reduced feature set: {valid_rows} valid record(s) is "
            f"enough for ~1h rolling features but below {MIN_ROWS_FOR_LONG_WINDOW} "
            f"needed for the ~1d rolling window, so the *_1d columns are NaN "
            f"for this meter until more history accumulates."
        )

    return PreprocessResult(
        pzem_number=pzem_number,
        status="READY",
        reason=reason,
        record_count=record_count,
        oldest_timestamp=oldest_ts,
        newest_timestamp=newest_ts,
        available_days=available_days,
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
        duplicates_removed=duplicates_removed,
        missing_values=missing_values,
        feature_frame=feature_frame,
    )


def run_preprocessing_pipeline(settings: Optional[Settings] = None) -> dict[int, PreprocessResult]:
    """Discovers and preprocesses every configured PZEM (1..settings.pzem_count)
    independently. This is THE dynamic-discovery entry point — it makes no
    assumption about which meter numbers have data; each is queried fresh
    from the data loader every time this runs. One meter raising an
    unexpected error doesn't stop the others from being reported.

    preprocess_meter() now catches and reports cleaning/feature-building
    errors itself (preserving record_count when it does) — so this outer
    handler is only a safety net for something failing before that, e.g.
    the initial data_loader.fetch_meter_history() call itself raising.
    Genuinely: at that point we don't know how many records exist, so the
    reason says so explicitly rather than implying "0 records" the way
    the bug this replaces used to.
    """
    settings = settings or get_settings()
    results: dict[int, PreprocessResult] = {}
    for n in range(1, settings.pzem_count + 1):
        try:
            results[n] = preprocess_meter(n, settings=settings)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: one
            # meter's unexpected failure must not take down the report for
            # the other eight.
            tb = traceback.format_exc()
            logger.error("Unexpected error fetching/preprocessing PZEM %d:\n%s", n, tb)
            results[n] = _insufficient_data_result(
                n,
                reason=(
                    f"Data loading failed before a record count could be "
                    f"determined (this is NOT the same as the loader reporting "
                    f"zero records): {exc}"
                ),
                record_count=0,
                debug_traceback=tb,
            )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_ts(ts: Optional[int]) -> str:
    if ts is None:
        return "—"
    return f"{ts} ({datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()})"


def format_report(results: dict[int, PreprocessResult]) -> str:
    """Renders the per-PZEM data-discovery report in the format requested:
    record count, oldest/newest timestamp, available days, valid/invalid/
    duplicate/missing counts, and status — for every PZEM, in order,
    regardless of which ones actually have data."""
    lines = []
    for n in sorted(results):
        r = results[n]
        lines.append(f"PZEM {n}")
        lines.append(f"Records: {r.record_count}")
        lines.append(f"Oldest: {_fmt_ts(r.oldest_timestamp)}")
        lines.append(f"Newest: {_fmt_ts(r.newest_timestamp)}")
        lines.append(f"Available days: {r.available_days}")
        lines.append(f"Valid rows: {r.valid_rows}")
        lines.append(f"Invalid rows: {r.invalid_rows}")
        lines.append(f"Duplicates removed: {r.duplicates_removed}")
        lines.append(f"Missing values: {r.missing_values}")
        lines.append(f"Status: {r.status}")
        if r.reason:
            lines.append(f"Reason: {r.reason}")
        if r.debug_traceback:
            lines.append(
                "  (!) An exception occurred while processing this meter — "
                "see print_debug_tracebacks() output / logs for the full "
                "traceback. Records/valid/invalid counts above reflect "
                "whatever was successfully computed before the failure."
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def print_debug_tracebacks(results: dict[int, PreprocessResult]) -> None:
    """Prints the full traceback for every PZEM that hit an unexpected
    exception during this run. Call this after format_report() during
    development/diagnosis — it's intentionally separate from the report
    itself so the normal report stays concise."""
    any_failures = False
    for n in sorted(results):
        r = results[n]
        if r.debug_traceback:
            any_failures = True
            print(f"\n{'=' * 70}\nFull traceback for PZEM {n}\n{'=' * 70}")
            print(r.debug_traceback)
    if not any_failures:
        print("No exceptions were caught during this run.")
