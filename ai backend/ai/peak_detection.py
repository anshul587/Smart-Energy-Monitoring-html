"""
ai/peak_detection.py
--------------------
STAGE 7: Peak Load Detection.

Detects meaningful electrical power peaks from the EXISTING 5-minute
historical data (history/pzem_N/<unix-seconds>) — never from the
10-second live stream — by consuming the Stage 2 preprocessing output
(ai.preprocessing.PreprocessResult.feature_frame), which already went
through Stage 1's incremental-cache loading and Stage 2's cleaning.
This module creates NO second data-loading path and NEVER re-downloads
Firebase history.

A peak is NOT a fault, an anomaly, or an emergency. This module only
reports observed maxima; it writes to its own /ai/peaks hierarchy and
does not touch /meters, /history, /alerts, /ai/anomalies, or /ai/faults.

===========================================================================
PEAK-SELECTION RULE (exact, deterministic)
===========================================================================
Input for one PZEM: the valid rows of that meter's analysis window,
where "valid" means: integer unix-second timestamp AND power value that
is numeric, finite (not NaN/Inf), and >= 0. Rows failing this are
dropped and counted (invalid_rows), never guessed at.

1. If fewer than MIN_PEAK_SAMPLES valid samples exist -> NO_PEAK with
   reason "insufficient_data". Nothing is invented.
2. Candidates are examined in DESCENDING order of distinct power value.
   A candidate value v is REJECTED as an isolated single-sample spike —
   likely a sensor glitch rather than a meaningful operational peak —
   only when BOTH hold:
     a. LONELY: no row with power == v has an adjacent-in-time valid row
        (gap <= MAX_ADJACENT_GAP_SECONDS, i.e. the normal 300 s cadence)
        whose power >= PEAK_SUSTAIN_RATIO * v; AND
     b. EXTREME: v > ISOLATION_OUTLIER_FACTOR * median(valid powers).
   A short real load event (e.g. one 5-minute sample of a kettle against
   a modest background) fails condition (b) and stays a legitimate peak;
   an absurd reading (e.g. 5000 W on a 50 W circuit) fails both and is
   dropped. Rejected samples' values/timestamps are preserved in
   PeakResult (never deleted from history, never invented).
3. The first candidate not rejected becomes THE peak for this window.
4. Equal maximum values are ONE peak event: peak_timestamp is the
   EARLIEST timestamp among the winning value's occurrences
   (deterministic; no duplicate events per tied timestamp).
5. Zero-power series are valid data: peak_power_w = 0.0 at the earliest
   zero sample. A real maximum of zero is reported honestly.
6. If every distinct candidate value is rejected -> NO_PEAK ("all high
   samples were isolated single-sample spikes").

SUSTAINED PEAK / DURATION (5-minute sampling only)
--------------------------------------------------
The sustained run around the peak is the maximal chain of consecutive
valid samples containing the peak where each link's timestamp gap is
<= MAX_ADJACENT_GAP_SECONDS and each member's power >=
PEAK_SUSTAIN_RATIO * peak_power_w. duration_seconds = (run_len - 1) *
HISTORY_SLOT_SECONDS — a multiple of the actual sampling interval, so
no second-level precision is ever inferred. sustained = run_len >= 2;
a lone sample reports duration_seconds=0 and sustained=False.

THRESHOLDS (separate from Stage 4 fault thresholds)
---------------------------------------------------
PEAK_POWER_THRESHOLD_W (Settings.peak_power_threshold_w, env var,
default 0.0 = disabled) is an ANNOTATION ONLY: it never gates whether a
peak is detected or persisted, it just fills threshold_w /
exceeds_threshold / peak_above_threshold_w in the output. It is NOT the
Stage 4 FAULT_HIGH_POWER_W and does not create alerts.

BASELINE: baseline_power_w is the median power of the valid samples —
the same per-meter median-baseline convention Stage 2 uses.
peak_above_baseline_w = peak_power_w - baseline_power_w.

SYSTEM-WIDE PEAK
----------------
Valid samples of every analyzed PZEM are bucketed onto 300 s slots
(slot = floor(ts / HISTORY_SLOT_SECONDS) * HISTORY_SLOT_SECONDS) and
only buckets where EVERY analyzed PZEM has at least one valid sample
are used, so total_peak_power_w is always a true simultaneous sum.
total_peak_power_w = max bucket sum (earliest bucket on ties);
dominant_pzems = meter(s) with the largest individual contribution in
that bucket (all tied meters listed, sorted ascending).

PERSISTENCE / IDEMPOTENCY
-------------------------
/ai/peaks/pzem_N/<peak-timestamp>    one record per PZEM per window
/ai/peaks/system/<system-peak-slot>  fleet-wide record

The Firebase child key IS the deterministic function of the source data
(peak timestamp / system slot), and existing keys are checked before
writing (same get-then-set pattern as Stage 5), so re-running the same
historical analysis writes nothing new — no uncontrolled duplicates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import Settings, get_settings
from .preprocessing import (
    HISTORY_SLOT_SECONDS,
    PreprocessResult,
)

logger = logging.getLogger("ai.peak_detection")

# ---------------------------------------------------------------------------
# Tunable constants — documented, none invented at call time.
# ---------------------------------------------------------------------------

# Minimum number of VALID samples required before any peak claim is made.
# Below this (~15 min of coverage at the 5-minute cadence) the "max" of the
# series says more about how little we observed than about the load.
MIN_PEAK_SAMPLES = 3

# A neighbor/run member counts as sustaining a candidate peak when its
# power is at least this fraction of the candidate's power.
PEAK_SUSTAIN_RATIO = 0.5

# Maximum gap between consecutive samples that still counts as "adjacent"
# for sustainment runs and isolation checks. Equals one 5-minute slot; a
# larger gap means missing slots, which breaks adjacency honestly.
MAX_ADJACENT_GAP_SECONDS = HISTORY_SLOT_SECONDS

# A candidate is only treated as an isolated sensor-glitch spike when it is
# BOTH lonely (see rule 2a in the module docstring) AND more than this
# multiple of the window's median power. 10x keeps realistic short load
# events as peaks while catching absurd readings; tunable, and deliberately
# NOT the Stage 4 FAULT_HIGH_POWER_W absolute-watt fault threshold.
ISOLATION_OUTLIER_FACTOR = 10.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PeakResult:
    """Stage 7 output for ONE PZEM's analysis window. Numeric fields stay
    None whenever status == NO_PEAK — nothing is fabricated."""

    pzem_number: int
    status: str                        # "PEAK_FOUND" | "NO_PEAK"
    reason: Optional[str] = None

    peak_power_w: Optional[float] = None
    peak_timestamp: Optional[int] = None
    peak_duration_seconds: Optional[int] = None   # multiples of 300 s; 0 if lone sample
    sustained: Optional[bool] = None

    average_power_w: Optional[float] = None
    baseline_power_w: Optional[float] = None      # median of valid powers
    peak_above_baseline_w: Optional[float] = None

    threshold_w: float = 0.0
    exceeds_threshold: Optional[bool] = None
    peak_above_threshold_w: Optional[float] = None

    samples_analyzed: int = 0        # valid samples after cleaning
    invalid_rows_dropped: int = 0    # NaN/negative/non-numeric/bad-ts rows
    analysis_start_ts: Optional[int] = None
    analysis_end_ts: Optional[int] = None
    # Real observed readings rejected as isolated single-sample spikes
    # (rule 2a+2b) — recorded, never silently deleted.
    isolated_outliers_dropped: int = 0
    dropped_outlier_power_w: Optional[float] = None   # largest one dropped
    dropped_outlier_timestamp: Optional[int] = None   # its earliest occurrence
    # The series actually analyzed (timestamp + power), kept for the
    # system-wide aggregation. Not persisted.
    _series: Optional[pd.DataFrame] = field(default=None, repr=False)


@dataclass
class SystemPeakResult:
    """Fleet-wide peak across all analyzed PZEMs."""

    status: str                          # "PEAK_FOUND" | "NO_PEAK"
    reason: Optional[str] = None

    total_peak_power_w: Optional[float] = None
    timestamp: Optional[int] = None      # start of the 300 s slot (UTC unix s)
    dominant_pzems: list[int] = field(default_factory=list)
    per_pzem_power_w: dict[str, float] = field(default_factory=dict)
    meters_analyzed: int = 0
    threshold_w: float = 0.0
    exceeds_threshold: Optional[bool] = None
    slot_seconds: int = HISTORY_SLOT_SECONDS


# ---------------------------------------------------------------------------
# Per-PZEM detection
# ---------------------------------------------------------------------------

def _prepare_series(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Extracts the (timestamp, power) series the peak rules operate on.

    Drops and counts rows with non-numeric/missing timestamps or power,
    non-finite power (NaN/Inf), or negative power. Returns
    (series_sorted_by_timestamp, dropped_count).
    """
    work = pd.DataFrame({
        "timestamp": pd.to_numeric(frame["timestamp"], errors="coerce"),
        "power": pd.to_numeric(frame["power"], errors="coerce"),
    })
    before = len(work)
    work = work.dropna()
    work = work[np.isfinite(work["power"].to_numpy(dtype="float64"))]
    work = work[work["power"] >= 0]
    work["timestamp"] = work["timestamp"].astype("int64")
    work = (
        work.sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="last")
        .reset_index(drop=True)
    )
    return work, before - len(work)


def _has_supported_occurrence(series: pd.DataFrame, value: float) -> bool:
    """True if ANY row whose power == `value` has an adjacent-in-time row
    (gap <= MAX_ADJACENT_GAP_SECONDS) with power >= PEAK_SUSTAIN_RATIO *
    value. Series must be timestamp-sorted."""
    idx = series.index[series["power"] == value]
    ts = series["timestamp"]
    power = series["power"]
    pos = series.index.get_indexer(idx)
    n = len(series)
    for p in pos:
        for q in (p - 1, p + 1):
            if 0 <= q < n and abs(int(ts.iloc[q]) - int(ts.iloc[p])) <= MAX_ADJACENT_GAP_SECONDS:
                if power.iloc[q] >= PEAK_SUSTAIN_RATIO * value:
                    return True
    return False


def _sustained_run(series: pd.DataFrame, value: float, first_pos: int) -> int:
    """Length of the maximal consecutive run containing position
    `first_pos` where each link's gap <= MAX_ADJACENT_GAP_SECONDS and each
    member's power >= PEAK_SUSTAIN_RATIO * value."""
    ts = series["timestamp"].to_numpy(dtype="int64")
    power = series["power"].to_numpy(dtype="float64")
    floor = PEAK_SUSTAIN_RATIO * value

    def ok(i: int) -> bool:
        return power[i] >= floor

    lo = first_pos
    while lo - 1 >= 0 and ok(lo - 1) and ts[lo] - ts[lo - 1] <= MAX_ADJACENT_GAP_SECONDS:
        lo -= 1
    hi = first_pos
    while hi + 1 < len(series) and ok(hi + 1) and ts[hi + 1] - ts[hi] <= MAX_ADJACENT_GAP_SECONDS:
        hi += 1
    return hi - lo + 1


def detect_peak_for_meter(
    pzem_number: int,
    preprocess_result: Optional[PreprocessResult] = None,
    settings: Optional[Settings] = None,
) -> PeakResult:
    """Runs Stage 7 for one PZEM over its available historical window.

    If preprocess_result is omitted, calls ai.preprocessing.preprocess_meter()
    itself (the normal path — which hits Stage 1's cache/Firebase). Tests
    pass a PreprocessResult directly, exactly like the other stages' tests.
    """
    settings = settings or get_settings()
    if preprocess_result is None:
        from . import preprocessing
        preprocess_result = preprocessing.preprocess_meter(pzem_number, settings=settings)

    result = PeakResult(
        pzem_number=pzem_number,
        status="NO_PEAK",
        threshold_w=settings.peak_power_threshold_w,
    )

    frame = preprocess_result.feature_frame
    if frame is None or frame.empty or "power" not in frame.columns or "timestamp" not in frame.columns:
        result.reason = f"No usable preprocessed feature frame for PZEM {pzem_number} ({preprocess_result.status})."
        return result

    series, dropped = _prepare_series(frame)
    result.invalid_rows_dropped = dropped
    result.samples_analyzed = len(series)

    if len(series) < MIN_PEAK_SAMPLES:
        result.reason = (
            f"insufficient_data: only {len(series)} valid sample(s) "
            f"(need >= {MIN_PEAK_SAMPLES}) after dropping {dropped} invalid row(s)."
        )
        return result

    result.analysis_start_ts = int(series["timestamp"].iloc[0])
    result.analysis_end_ts = int(series["timestamp"].iloc[-1])
    avg = float(series["power"].mean())
    baseline = float(series["power"].median())

    # Walk distinct values descending; skip lonely AND extreme spikes.
    chosen_pos: Optional[int] = None
    chosen_value: Optional[float] = None
    for value in sorted(series["power"].unique(), reverse=True):
        if _has_supported_occurrence(series, value) or value <= ISOLATION_OUTLIER_FACTOR * baseline:
            chosen_value = float(value)
            # Earliest occurrence wins on ties (deterministic).
            occurrences = series.index[series["power"] == value]
            earliest_ts = min(int(series["timestamp"].iloc[i]) for i in occurrences)
            chosen_pos = int(
                series.index[
                    (series["power"] == value)
                    & (series["timestamp"] == earliest_ts)
                ][0]
            )
            break
        logger.debug(
            "PZEM %d: peak candidate %.3f W rejected as isolated single-sample spike.",
            pzem_number, value,
        )
        result.isolated_outliers_dropped += 1
        if result.dropped_outlier_power_w is None:
            occ = series.index[series["power"] == value]
            result.dropped_outlier_power_w = float(value)
            result.dropped_outlier_timestamp = int(
                min(int(series["timestamp"].iloc[i]) for i in occ)
            )

    if chosen_value is None:
        result.average_power_w = avg
        result.baseline_power_w = baseline
        result.reason = (
            "Every distinct high-power candidate was an isolated "
            f"single-sample spike (no adjacent slot within "
            f"{PEAK_SUSTAIN_RATIO:.0%} of its value AND more than "
            f"{ISOLATION_OUTLIER_FACTOR:.0f}x the window median); no "
            "operational peak found."
        )
        return result

    result.status = "PEAK_FOUND"
    result.peak_power_w = chosen_value
    result.peak_timestamp = int(series["timestamp"].iloc[chosen_pos])
    run_len = _sustained_run(series, chosen_value, chosen_pos)
    result.sustained = run_len >= 2
    result.peak_duration_seconds = (run_len - 1) * HISTORY_SLOT_SECONDS
    result.average_power_w = avg
    result.baseline_power_w = baseline
    result.peak_above_baseline_w = chosen_value - baseline

    threshold = settings.peak_power_threshold_w
    result.exceeds_threshold = bool(chosen_value > threshold) if threshold > 0 else None
    if threshold > 0:
        result.peak_above_threshold_w = chosen_value - threshold

    result._series = series
    return result


# ---------------------------------------------------------------------------
# System-wide peak
# ---------------------------------------------------------------------------

def compute_system_peak(
    peak_results: dict[int, PeakResult],
    settings: Optional[Settings] = None,
) -> SystemPeakResult:
    """Aggregates per-PZEM PeakResults into the fleet-wide peak.

    Only meters with PEAK_FOUND contribute their analyzed series. Buckets
    keep a slot only if EVERY contributing meter has a sample in it, so
    the total is always a genuine simultaneous observation.
    """
    settings = settings or get_settings()
    out = SystemPeakResult(status="NO_PEAK", threshold_w=settings.peak_power_threshold_w)

    ready = {
        n: r._series
        for n, r in peak_results.items()
        if r.status == "PEAK_FOUND" and r._series is not None and not r._series.empty
    }
    out.meters_analyzed = len(ready)
    if not ready:
        out.reason = "No PZEM produced a PEAK_FOUND result to aggregate."
        return out

    def bucket(ts: pd.Series) -> pd.Series:
        return (ts // HISTORY_SLOT_SECONDS) * HISTORY_SLOT_SECONDS

    # Inner-join every meter's bucketed series on the slot column.
    merged: Optional[pd.DataFrame] = None
    for n in sorted(ready):
        s = ready[n]
        part = pd.DataFrame({
            "slot": bucket(s["timestamp"]).astype("int64"),
            f"p{n}": s["power"].to_numpy(dtype="float64"),
        }).groupby("slot", as_index=False).max()  # one sample per meter per slot max
        merged = part if merged is None else merged.merge(part, on="slot", how="inner")

    assert merged is not None
    if merged.empty:
        out.reason = (
            "No common 300 s slot had valid samples from every analyzed "
            "meter, so no simultaneous system total can be computed."
        )
        return out

    pcols = [f"p{n}" for n in sorted(ready)]
    totals = merged[pcols].sum(axis=1)
    max_total = float(totals.max())
    # Deterministic tie handling: earliest slot wins.
    win_slot = int(merged.loc[totals.idxmax(), "slot"])
    win_row = merged[merged["slot"] == win_slot].iloc[0]

    contributions = {n: float(win_row[f"p{n}"]) for n in sorted(ready)}
    best = max(contributions.values())
    dominant = sorted(n for n, w in contributions.items() if w == best)

    out.status = "PEAK_FOUND"
    out.total_peak_power_w = max_total
    out.timestamp = win_slot
    out.dominant_pzems = dominant
    out.per_pzem_power_w = {f"pzem_{n}": w for n, w in contributions.items()}
    threshold = settings.peak_power_threshold_w
    out.exceeds_threshold = bool(max_total > threshold) if threshold > 0 else None
    return out


# ---------------------------------------------------------------------------
# Fleet pipeline
# ---------------------------------------------------------------------------

def run_peak_detection_pipeline(
    settings: Optional[Settings] = None,
    preprocess_results: Optional[dict[int, PreprocessResult]] = None,
) -> tuple[dict[int, PeakResult], SystemPeakResult]:
    """Runs Stage 7 for every configured PZEM plus the system-wide peak.

    Reuses the caller's Stage 2 results when supplied (the normal flow —
    no second Firebase load); otherwise runs Stage 2 itself. One meter's
    failure never blocks the others.
    """
    settings = settings or get_settings()
    if preprocess_results is None:
        from . import preprocessing
        preprocess_results = preprocessing.run_preprocessing_pipeline(settings=settings)

    results: dict[int, PeakResult] = {}
    for n in range(1, settings.pzem_count + 1):
        pre = preprocess_results.get(n)
        if pre is None:
            # Same fleet contract as Stage 3: a meter with no supplied
            # Stage 2 result is reported as-is, never re-fetched behind
            # the caller's back.
            results[n] = PeakResult(
                pzem_number=n,
                status="NO_PEAK",
                reason="No Stage 2 preprocessing result was available for this meter.",
                threshold_w=settings.peak_power_threshold_w,
            )
            continue
        try:
            results[n] = detect_peak_for_meter(n, preprocess_result=pre, settings=settings)
        except Exception as exc:  # noqa: BLE001 - fleet resilience contract
            logger.exception("Peak detection failed unexpectedly for PZEM %d", n)
            results[n] = PeakResult(
                pzem_number=n,
                status="NO_PEAK",
                reason=f"Unexpected error during peak detection: {exc}",
                threshold_w=settings.peak_power_threshold_w,
            )
    return results, compute_system_peak(results, settings=settings)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(results: dict[int, PeakResult], system: SystemPeakResult) -> str:
    lines = []
    for n in sorted(results):
        r = results[n]
        lines.append(f"PZEM {n}")
        lines.append(f"Status: {r.status}")
        if r.status == "PEAK_FOUND":
            lines.append(f"Peak power: {r.peak_power_w:.2f} W at {r.peak_timestamp}")
            lines.append(f"Sustained: {r.sustained} (duration ~{r.peak_duration_seconds} s)")
            lines.append(f"Average power: {r.average_power_w:.2f} W")
            lines.append(f"Baseline (median): {r.baseline_power_w:.2f} W (+{r.peak_above_baseline_w:.2f} W above)")
            thr = f", exceeds configured threshold {r.threshold_w} W" if r.exceeds_threshold else ""
            lines.append(f"Samples analyzed: {r.samples_analyzed}{thr}")
        else:
            lines.append(f"Reason: {r.reason}")
        lines.append("")
    lines.append("SYSTEM-WIDE PEAK")
    lines.append(f"Status: {system.status}")
    if system.status == "PEAK_FOUND":
        lines.append(f"Total peak power: {system.total_peak_power_w:.2f} W in slot {system.timestamp}")
        lines.append(f"Dominant PZEM(s): {system.dominant_pzems}")
        lines.append(f"Meters aggregated: {system.meters_analyzed}")
    else:
        lines.append(f"Reason: {system.reason}")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Firebase persistence (dedicated /ai/peaks hierarchy; mirrors Stage 5's
# get-then-set idempotency pattern exactly)
# ---------------------------------------------------------------------------

_firebase_app = None


def _init_firebase():
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app
    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "firebase-admin is not installed. Run: pip install -r requirements.txt"
        ) from exc

    settings = get_settings()
    cred_path = settings.firebase_service_account_path
    from pathlib import Path
    if not Path(cred_path).exists():
        raise RuntimeError(
            f"Service account file not found at {cred_path}. Never commit this file."
        )
    cred = credentials.Certificate(cred_path)
    _firebase_app = firebase_admin.initialize_app(
        cred, {"databaseURL": settings.firebase_database_url}
    )
    logger.info("Firebase Admin SDK initialized against %s", settings.firebase_database_url)
    return _firebase_app


def _db_ref(path: str):
    from firebase_admin import db
    _init_firebase()
    return db.reference(path)


def peak_payload(result: PeakResult) -> dict:
    """JSON-safe Firebase payload for one PZEM's peak (no numpy types,
    no NaN — invalid values raise rather than being written)."""
    if result.peak_power_w is None or result.peak_timestamp is None:
        raise ValueError("NO_PEAK result has no peak payload to persist.")
    values = [result.peak_power_w, result.average_power_w, result.baseline_power_w]
    for v in values:
        if v is not None and not np.isfinite(v):
            raise ValueError(f"Non-finite value {v!r} in peak result; refusing to persist.")
    return {
        "pzem_number": int(result.pzem_number),
        "timestamp": int(result.peak_timestamp),
        "peak_power_w": round(float(result.peak_power_w), 3),
        "average_power_w": round(float(result.average_power_w), 3),
        "baseline_power_w": round(float(result.baseline_power_w), 3),
        "peak_above_baseline_w": round(float(result.peak_above_baseline_w), 3),
        "peak_duration_seconds": int(result.peak_duration_seconds or 0),
        "sustained": bool(result.sustained),
        "threshold_w": float(result.threshold_w),
        "exceeds_threshold": result.exceeds_threshold,
        "peak_above_threshold_w": (
            round(float(result.peak_above_threshold_w), 3)
            if result.peak_above_threshold_w is not None else None
        ),
        "samples_analyzed": int(result.samples_analyzed),
        "invalid_rows_dropped": int(result.invalid_rows_dropped),
        "isolated_outliers_dropped": int(result.isolated_outliers_dropped),
        "dropped_outlier_power_w": (
            round(float(result.dropped_outlier_power_w), 3)
            if result.dropped_outlier_power_w is not None else None
        ),
        "dropped_outlier_timestamp": result.dropped_outlier_timestamp,
        "analysis_window": {
            "start": int(result.analysis_start_ts),
            "end": int(result.analysis_end_ts),
        },
        "source_stage": "stage7/peak_detection",
    }


def system_peak_payload(result: SystemPeakResult) -> dict:
    if result.total_peak_power_w is None or result.timestamp is None:
        raise ValueError("NO_PEAK system result has no payload to persist.")
    if not np.isfinite(result.total_peak_power_w):
        raise ValueError("Non-finite system peak; refusing to persist.")
    return {
        "timestamp": int(result.timestamp),
        "total_peak_power_w": round(float(result.total_peak_power_w), 3),
        "dominant_pzems": [int(n) for n in result.dominant_pzems],
        "per_pzem_power_w": {
            k: round(float(v), 3) for k, v in result.per_pzem_power_w.items()
        },
        "meters_analyzed": int(result.meters_analyzed),
        "threshold_w": float(result.threshold_w),
        "exceeds_threshold": result.exceeds_threshold,
        "slot_seconds": int(result.slot_seconds),
        "source_stage": "stage7/peak_detection",
    }


def write_peak_result(result: PeakResult) -> bool:
    """Writes one PZEM's peak to /ai/peaks/pzem_N/<peak-timestamp>.

    Idempotent: the child key is the deterministic peak timestamp and an
    existing key is skipped, so rerunning the same analysis is a no-op.
    Returns True if written OR already present, False on failure/no-peak.
    """
    try:
        payload = peak_payload(result)
    except ValueError as exc:
        logger.debug("Skipping peak persist for PZEM %s: %s", result.pzem_number, exc)
        return False
    path = f"ai/peaks/pzem_{result.pzem_number}"
    key = str(payload["timestamp"])
    try:
        ref = _db_ref(path)
        if ref.child(key).get() is not None:
            logger.debug("%s/%s already exists; skipping (idempotent).", path, key)
            return True
        ref.child(key).set(payload)
        logger.info("Wrote peak for PZEM %d to /%s/%s", result.pzem_number, path, key)
        return True
    except Exception as exc:  # noqa: BLE001 - one meter's write failure must
        # not take down the rest of the pipeline (same contract as Stage 5).
        logger.error("Firebase write failed for %s/%s: %s", path, key, exc)
        return False


def write_system_peak(result: SystemPeakResult) -> bool:
    """Writes the system-wide peak to /ai/peaks/system/<slot-timestamp>.
    Same idempotency contract as write_peak_result()."""
    try:
        payload = system_peak_payload(result)
    except ValueError as exc:
        logger.debug("Skipping system peak persist: %s", exc)
        return False
    key = str(payload["timestamp"])
    try:
        ref = _db_ref("ai/peaks/system")
        if ref.child(key).get() is not None:
            logger.debug("ai/peaks/system/%s already exists; skipping (idempotent).", key)
            return True
        ref.child(key).set(payload)
        logger.info("Wrote system-wide peak to ai/peaks/system/%s", key)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Firebase write failed for ai/peaks/system/%s: %s", key, exc)
        return False


def run_stage_7_pipeline(
    settings: Optional[Settings] = None,
    preprocess_results: Optional[dict[int, PreprocessResult]] = None,
) -> dict[str, object]:
    """Full Stage 7 flow: detect peaks fleet-wide + system-wide, then
    persist everything under /ai/peaks. Returns write counts for the
    report. Designed to run AFTER Stage 5 in the existing execution flow."""
    settings = settings or get_settings()
    peaks, system = run_peak_detection_pipeline(
        settings=settings, preprocess_results=preprocess_results
    )
    per_pzem_counts = {
        n: 1 if write_peak_result(r) else 0 for n, r in sorted(peaks.items())
    }
    return {
        "per_pzem": per_pzem_counts,
        "system": 1 if write_system_peak(system) else 0,
        "results": peaks,
        "system_result": system,
    }
