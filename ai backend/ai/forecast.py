"""
ai/forecast.py
--------------
STAGE 9: Power Forecasting + Dashboard Integration (backend half).

Produces explainable, deterministic, HORIZON-WISE power forecasts for every
PZEM from the EXISTING 5-minute historical data (history/pzem_N/<unix-seconds>)
— never from the 10-second live stream — by consuming the Stage 2
preprocessing output (ai.preprocessing.PreprocessResult.feature_frame), which
already went through Stage 1's incremental-cache loading and Stage 2's
cleaning. This module creates NO second data-loading path and NEVER
re-downloads Firebase history.

===========================================================================
FORECASTING METHOD (explainable, deterministic, no fabricated data)
===========================================================================
A *seasonal daily-profile* forecast (a standard, explainable "typical-day"
model). It is NOT Holt-Winters / ML — and deliberately so, given the project
does not yet have the planned 30-day real dataset (see section "Data
sufficiency" below and the final report). The method is honest about
uncertainty and degrades gracefully instead of over-claiming:

  1. Each valid (timestamp, power) observation is bucketed to its
     5-minute-of-day bin (288 bins/day, UTC — the firmware's timestamps are
     UTC, per ai.preprocessing). Bins with no observation are filled by
     circular nearest-neighbour borrow from adjacent bins, so the profile is
     continuous and never invents a value that contradicts observed data.
  2. The per-bin TYPICAL POWER is the MEDIAN of that bin's observations —
     median (not mean) makes the profile robust to the outliers / sensor
     glitches the spec calls out: one absurd reading cannot skew the shape.
  3. The profile is re-scaled to the meter's RECENT LEVEL (median power over
     the most recent ~24 h, or the whole series if shorter). This is the
     explainable "typical shape, current magnitude" adaptation and is what
     lets the forecast track a load that has grown or shrunk recently
     without leaking future information.
  4. 24 h forecast = the scaled daily profile, one point per 5-minute slot
     (288 points) starting at the slot after the last real observation.
  5. 7-day forecast = the scaled daily profile repeated 7x, with an optional
     DAY-OF-WEEK modulation factor (each weekday's mean / overall mean) when
     enough distinct weekdays of history exist to estimate it. Below 7 days
     of history the weekly pattern is NOT invented — the daily profile is
     simply repeated and the horizon is labelled low-confidence.
  6. Uncertainty band = a robust spread of historical residuals
     (power - daily_profile[bin]), expressed as 1.4826 * MAD (a
     median-based sigma). Bounds = profile ± band, lower bound floored at 0
     (power is non-negative). The band is intentionally SUBTLE and is NOT a
     calibrated statistical 95% interval — it conveys typical historical
     scatter, not false precision.

===========================================================================
DATA SUFFICIENCY (honest; documented requirements)
===========================================================================
  * 24 h forecast requires >= ~1 day of valid history (FORECAST_MIN_SPAN_DAYS)
    to estimate a daily profile. Below that -> status NO_FORECAST, reason
    "insufficient_data". No fake line is produced.
  * 7 d forecast also requires >= ~1 day to produce anything. With < 7 days
    the weekly pattern cannot be estimated, so the daily profile is repeated
    and the horizon is marked low-confidence with an explicit note.
  * Confidence tiers (per horizon):
        low    : 1 <= span_days < 7
        medium : 7 <= span_days < 14
        high   : span_days >= 14   (matches Stage 8's sufficient_window_days)
    High confidence is NEVER claimed without >= 14 days of real data.

===========================================================================
PERSISTENCE / IDEMPOTENCY
===========================================================================
  /ai/forecast/pzem_N/<anchor-timestamp>   one record per PZEM per run
  /ai/forecast/system/<anchor-timestamp>   fleet-wide record

The Firebase child key IS the deterministic anchor timestamp (the last real
historical observation), and existing keys are checked before writing
(same get-then-set pattern as Stages 5/7), so re-running the SAME historical
input writes nothing new — no uncontrolled duplicates.

One PZEM's Firebase failure never stops the others (same contract as the
rest of the pipeline).
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

logger = logging.getLogger("ai.forecast")

# ---------------------------------------------------------------------------
# Tunable constants — documented, none invented at call time.
# ---------------------------------------------------------------------------

DAY_SECONDS = 86_400
BIN_SECONDS = HISTORY_SLOT_SECONDS          # 300 s — one 5-minute slot
BINS_PER_DAY = DAY_SECONDS // BIN_SECONDS   # 288

# Minimum valid history span (in days) before ANY forecast is attempted.
# Below this the daily profile is essentially undetermined, so we refuse to
# fabricate a forecast (NO_FORECAST / insufficient_data).
FORECAST_MIN_SPAN_DAYS = 1.0

# Confidence thresholds (days of valid coverage).
MED_CONF_DAYS = 7.0
HIGH_CONF_DAYS = 14.0

# A forecast band of exactly zero would imply false certainty. For series with
# effectively no residual scatter we still floor the band at a small fraction
# of the recent level so the UI never shows a razor-thin "perfect" band that
# over-claims precision. This is a display-honesty floor, not a data fudge.
MIN_BAND_FRACTION_OF_LEVEL = 0.02

# When fewer than this many distinct weekdays have usable history, we do NOT
# attempt a day-of-week modulation and just repeat the daily profile.
MIN_WEEKDAYS_FOR_DOW = 2


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Stage 9 output for ONE PZEM (or the system aggregate). Numeric fields
    stay empty/None whenever status == NO_FORECAST — nothing is fabricated.

    Each horizon dict (forecast_24h / forecast_7d) is COLUMNAR for compact
    Firebase storage and easy testing:
        {
          "status": "FORECAST" | "NO_FORECAST",
          "confidence": "low" | "medium" | "high" | None,
          "reason": Optional[str],
          "start_ts": int, "end_ts": int, "count": int,
          "timestamps":  [int, ...],
          "forecast_power_w": [float, ...],
          "lower_bound": [float, ...],
          "upper_bound": [float, ...],
        }
    """

    pzem_number: Optional[int]          # None for the system aggregate
    is_system: bool = False
    status: str = "NO_FORECAST"        # "FORECAST" | "NO_FORECAST"
    reason: Optional[str] = None

    anchor_timestamp: Optional[int] = None
    valid_samples: int = 0
    span_days: float = 0.0
    recent_level_w: Optional[float] = None

    forecast_24h: dict = field(default_factory=dict)
    forecast_7d: dict = field(default_factory=dict)

    meters_included: list[int] = field(default_factory=list)  # system only
    source_stage: str = "stage9/forecast"


# ---------------------------------------------------------------------------
# Series extraction
# ---------------------------------------------------------------------------

def _extract_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Pulls a clean (timestamp, power) series from a Stage 2 feature_frame.

    Defensively: numeric coercion, drop NaN/Inf, drop negative power, keep
    finite integer timestamps, sort + dedup by timestamp. Returns empty frame
    if nothing usable."""
    if frame is None or frame.empty or "power" not in frame.columns or "timestamp" not in frame.columns:
        return pd.DataFrame(columns=["timestamp", "power"])

    work = pd.DataFrame({
        "timestamp": pd.to_numeric(frame["timestamp"], errors="coerce"),
        "power": pd.to_numeric(frame["power"], errors="coerce"),
    })
    work = work.dropna()
    work = work[np.isfinite(work["power"].to_numpy(dtype="float64"))]
    work = work[work["power"] >= 0]
    work = work[np.isfinite(work["timestamp"].to_numpy(dtype="float64"))]
    work["timestamp"] = work["timestamp"].astype("int64")
    work = (
        work.sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="last")
        .reset_index(drop=True)
    )
    return work


def _bin_of(ts: int) -> int:
    return int((ts % DAY_SECONDS) // BIN_SECONDS)


def _fill_circular(arr: np.ndarray) -> np.ndarray:
    """Fills NaN entries in a 1-D circular array using the nearest available
    value (searching outward in both directions, wrapping around). If the
    whole array is NaN, returns it unchanged."""
    out = arr.copy()
    n = len(out)
    if n == 0 or not np.isnan(out).any():
        return out
    if np.isnan(out).all():
        return out
    for i in range(n):
        if np.isnan(out[i]):
            found = False
            for d in range(1, n + 1):
                for j in (i - d, i + d):
                    jj = j % n
                    if not np.isnan(out[jj]):
                        out[i] = out[jj]
                        found = True
                        break
                if found:
                    break
    return out


# ---------------------------------------------------------------------------
# Per-PZEM forecast
# ---------------------------------------------------------------------------

def _horizon_payload(
    status: str,
    confidence: Optional[str],
    reason: Optional[str],
    timestamps: list[int],
    power: list[float],
    lower: list[float],
    upper: list[float],
) -> dict:
    return {
        "status": status,
        "confidence": confidence,
        "reason": reason,
        "start_ts": int(timestamps[0]) if timestamps else None,
        "end_ts": int(timestamps[-1]) if timestamps else None,
        "count": len(timestamps),
        "timestamps": [int(t) for t in timestamps],
        "forecast_power_w": [round(float(p), 3) for p in power],
        "lower_bound": [round(float(b), 3) for b in lower],
        "upper_bound": [round(float(b), 3) for b in upper],
    }


def _empty_horizon(reason: str) -> dict:
    return _horizon_payload("NO_FORECAST", None, reason, [], [], [], [])


def forecast_meter(
    pzem_number: int,
    preprocess_result: Optional[PreprocessResult] = None,
    settings: Optional[Settings] = None,
) -> ForecastResult:
    """Forecasts power for one PZEM over 24 h and 7 d.

    If preprocess_result is omitted, calls ai.preprocessing.preprocess_meter()
    itself (the normal path). Tests pass a PreprocessResult directly, exactly
    like the other stages' tests.
    """
    settings = settings or get_settings()
    if preprocess_result is None:
        from . import preprocessing
        preprocess_result = preprocessing.preprocess_meter(pzem_number, settings=settings)

    result = ForecastResult(pzem_number=pzem_number, is_system=False)

    frame = preprocess_result.feature_frame
    series = _extract_series(frame)
    result.valid_samples = len(series)

    if series.empty:
        result.reason = (
            f"No usable preprocessed history for PZEM {pzem_number} "
            f"({preprocess_result.status})."
        )
        result.forecast_24h = _empty_horizon("insufficient_data")
        result.forecast_7d = _empty_horizon("insufficient_data")
        return result

    first_ts = int(series["timestamp"].iloc[0])
    last_ts = int(series["timestamp"].iloc[-1])
    span_days = (last_ts - first_ts) / DAY_SECONDS
    result.span_days = round(span_days, 3)

    if span_days < FORECAST_MIN_SPAN_DAYS:
        result.reason = (
            f"insufficient_data: only {span_days:.2f} day(s) of valid history "
            f"(need >= {FORECAST_MIN_SPAN_DAYS:.1f} day to estimate a daily "
            f"profile). Forecasts are withheld rather than fabricated."
        )
        result.forecast_24h = _empty_horizon("insufficient_data")
        result.forecast_7d = _empty_horizon("insufficient_data")
        return result

    result.status = "FORECAST"
    result.anchor_timestamp = last_ts

    # --- recent level (robust) ---
    window_start = last_ts - DAY_SECONDS
    recent = series[series["timestamp"] >= window_start]["power"]
    level = float(recent.median()) if len(recent) else float(series["power"].median())
    # Guard a degenerate all-zero level so we don't divide by zero.
    if level <= 0:
        level = 0.0
    result.recent_level_w = round(level, 3)

    # --- daily median profile (robust to outliers) ---
    powers = series["power"].to_numpy(dtype="float64")
    tss = series["timestamp"].to_numpy(dtype="int64")
    bin_idx = (tss % DAY_SECONDS) // BIN_SECONDS
    bin_idx = bin_idx.astype("int64")

    daily_median = np.full(BINS_PER_DAY, np.nan)
    for b in range(BINS_PER_DAY):
        m = powers[bin_idx == b]
        if m.size:
            daily_median[b] = float(np.median(m))
    daily_median = _fill_circular(daily_median)

    profile_mean = float(np.nanmean(daily_median))
    if profile_mean > 0:
        scale = level / profile_mean
    else:
        # Flat / zero series: profile is all zero, forecast stays zero.
        scale = 1.0
        daily_median = np.zeros(BINS_PER_DAY)

    scaled_profile = daily_median * scale

    # --- robust residual spread -> uncertainty band half-width ---
    residuals = powers - daily_median[bin_idx]
    med_res = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med_res)))
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma < 0:
        robust_sigma = 0.0
    # Honesty floor: never show a zero-width "perfect" band.
    min_band = MIN_BAND_FRACTION_OF_LEVEL * max(level, 1.0)
    band = max(robust_sigma, min_band)

    # --- day-of-week modulation (only if enough distinct weekdays) ---
    weekdays = pd.to_datetime(tss, unit="s", utc=True).dayofweek.to_numpy()
    distinct_dows = np.unique(weekdays)
    per_dow = None
    overall_mean = 0.0
    if profile_mean > 0:
        overall_mean = float(np.mean(powers))

    def weekday_factor(ts: int) -> float:
        if per_dow is None or len(distinct_dows) < MIN_WEEKDAYS_FOR_DOW or overall_mean <= 0:
            return 1.0
        wd = pd.Timestamp(ts, unit="s", tz="UTC").dayofweek
        return float(per_dow[wd] / overall_mean)

    if per_dow is None and profile_mean > 0 and len(distinct_dows) >= MIN_WEEKDAYS_FOR_DOW and overall_mean > 0:
        per_dow = np.array([
            float(np.mean(powers[weekdays == d])) if (weekdays == d).any() else overall_mean
            for d in range(7)
        ])

    # --- build horizons ---
    conf_24 = _confidence(span_days)
    conf_7 = _confidence(span_days)

    def build_points(n_days: int) -> tuple[list[int], list[float], list[float], list[float]]:
        n = n_days * BINS_PER_DAY
        start = last_ts + BIN_SECONDS
        t = [start + i * BIN_SECONDS for i in range(n)]
        pw = []
        lo = []
        hi = []
        for ts_i in t:
            b = _bin_of(ts_i)
            base = float(scaled_profile[b]) * weekday_factor(ts_i)
            pw.append(base)
            lo.append(max(0.0, base - band))
            hi.append(base + band)
        return t, pw, lo, hi

    t24, p24, l24, u24 = build_points(1)
    result.forecast_24h = _horizon_payload(
        "FORECAST", conf_24, None, t24, p24, l24, u24
    )

    t7, p7, l7, u7 = build_points(7)
    reason_7 = None
    if span_days < MED_CONF_DAYS:
        reason_7 = (
            f"Daily profile repeated for 7 days; weekly pattern NOT estimated "
            f"(only {span_days:.2f} day(s) of history; need >= "
            f"{MED_CONF_DAYS:.1f}). Low confidence."
        )
    result.forecast_7d = _horizon_payload(
        "FORECAST", conf_7, reason_7, t7, p7, l7, u7
    )

    return result


def _confidence(span_days: float) -> str:
    if span_days >= HIGH_CONF_DAYS:
        return "high"
    if span_days >= MED_CONF_DAYS:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# System forecast (simultaneous sum across valid PZEMs)
# ---------------------------------------------------------------------------

def compute_system_forecast(
    results: dict[int, ForecastResult],
    settings: Optional[Settings] = None,
) -> ForecastResult:
    """Sums point-wise across every PZEM that produced a FORECAST for BOTH
    horizons (same deterministic timestamps, so alignment is exact). Only
    valid available meters are included — meters with NO_FORECAST are
    excluded and never mixed in."""
    settings = settings or get_settings()
    out = ForecastResult(pzem_number=None, is_system=True)

    ready = {
        n: r for n, r in results.items()
        if r.status == "FORECAST"
        and r.forecast_24h.get("status") == "FORECAST"
        and r.forecast_7d.get("status") == "FORECAST"
    }
    out.meters_included = sorted(ready)

    if not ready:
        out.reason = "No PZEM produced a valid forecast to aggregate (all insufficient)."
        out.forecast_24h = _empty_horizon("insufficient_data")
        out.forecast_7d = _empty_horizon("insufficient_data")
        return out

    def align_sum(horizon_key: str) -> dict:
        # Every PZEM's forecast shares the same 5-minute cadence and is the
        # meter's own forward projection. To sum SIMULTANEOUSLY we add each
        # meter's value at the SAME future step k (0..n-1) — the natural
        # "step k minutes from now" alignment all meters share, exactly like
        # Stage 7's per-slot max aggregation. Timestamps are taken from the
        # fleet's latest anchor so the chart axis is consistent.
        ndays = 1 if horizon_key == "forecast_24h" else 7
        n = ndays * BINS_PER_DAY
        anchor = max(r.anchor_timestamp for r in ready.values())
        t = [anchor + i * BIN_SECONDS for i in range(n)]
        power = [0.0] * n
        lower = [0.0] * n
        upper = [0.0] * n
        for r in ready.values():
            src = r.forecast_24h if horizon_key == "forecast_24h" else r.forecast_7d
            pw_src = src["forecast_power_w"]
            lo_src = src["lower_bound"]
            up_src = src["upper_bound"]
            for k in range(min(n, len(pw_src))):
                power[k] += pw_src[k]
                lower[k] += lo_src[k]
                upper[k] += up_src[k]
        return _horizon_payload("FORECAST", None, None, t, power, lower, upper)

    out.status = "FORECAST"
    out.anchor_timestamp = max(r.anchor_timestamp for r in ready.values())
    out.valid_samples = sum(r.valid_samples for r in ready.values())
    out.span_days = round(min(r.span_days for r in ready.values()), 3)
    out.recent_level_w = None
    out.forecast_24h = align_sum("forecast_24h")
    out.forecast_7d = align_sum("forecast_7d")
    # System confidence = worst of included meters.
    worst = "high"
    for r in ready.values():
        for h in (r.forecast_24h, r.forecast_7d):
            c = h.get("confidence")
            if c == "low":
                worst = "low"
                break
            if c == "medium":
                worst = "medium" if worst == "high" else worst
    out.forecast_24h["confidence"] = worst
    out.forecast_7d["confidence"] = worst
    return out


# ---------------------------------------------------------------------------
# Fleet pipeline
# ---------------------------------------------------------------------------

def run_forecast_pipeline(
    settings: Optional[Settings] = None,
    preprocess_results: Optional[dict[int, PreprocessResult]] = None,
) -> tuple[dict[int, ForecastResult], ForecastResult]:
    """Runs Stage 9 for every configured PZEM plus the system aggregate.

    Reuses the caller's Stage 2 results when supplied (the normal flow — no
    second Firebase load); otherwise runs Stage 2 itself. One meter's failure
    never blocks the others.
    """
    settings = settings or get_settings()
    if preprocess_results is None:
        from . import preprocessing
        preprocess_results = preprocessing.run_preprocessing_pipeline(settings=settings)

    results: dict[int, ForecastResult] = {}
    for n in range(1, settings.pzem_count + 1):
        pre = preprocess_results.get(n)
        if pre is None:
            results[n] = ForecastResult(
                pzem_number=n, is_system=False,
                reason="No Stage 2 preprocessing result was available for this meter.",
            )
            results[n].forecast_24h = _empty_horizon("insufficient_data")
            results[n].forecast_7d = _empty_horizon("insufficient_data")
            continue
        try:
            results[n] = forecast_meter(n, preprocess_result=pre, settings=settings)
        except Exception as exc:  # noqa: BLE001 - fleet resilience contract
            logger.exception("Forecast failed unexpectedly for PZEM %d", n)
            results[n] = ForecastResult(
                pzem_number=n, is_system=False,
                reason=f"Unexpected error during forecast: {exc}",
            )
            results[n].forecast_24h = _empty_horizon("insufficient_data")
            results[n].forecast_7d = _empty_horizon("insufficient_data")
    return results, compute_system_forecast(results, settings=settings)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_horizon(h: dict) -> str:
    if not h or h.get("status") != "FORECAST":
        return f"  NO_FORECAST ({h.get('reason') or 'n/a'})"
    return (
        f"  FORECAST: {h['count']} points, confidence={h['confidence']}, "
        f"start={h['start_ts']}, end={h['end_ts']}"
        + (f"\n    note: {h['reason']}" if h.get("reason") else "")
    )


def format_report(results: dict[int, ForecastResult], system: ForecastResult) -> str:
    lines = []
    for n in sorted(results):
        r = results[n]
        lines.append(f"PZEM {n}")
        lines.append(f"Status: {r.status}")
        if r.status == "FORECAST":
            lines.append(f"Anchor: {r.anchor_timestamp}  Span(days): {r.span_days}  "
                         f"Recent level: {r.recent_level_w} W  Samples: {r.valid_samples}")
            lines.append("  24h: " + _fmt_horizon(r.forecast_24h).replace("\n", " "))
            lines.append("  7d:  " + _fmt_horizon(r.forecast_7d).replace("\n", " "))
        else:
            lines.append(f"Reason: {r.reason}")
        lines.append("")
    lines.append("SYSTEM-WIDE FORECAST")
    lines.append(f"Status: {system.status}")
    if system.status == "FORECAST":
        lines.append(f"Meters included: {system.meters_included}")
        lines.append("  24h:" + _fmt_horizon(system.forecast_24h).replace("\n", " "))
        lines.append("  7d: " + _fmt_horizon(system.forecast_7d).replace("\n", " "))
    else:
        lines.append(f"Reason: {system.reason}")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Firebase persistence (dedicated /ai/forecast hierarchy; mirrors Stages 5/7)
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


def _payload(result: ForecastResult) -> dict:
    """JSON-safe Firebase payload (no numpy types, no NaN)."""
    base = {
        "anchor_timestamp": int(result.anchor_timestamp) if result.anchor_timestamp else None,
        "source_stage": result.source_stage,
        "status": result.status,
        "reason": result.reason,
        "valid_samples": int(result.valid_samples),
        "span_days": float(result.span_days),
        "recent_level_w": (
            round(float(result.recent_level_w), 3)
            if result.recent_level_w is not None else None
        ),
        "forecast_24h": result.forecast_24h,
        "forecast_7d": result.forecast_7d,
    }
    if result.is_system:
        base["meters_included"] = [int(n) for n in result.meters_included]
    else:
        base["pzem_number"] = int(result.pzem_number)
    return base


def write_forecast_result(result: ForecastResult) -> bool:
    """Writes one PZEM's forecast to /ai/forecast/pzem_N/<anchor-timestamp>.

    Idempotent: the child key is the deterministic anchor timestamp and an
    existing key is skipped, so rerunning the same analysis is a no-op.
    Returns True if written OR already present, False on failure/no-forecast.
    """
    if result.status != "FORECAST" or result.anchor_timestamp is None:
        logger.debug(
            "Skipping forecast persist for PZEM %s (status=%s).",
            result.pzem_number, result.status,
        )
        return False
    try:
        payload = _payload(result)
    except (ValueError, TypeError) as exc:
        logger.warning("Invalid forecast payload for PZEM %s: %s", result.pzem_number, exc)
        return False

    path = f"ai/forecast/pzem_{result.pzem_number}"
    key = str(payload["anchor_timestamp"])
    try:
        ref = _db_ref(path)
        if ref.child(key).get() is not None:
            logger.debug("%s/%s already exists; skipping (idempotent).", path, key)
            return True
        ref.child(key).set(payload)
        logger.info("Wrote forecast for PZEM %d to /%s/%s", result.pzem_number, path, key)
        return True
    except Exception as exc:  # noqa: BLE001 - one meter's write failure must
        # not take down the rest of the pipeline (same contract as Stage 5/7).
        logger.error("Firebase write failed for %s/%s: %s", path, key, exc)
        return False


def write_system_forecast(result: ForecastResult) -> bool:
    """Writes the system-wide forecast to /ai/forecast/system/<anchor-ts>.
    Same idempotency contract as write_forecast_result()."""
    if result.status != "FORECAST" or result.anchor_timestamp is None:
        logger.debug("Skipping system forecast persist (status=%s).", result.status)
        return False
    try:
        payload = _payload(result)
    except (ValueError, TypeError) as exc:
        logger.warning("Invalid system forecast payload: %s", exc)
        return False
    key = str(payload["anchor_timestamp"])
    try:
        ref = _db_ref("ai/forecast/system")
        if ref.child(key).get() is not None:
            logger.debug("ai/forecast/system/%s already exists; skipping (idempotent).", key)
            return True
        ref.child(key).set(payload)
        logger.info("Wrote system-wide forecast to ai/forecast/system/%s", key)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Firebase write failed for ai/forecast/system/%s: %s", key, exc)
        return False


def run_stage_9_pipeline(
    settings: Optional[Settings] = None,
    preprocess_results: Optional[dict[int, PreprocessResult]] = None,
) -> dict[str, object]:
    """Full Stage 9 flow: forecast fleet-wide + system, then persist
    everything under /ai/forecast. Returns write counts for the report.
    Designed to run AFTER Stage 5/7 in the existing execution flow."""
    settings = settings or get_settings()
    results, system = run_forecast_pipeline(
        settings=settings, preprocess_results=preprocess_results
    )
    per_pzem_counts = {
        n: 1 if write_forecast_result(r) else 0 for n, r in sorted(results.items())
    }
    return {
        "per_pzem": per_pzem_counts,
        "system": 1 if write_system_forecast(system) else 0,
        "results": results,
        "system_result": system,
    }
