"""
ai/maintenance_risk.py
----------------------
STAGE 8: Predictive Maintenance RISK analysis (explainable, not predictive
of failure dates).

For each PZEM independently, combines:
  - Stage 2 preprocessed electrical history (feature_frame: readings,
    baselines, rolling features) — via the EXISTING Stage 1 cache/loading
    architecture; no second data loader exists or is created here;
  - Stage 3 anomaly results (optional AnomalyDetectionResult);
  - Stage 4 fault events (optional list[FaultEvent]);
  - Stage 7 peak results (optional PeakResult)

into a deterministic, weighted-evidence maintenance-risk score (0-100)
with human-readable indicators. THIS IS A RISK RANKING, NOT A FAILURE
PROBABILITY and never a failure-date prediction; every non-NORMAL result
carries the evidence that produced it.

STATUS: pipeline implemented on synthetic test fixtures only. Real-data
validation and weight/threshold tuning are PENDING until the planned
30-day real dataset exists.

===========================================================================
DATA SUFFICIENCY (evaluated before any risk claim)
===========================================================================
INSUFFICIENT_DATA              valid samples < MIN_MAINTENANCE_SAMPLES.
                               No score is computed and nothing is persisted.
LOW_CONFIDENCE                 enough samples but poor completeness
                               (< MIN_COMPLETENESS of the expected 5-minute
                               slots in the window) or fewer than
                               LOW_CONFIDENCE_MIN_SAMPLES (~1 day).
DEVELOPING                     real window shorter than
                               SUFFICIENT_WINDOW_DAYS.
SUFFICIENT_FOR_RISK_ANALYSIS   none of the above.

Risk-level cap by sufficiency (never over-claim from thin history):
  LOW_CONFIDENCE -> at most WATCH; DEVELOPING -> at most HIGH;
  INSUFFICIENT_DATA -> no score; SUFFICIENT -> uncapped.

===========================================================================
RISK-SCORING FORMULA (deterministic)
===========================================================================
Each triggered indicator adds its configured WEIGHTS[name] points
(see MAINTENANCE CONFIG block). risk_score = min(100, round(sum)).
risk_level: NORMAL < WATCH_AT(20) <= WATCH < HIGH_AT(45) <= HIGH <
CRITICAL_AT(70) <= CRITICAL, then capped by data sufficiency.
Identical input -> identical output; no randomness anywhere.

All thresholds live in MaintenanceRiskConfig (documented defaults,
overridable per call for validation/tuning later) and are SEPARATE from
Stage 4 fault thresholds and the Stage 7 peak threshold. Stage 8 NEVER
triggers emergency behaviour and writes only to /ai/maintenance/*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import Settings, get_settings
from .preprocessing import HISTORY_SLOT_SECONDS, PreprocessResult

logger = logging.getLogger("ai.maintenance_risk")

# ===========================================================================
# MAINTENANCE CONFIG — every Stage 8 knob in one documented place.
# Deliberately separate from fault_diagnosis.* (Stage 4) and
# peak_detection.* / PEAK_POWER_THRESHOLD_W (Stage 7).
# ===========================================================================

@dataclass(frozen=True)
class MaintenanceRiskConfig:
    """Defaults are engineering starting points chosen to be conservative
    on synthetic/short data; they MUST be re-tuned against the real 30-day
    dataset before any operational reliance."""

    # --- Data sufficiency -------------------------------------------------
    min_samples: int = 24                    # ~2 h of 5-min slots; below this: INSUFFICIENT_DATA
    low_confidence_min_samples: int = 288    # ~1 day; below this (but >= min_samples): LOW_CONFIDENCE
    min_completeness: float = 0.5            # valid samples / expected 5-min slots in window
    sufficient_window_days: float = 14.0     # below this window: DEVELOPING

    # --- Trend / indicator triggers ---------------------------------------
    trend_min_samples: int = 12              # fewer valid samples -> no trend claim at all
    trend_trigger_pct_of_baseline: float = 10.0   # |slope| per day, % of meter baseline
    pf_decline_per_day: float = 0.005        # PF points lost per day counts as decline
    low_pf: float = 0.90                     # "repeated low PF" threshold
    abnormal_fraction: float = 0.20          # share of samples needed for repeated-X indicators
    voltage_dev_pct: float = 5.0             # % deviation from the meter's median voltage
    frequency_dev_pct: float = 2.0           # % deviation from the meter's median frequency
    anomaly_rate_trigger: float = 0.10       # scored-row ANOMALY share triggering history indicator
    fault_repeat_count: int = 2              # same-category faults counting as "repeated"
    stress_above_baseline_factor: float = 3.0  # power > baseline x this = high-load sample
    stress_fraction: float = 0.05            # required share of high-load samples
    invalid_fraction_trigger: float = 0.20   # dropped/missing share triggering data-quality flag

    # --- Scoring weights (points; total possible 125, score caps at 100) --
    weights: dict = field(default_factory=lambda: {
        "power_trend": 15,
        "current_trend": 10,
        "pf_decline": 10,
        "pf_low": 15,
        "voltage_instability": 10,
        "frequency_instability": 5,
        "anomaly_history": 15,
        "fault_recurrence": 20,
        "fault_isolated": 5,
        "peak_threshold_exceeded": 5,
        "load_stress": 10,
        "data_quality": 10,
    })

    # --- Risk level cut-offs (on the 0-100 score) --------------------------
    watch_at: int = 20
    high_at: int = 45
    critical_at: int = 70


DEFAULT_CONFIG = MaintenanceRiskConfig()

SUFFICIENCY_INSUFFICIENT = "INSUFFICIENT_DATA"
SUFFICIENCY_LOW = "LOW_CONFIDENCE"
SUFFICIENCY_DEVELOPING = "DEVELOPING"
SUFFICIENCY_SUFFICIENT = "SUFFICIENT_FOR_RISK_ANALYSIS"

# Fixed interpretation banner persisted with every record (spec §16: never
# present the score as a failure probability or date).
RISK_INTERPRETATION = (
    "Relative maintenance-risk ranking from observed electrical history "
    "(0-100). NOT a probability of failure and NOT a failure-date "
    "prediction. Weights/thresholds pending tuning on real 30-day data."
)


@dataclass
class RiskResult:
    """Stage 8 output for ONE PZEM. Score fields stay None when the meter
    is INSUFFICIENT_DATA — nothing about risk is claimed without data."""

    pzem_number: int
    status: str                              # "RISK_ASSESSED" | "INSUFFICIENT_DATA"
    reason: Optional[str] = None

    risk_score: Optional[int] = None         # 0-100
    risk_level: Optional[str] = None         # NORMAL/WATCH/HIGH/CRITICAL
    data_sufficiency: str = SUFFICIENCY_INSUFFICIENT

    samples_analyzed: int = 0
    completeness: Optional[float] = None     # 0..1 vs expected 5-min slots
    window_start_ts: Optional[int] = None
    window_end_ts: Optional[int] = None      # ALSO the deterministic Firebase key

    indicators: list = field(default_factory=list)   # human-readable strings
    evidence: dict = field(default_factory=dict)     # structured numbers per indicator
    confidence: str = "none"                 # none/very_low/low/moderate (by sufficiency)


@dataclass
class SystemMaintenanceSummary:
    meters_analyzed: int = 0
    high_risk_meters: list = field(default_factory=list)   # pzem numbers, sorted
    watch_meters: list = field(default_factory=list)
    normal_meters: list = field(default_factory=list)
    insufficient_data_meters: list = field(default_factory=list)
    highest_risk_pzem: Optional[int] = None   # ties -> lowest number
    highest_risk_score: Optional[int] = None
    system_data_sufficiency: str = SUFFICIENCY_INSUFFICIENT   # worst-of-fleet
    summary_evidence: list = field(default_factory=list)
    timestamp: Optional[int] = None           # newest assessed window end


# ---------------------------------------------------------------------------
# Small deterministic stats helpers (guarded against degenerate series)
# ---------------------------------------------------------------------------

def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")


def _clean(ts: pd.Series, values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-drops NaN/non-numeric entries; returns finite arrays."""
    t = pd.to_numeric(ts, errors="coerce").to_numpy(dtype="float64")
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    mask = np.isfinite(t) & np.isfinite(v)
    return t[mask], v[mask]


def _slope_per_day(t: np.ndarray, v: np.ndarray) -> Optional[float]:
    """Least-squares slope of v against elapsed days. None when the window
    has too few samples or zero time-span (constant/degenerate) — no trend
    is ever inferred from those."""
    if len(t) < 2:
        return None
    x = (t - t[0]) / 86400.0
    vx = float(np.var(x))
    if vx <= 0:
        return None
    return float(np.cov(x, v)[0, 1] / vx)


def _fraction_outside(values: np.ndarray, reference: float, pct: float) -> float:
    if len(values) == 0 or reference == 0 or not np.isfinite(reference):
        return 0.0
    dev = np.abs(values - reference) / abs(reference)
    return float(np.mean(dev > pct / 100.0))


# ---------------------------------------------------------------------------
# Indicator evaluations — each returns (triggered, name, human_text, evidence)
# ---------------------------------------------------------------------------

def _trend_indicator(
    key: str, label: str, unit: str, frame: pd.DataFrame,
    baseline: float, cfg: MaintenanceRiskConfig,
    *, direction: str = "any", fmt_val=lambda v: f"{v:.3g}",
) -> tuple[bool, dict]:
    """Long-term linear trend of one reading vs this meter's baseline."""
    t, v = _clean(frame["timestamp"], frame[label])
    if len(v) < cfg.trend_min_samples:
        return False, {"indicator": key, "triggered": False,
                       "reason": f"only {len(v)} usable {label} sample(s); no trend inferred"}
    slope_day = _slope_per_day(t, v)
    if slope_day is None:
        return False, {"indicator": key, "triggered": False, "reason": "degenerate/constant time window"}
    trigger_abs = cfg.trend_trigger_pct_of_baseline / 100.0 * baseline if baseline else float("inf")
    hit = abs(slope_day) >= trigger_abs and (
        direction == "any" or (direction == "up" and slope_day > 0) or (direction == "down" and slope_day < 0)
    )
    pct = (slope_day / baseline * 100.0) if baseline else None
    text = (
        f"{label.capitalize()} trend {'+' if slope_day >= 0 else ''}"
        f"{fmt_val(slope_day)} {unit}/day"
        + (f" ({pct:+.1f}% of baseline {fmt_val(baseline)})" if pct is not None else "")
    )
    ev = {"indicator": key, "triggered": bool(hit), "slope_per_day": round(slope_day, 6),
          "baseline": round(baseline, 4), "samples": int(len(v))}
    return hit, {**ev, "text": text}


def _share_indicator(key: str, fraction: float, threshold: float,
                     what: str, detail: dict) -> tuple[bool, dict]:
    hit = fraction >= threshold
    text = f"{what}: {fraction:.1%} of samples (threshold {threshold:.0%})"
    return hit, {"indicator": key, "triggered": bool(hit), "fraction": round(fraction, 4),
                 "threshold": threshold, **detail, "text": text}


def _assess_indicators(
    frame: pd.DataFrame,
    cfg: MaintenanceRiskConfig,
    anomaly_result=None,
    fault_events=None,
    peak_result=None,
    preprocess_result: Optional[PreprocessResult] = None,
) -> tuple[list, list]:
    """Runs every indicator; returns (triggered_names_with_weights, all_evidence)."""
    evidence: list = []
    hits: list[tuple[str, dict]] = []

    def base(col: str) -> float:
        vals = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(dtype="float64")
        return float(np.median(vals)) if len(vals) else 0.0

    # A/B. Power & current long-term trends (power up = degradation/stress;
    # current up additionally suggests load or winding issues).
    for col, wkey, direction in (("power", "power_trend", "up"), ("current", "current_trend", "up")):
        hit, ev = _trend_indicator(wkey, col, "W" if col == "power" else "A",
                                   frame, base(col), cfg, direction=direction)
        evidence.append(ev)
        if hit:
            hits.append((wkey, ev))

    # C. Power factor: declining trend AND/OR repeated low PF.
    hit, ev = _trend_indicator("pf_decline", "pf", "PF", frame, base("pf"), cfg, direction="down",
                               fmt_val=lambda v: f"{v:.4g}")
    evidence.append(ev)
    if hit:
        hits.append(("pf_decline", ev))
    pf = pd.to_numeric(frame["pf"], errors="coerce").dropna().to_numpy(dtype="float64")
    frac_low_pf = float(np.mean(pf < cfg.low_pf)) if len(pf) else 0.0
    hit, ev = _share_indicator("pf_low", frac_low_pf, cfg.abnormal_fraction,
                               f"Repeated low power factor (< {cfg.low_pf})",
                               {"low_pf_threshold": cfg.low_pf})
    evidence.append(ev)
    if hit:
        hits.append(("pf_low", ev))

    # D/E. Voltage & frequency stability (deviation from own medians).
    for col, wkey, pct_name in (("voltage", "voltage_instability", cfg.voltage_dev_pct),
                                ("frequency", "frequency_instability", cfg.frequency_dev_pct)):
        vals = pd.to_numeric(frame[col], errors="coerce").dropna().to_numpy(dtype="float64")
        frac = _fraction_outside(vals, float(np.median(vals)) if len(vals) else 0.0, pct_name)
        hit, ev = _share_indicator(
            wkey, frac, cfg.abnormal_fraction,
            f"Repeated {col} deviation (> ±{pct_name:g}% of median)",
            {"deviation_pct": pct_name},
        )
        evidence.append(ev)
        if hit:
            hits.append((wkey, ev))

    # F. Anomaly history (Stage 3).
    if anomaly_result is not None and getattr(anomaly_result, "result_frame", None) is not None \
            and not anomaly_result.result_frame.empty:
        rf = anomaly_result.result_frame
        scored = rf[rf["anomaly_label"] != "NOT_SCORED"] if "anomaly_label" in rf.columns else rf.iloc[0:0]
        n_anom = int((scored["anomaly_label"] == "ANOMALY").sum()) if not scored.empty else 0
        rate = n_anom / len(scored) if len(scored) else 0.0
        hit, ev = _share_indicator("anomaly_history", rate, cfg.anomaly_rate_trigger,
                                   f"Anomaly frequency ({n_anom}/{len(scored)} scored rows)",
                                   {"anomalies": n_anom, "scored_rows": int(len(scored))})
        evidence.append(ev)
        if hit:
            hits.append(("anomaly_history", ev))
    else:
        evidence.append({"indicator": "anomaly_history", "triggered": False,
                         "reason": "no Stage 3 anomaly result available"})

    # G. Fault history (Stage 4).
    events = list(fault_events or [])
    by_cat: dict[str, int] = {}
    for e in events:
        by_cat[e.fault_type] = by_cat.get(e.fault_type, 0) + 1
    repeats = {c: n for c, n in by_cat.items() if n >= cfg.fault_repeat_count}
    if repeats:
        text = f"Repeated fault categories (Stage 4): " + ", ".join(f"{c} x{n}" for c, n in sorted(repeats.items()))
        hits.append(("fault_recurrence", {"indicator": "fault_recurrence", "triggered": True,
                                          "categories": by_cat, "text": text}))
        evidence.append({"indicator": "fault_recurrence", "triggered": True,
                         "categories": by_cat, "text": text})
    elif by_cat:
        text = f"Isolated fault event(s): {', '.join(sorted(by_cat))} (no category repeated yet)"
        hits.append(("fault_isolated", {"indicator": "fault_isolated", "triggered": True,
                                        "categories": by_cat, "text": text}))
        evidence.append({"indicator": "fault_isolated", "triggered": True,
                         "categories": by_cat, "text": text})
    else:
        evidence.append({"indicator": "fault_history", "triggered": False, "reason": "no Stage 4 fault events"})

    # H. Peak / load stress (Stage 7 annotation + own-history stress share).
    if peak_result is not None and getattr(peak_result, "exceeds_threshold", None):
        text = (f"Peak load {peak_result.peak_power_w:.1f} W exceeded the configured "
                f"annotation threshold {peak_result.threshold_w:g} W")
        hits.append(("peak_threshold_exceeded", {"indicator": "peak_threshold_exceeded",
                                                 "triggered": True, "text": text}))
        evidence.append({"indicator": "peak_threshold_exceeded", "triggered": True, "text": text})
    pw = pd.to_numeric(frame["power"], errors="coerce").dropna().to_numpy(dtype="float64")
    pbase = float(np.median(pw)) if len(pw) else 0.0
    if pbase > 0:
        frac_stress = float(np.mean(pw > pbase * cfg.stress_above_baseline_factor))
        hit, ev = _share_indicator(
            "load_stress", frac_stress, cfg.stress_fraction,
            f"High-load operation (> {cfg.stress_above_baseline_factor:g}x baseline {pbase:.1f} W)",
            {"stress_factor": cfg.stress_above_baseline_factor},
        )
        evidence.append(ev)
        if hit:
            hits.append(("load_stress", ev))

    # I. Communication / data quality (from Stage 2's honest drop counters).
    if preprocess_result is not None and preprocess_result.record_count > 0:
        bad = preprocess_result.invalid_rows + preprocess_result.missing_values + preprocess_result.duplicates_removed
        frac_bad = bad / preprocess_result.record_count
        hit, ev = _share_indicator("data_quality", frac_bad, cfg.invalid_fraction_trigger,
                                   "Data-quality/communication losses",
                                   {"bad_rows": int(bad), "record_count": preprocess_result.record_count})
        evidence.append(ev)
        if hit:
            hits.append(("data_quality", ev))

    return hits, evidence


# ---------------------------------------------------------------------------
# Per-PZEM risk assessment
# ---------------------------------------------------------------------------

def _sufficiency_and_confidence(
    n: int, window_days: float, completeness: float, cfg: MaintenanceRiskConfig,
) -> tuple[str, str]:
    if n < cfg.min_samples:
        return SUFFICIENCY_INSUFFICIENT, "none"
    if n < cfg.low_confidence_min_samples or completeness < cfg.min_completeness:
        return SUFFICIENCY_LOW, "very_low"
    if window_days < cfg.sufficient_window_days:
        return SUFFICIENCY_DEVELOPING, "low"
    return SUFFICIENCY_SUFFICIENT, "moderate"


_LEVEL_CAP = {SUFFICIENCY_LOW: "WATCH", SUFFICIENCY_DEVELOPING: "WARNING"}


def assess_maintenance_risk(
    pzem_number: int,
    preprocess_result: PreprocessResult,
    settings: Optional[Settings] = None,
    config: MaintenanceRiskConfig = DEFAULT_CONFIG,
    anomaly_result=None,
    fault_events=None,
    peak_result=None,
) -> RiskResult:
    """Assesses one PZEM deterministically. All AI inputs optional except
    the Stage 2 result; missing ones simply skip their indicators."""
    settings = settings or get_settings()
    cfg = config
    result = RiskResult(pzem_number=pzem_number, status="INSUFFICIENT_DATA")

    frame = preprocess_result.feature_frame
    required = ("timestamp", "power", "current", "pf", "voltage", "frequency")
    if frame is None or frame.empty or any(c not in frame.columns for c in required):
        result.reason = (
            f"No usable Stage 2 feature frame for PZEM {pzem_number} "
            f"({preprocess_result.status}); no risk claim made."
        )
        result.data_sufficiency = SUFFICIENCY_INSUFFICIENT
        return result

    work = frame.dropna(subset=["timestamp"]).copy()
    work["timestamp"] = pd.to_numeric(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"])
    n = len(work)
    result.samples_analyzed = int(n)
    if n < cfg.min_samples:
        result.reason = (
            f"insufficient_data: only {n} valid sample(s), need >= {cfg.min_samples}."
        )
        return result

    ts = work["timestamp"].astype("int64")
    result.window_start_ts = int(ts.iloc[0])
    result.window_end_ts = int(ts.iloc[-1])
    window_days = (result.window_end_ts - result.window_start_ts) / 86400.0
    expected = max(1.0, window_days * 86400.0 / HISTORY_SLOT_SECONDS)
    result.completeness = float(min(1.0, n / expected))

    sufficiency, confidence = _sufficiency_and_confidence(n, window_days, result.completeness, cfg)
    result.data_sufficiency = sufficiency
    result.confidence = confidence

    hits, evidence = _assess_indicators(
        work, cfg,
        anomaly_result=anomaly_result,
        fault_events=fault_events,
        peak_result=peak_result,
        preprocess_result=preprocess_result,
    )

    raw = sum(cfg.weights.get(name, 0) for name, _ in hits)
    score = int(min(100, round(raw)))
    if score < cfg.watch_at:
        level = "NORMAL"
    elif score < cfg.high_at:
        level = "WATCH"
    elif score < cfg.critical_at:
        level = "WARNING"
    else:
        level = "CRITICAL"

    cap = _LEVEL_CAP.get(sufficiency)
    if cap is not None:
        order = ["NORMAL", "WATCH", "WARNING", "CRITICAL"]
        if order.index(level) > order.index(cap):
            level = cap
            evidence.append({
                "indicator": "data_sufficiency_cap", "triggered": True,
                "text": f"Risk level capped at {cap}: data sufficiency is {sufficiency}.",
            })
            hits.append(("_cap", {}))

    result.status = "RISK_ASSESSED"
    result.risk_score = score
    result.risk_level = level
    result.indicators = [ev.get("text", "") for _, ev in hits if ev.get("text")]
    result.evidence = evidence
    if sufficiency != SUFFICIENCY_SUFFICIENT:
        result.reason = (
            f"{sufficiency}: {n} sample(s) over {window_days:.2f} day(s), "
            f"completeness {result.completeness:.0%}. Indicators are reported "
            f"for transparency; treat the risk level as provisional."
        )
    return result


# ---------------------------------------------------------------------------
# Fleet pipeline + system summary
# ---------------------------------------------------------------------------

def run_maintenance_risk_pipeline(
    settings: Optional[Settings] = None,
    preprocess_results: Optional[dict[int, PreprocessResult]] = None,
    anomaly_results: Optional[dict[int, object]] = None,
    fault_events: Optional[dict[int, list]] = None,
    peak_results: Optional[dict[int, object]] = None,
    config: MaintenanceRiskConfig = DEFAULT_CONFIG,
) -> tuple[dict[int, RiskResult], SystemMaintenanceSummary]:
    """Full Stage 8 flow. Reuses caller-owned Stage 2/3/4/7 outputs when
    given; anything missing is produced via the EXISTING pipelines (no new
    loading path). One meter failing never blocks the others."""
    settings = settings or get_settings()

    if preprocess_results is None:
        from . import preprocessing
        preprocess_results = preprocessing.run_preprocessing_pipeline(settings=settings)
    if anomaly_results is None:
        from . import anomaly_detection
        anomaly_results = anomaly_detection.run_anomaly_detection_pipeline(
            settings=settings, preprocess_results=preprocess_results)
    if fault_events is None:
        from . import fault_diagnosis
        fault_events = fault_diagnosis.run_fault_diagnosis_pipeline(
            preprocess_results, settings=settings)
    if peak_results is None:
        from . import peak_detection
        peak_results, _ = peak_detection.run_peak_detection_pipeline(
            settings=settings, preprocess_results=preprocess_results)

    results: dict[int, RiskResult] = {}
    for n in range(1, settings.pzem_count + 1):
        pre = preprocess_results.get(n)
        try:
            if pre is None:
                results[n] = RiskResult(
                    pzem_number=n, status="INSUFFICIENT_DATA",
                    reason="No Stage 2 preprocessing result was available for this meter.",
                )
                continue
            results[n] = assess_maintenance_risk(
                n, pre, settings=settings, config=config,
                anomaly_result=(anomaly_results or {}).get(n),
                fault_events=(fault_events or {}).get(n),
                peak_result=(peak_results or {}).get(n),
            )
        except Exception as exc:  # noqa: BLE001 - fleet resilience contract
            logger.exception("Maintenance risk assessment failed for PZEM %d", n)
            results[n] = RiskResult(
                pzem_number=n, status="INSUFFICIENT_DATA",
                reason=f"Unexpected error during risk assessment: {exc}",
            )
    return results, summarize_system(results)


def summarize_system(results: dict[int, RiskResult]) -> SystemMaintenanceSummary:
    """Deterministic fleet roll-up: counts by level, worst meter (ties ->
    lowest number), fleet sufficiency = WORST member state."""
    s = SystemMaintenanceSummary()
    assessed = [r for r in results.values() if r.status == "RISK_ASSESSED"]

    def nums(rs): return sorted(r.pzem_number for r in rs)
    s.meters_analyzed = len(results)
    s.insufficient_data_meters = nums(r for r in results.values() if r.status != "RISK_ASSESSED")
    s.normal_meters = nums(r for r in assessed if r.risk_level == "NORMAL")
    s.watch_meters = nums(r for r in assessed if r.risk_level == "WATCH")
    s.high_risk_meters = nums(r for r in assessed if r.risk_level in ("HIGH", "CRITICAL"))

    if assessed:
        best = min(assessed, key=lambda r: (-r.risk_score, r.pzem_number))
        s.highest_risk_pzem = best.pzem_number
        s.highest_risk_score = best.risk_score
        s.timestamp = max(r.window_end_ts for r in assessed)

    order = [SUFFICIENCY_SUFFICIENT, SUFFICIENCY_DEVELOPING, SUFFICIENCY_LOW, SUFFICIENCY_INSUFFICIENT]
    states = [r.data_sufficiency for r in results.values()]
    s.system_data_sufficiency = max(states, key=order.index) if states else SUFFICIENCY_INSUFFICIENT

    evidence = []
    if s.highest_risk_pzem is not None:
        top = next(r for r in results.values() if r.pzem_number == s.highest_risk_pzem)
        evidence.append(f"Highest risk: PZEM {top.pzem_number} score {top.risk_score} ({top.risk_level})")
        for ind in top.indicators[:3]:
            evidence.append(f"PZEM {top.pzem_number}: {ind}")
    if s.system_data_sufficiency != SUFFICIENCY_SUFFICIENT:
        evidence.append(
            f"Fleet data sufficiency is {s.system_data_sufficiency}; "
            "conclusions are provisional until more real history accumulates."
        )
    s.summary_evidence = evidence
    return s


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(results: dict[int, RiskResult], system: SystemMaintenanceSummary) -> str:
    lines = []
    for n in sorted(results):
        r = results[n]
        lines.append(f"PZEM {n}")
        lines.append(f"Status: {r.status} | Sufficiency: {r.data_sufficiency} | Confidence: {r.confidence}")
        if r.status == "RISK_ASSESSED":
            lines.append(f"Risk: {r.risk_score}/100 ({r.risk_level})")
            lines.append(f"Window: {r.window_start_ts} .. {r.window_end_ts} "
                         f"({r.samples_analyzed} samples, completeness {r.completeness:.0%})")
            for ind in r.indicators:
                lines.append(f"  - {ind}")
            if not r.indicators:
                lines.append("  - No degradation indicators triggered.")
            if r.reason:
                lines.append(f"Note: {r.reason}")
        else:
            lines.append(f"Reason: {r.reason}")
        lines.append("")
    lines.append("SYSTEM MAINTENANCE SUMMARY")
    lines.append(f"Meters analyzed: {system.meters_analyzed}")
    lines.append(f"High risk: {system.high_risk_meters} | Watch: {system.watch_meters} | "
                 f"Normal: {system.normal_meters} | Insufficient: {system.insufficient_data_meters}")
    lines.append(f"Highest risk: PZEM {system.highest_risk_pzem} ({system.highest_risk_score})")
    for e in system.summary_evidence:
        lines.append(f"  - {e}")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Firebase persistence (/ai/maintenance/*; mirrors Stage 5/7 idempotency)
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


def risk_payload(result: RiskResult) -> dict:
    """JSON-safe payload; raises ValueError on malformed/no-data results so
    callers can skip them instead of persisting garbage."""
    if result.status != "RISK_ASSESSED":
        raise ValueError("INSUFFICIENT_DATA result has no risk payload to persist.")
    for v in (result.risk_score, result.completeness):
        if v is None or not np.isfinite(v):
            raise ValueError(f"Non-finite/missing value {v!r} in risk result; refusing to persist.")
    return {
        "pzem_number": int(result.pzem_number),
        "timestamp": int(result.window_end_ts),          # deterministic idempotency key
        "risk_score": int(result.risk_score),
        "risk_level": str(result.risk_level),
        "data_sufficiency": str(result.data_sufficiency),
        "confidence": str(result.confidence),
        "completeness": round(float(result.completeness), 4),
        "samples_analyzed": int(result.samples_analyzed),
        "analysis_window": {
            "start": int(result.window_start_ts),
            "end": int(result.window_end_ts),
        },
        "indicators": [str(i) for i in result.indicators],
        "evidence": _json_safe_evidence(result.evidence),
        "risk_interpretation": RISK_INTERPRETATION,
        "source_stage": "stage8/maintenance_risk",
    }


def _json_safe_evidence(evidence: list) -> list:
    out = []
    for item in evidence:
        clean = {}
        for k, v in item.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v) if np.isfinite(v) else None
            else:
                clean[k] = v
        out.append(clean)
    return out


def system_summary_payload(summary: SystemMaintenanceSummary) -> dict:
    if summary.timestamp is None:
        raise ValueError("Empty system summary has nothing to persist.")
    return {
        "timestamp": int(summary.timestamp),
        "meters_analyzed": int(summary.meters_analyzed),
        "high_risk_meters": [int(n) for n in summary.high_risk_meters],
        "watch_meters": [int(n) for n in summary.watch_meters],
        "normal_meters": [int(n) for n in summary.normal_meters],
        "insufficient_data_meters": [int(n) for n in summary.insufficient_data_meters],
        "highest_risk_pzem": (
            int(summary.highest_risk_pzem) if summary.highest_risk_pzem is not None else None
        ),
        "highest_risk_score": (
            int(summary.highest_risk_score) if summary.highest_risk_score is not None else None
        ),
        "system_data_sufficiency": str(summary.system_data_sufficiency),
        "summary_evidence": [str(e) for e in summary.summary_evidence],
        "risk_interpretation": RISK_INTERPRETATION,
        "source_stage": "stage8/maintenance_risk",
    }


def write_risk_result(result: RiskResult) -> bool:
    """/ai/maintenance/pzem_N/<window-end>. Idempotent: same analysis ->
    same key -> skipped. INSUFFICIENT_DATA results are not persisted.
    Returns True if written OR already present, False otherwise."""
    try:
        payload = risk_payload(result)
    except ValueError as exc:
        logger.debug("Skipping maintenance persist for PZEM %s: %s", result.pzem_number, exc)
        return False
    path = f"ai/maintenance/pzem_{result.pzem_number}"
    key = str(payload["timestamp"])
    try:
        ref = _db_ref(path)
        if ref.child(key).get() is not None:
            logger.debug("%s/%s already exists; skipping (idempotent).", path, key)
            return True
        ref.child(key).set(payload)
        logger.info("Wrote maintenance risk for PZEM %d to /%s/%s", result.pzem_number, path, key)
        return True
    except Exception as exc:  # noqa: BLE001 - one meter's write failure must
        # not take down the rest of the pipeline (same contract as Stages 5/7).
        logger.error("Firebase write failed for %s/%s: %s", path, key, exc)
        return False


def write_system_summary(summary: SystemMaintenanceSummary) -> bool:
    """/ai/maintenance/system/<newest-window-end>. Same idempotency."""
    try:
        payload = system_summary_payload(summary)
    except ValueError as exc:
        logger.debug("Skipping system maintenance persist: %s", exc)
        return False
    key = str(payload["timestamp"])
    try:
        ref = _db_ref("ai/maintenance/system")
        if ref.child(key).get() is not None:
            logger.debug("ai/maintenance/system/%s already exists; skipping (idempotent).", key)
            return True
        ref.child(key).set(payload)
        logger.info("Wrote system maintenance summary to ai/maintenance/system/%s", key)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Firebase write failed for ai/maintenance/system/%s: %s", key, exc)
        return False


def run_stage_8_pipeline(
    settings: Optional[Settings] = None,
    preprocess_results=None,
    anomaly_results=None,
    fault_events=None,
    peak_results=None,
    config: MaintenanceRiskConfig = DEFAULT_CONFIG,
) -> dict:
    """Full Stage 8 flow + persistence. Returns write counts and results
    for the report. Runs AFTER Stage 7 in the existing execution flow."""
    settings = settings or get_settings()
    results, summary = run_maintenance_risk_pipeline(
        settings=settings,
        preprocess_results=preprocess_results,
        anomaly_results=anomaly_results,
        fault_events=fault_events,
        peak_results=peak_results,
        config=config,
    )
    per_pzem = {n: 1 if write_risk_result(r) else 0 for n, r in sorted(results.items())}
    return {
        "per_pzem": per_pzem,
        "system": 1 if write_system_summary(summary) else 0,
        "results": results,
        "system_result": summary,
    }
