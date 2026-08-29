"""
ai/energy_saving.py
--------------------
STAGE 11: Evidence-based Energy-Saving Suggestions.

Generates specific recommendations from ACTUAL project data (never generic
advice) by consuming the EXISTING Stage 1/2 history and the EXISTING AI
outputs:

    - Stage 2 feature_frame        (historical power/current/voltage/pf/freq)
    - Stage 7 PeakResult           (peak load above baseline)
    - Stage 8 RiskResult           (maintenance-risk context, supporting only)
    - Stage 9 ForecastResult       (forecasted power windows)
    - Stage 10 bill rate           (for ESTIMATED cost savings)

Every recommendation carries: PZEM number (or SYSTEM), timestamp, type,
priority, reason, supporting metrics, evidence window, optional estimated
savings, and the source stages. No recommendation is emitted without
supporting evidence; insufficient data yields an empty list / a
NO_RECOMMENDATION record.

Priority (LOW/MEDIUM/HIGH) is SEPARATE from Stage 4 emergency severity and
Stage 8 maintenance-risk levels — it ranks how actionable/impactful a saving
opportunity is.

This module creates NO new data-loading path. It reads in-memory Stage 2
results and the other stages' result objects, and writes ONLY to
/ai/energy_saving/<anchor-ts>, never touching /meters, /history, /alerts,
/ai/anomalies, /ai/faults, /ai/peaks, /ai/maintenance, /ai/forecast, or
/ai/bill_prediction.

NOTE on savings: all savings are ESTIMATES. On a flat active-energy (kWh)
tariff, only reductions in actual consumed energy are estimable — chiefly
idle/standby load. Shift/peak/current/PF recommendations reduce demand or
reactive load but not necessarily kWh, so their potential_saving_kwh is left
null rather than fabricated. Real-data validation pending (see reports).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import get_settings

logger = logging.getLogger("ai.energy_saving")

# ---------------------------------------------------------------------------
# Tunable constants — documented, none invented at call time.
# ---------------------------------------------------------------------------

MIN_EVIDENCE_SAMPLES = 24          # < ~2 h of 5-min slots: too thin to claim

# Idle / standby load
IDLE_PERCENTILE = 5                # low-tail percentile == "always-on" baseline
IDLE_THRESHOLD_W = 15.0            # below this, no meaningful phantom load
IDLE_OPERATING_RATIO = 0.30        # idle must be < 30% of the operating level to
                                   # count as standby (else it's a constant load)
IDLE_REDUCTION_FRACTION = 0.30     # assumed curtailable share of idle load

# Power factor
PF_POOR_THRESHOLD = 0.90
PF_CRITICAL_THRESHOLD = 0.80

# Abnormal / high-power operation
HIGH_POWER_MULTIPLE = 2.5          # sample counts as "high" above median x this
ABNORMAL_FRACTION = 0.20           # share of samples needed for "repeated"
HIGH_POWER_MIN_EXCESS_W = 200.0    # absolute excess over baseline to matter

# Repeated high current
HIGH_CURRENT_MULTIPLE = 3.0

# Recurring high-demand window (history + forecast)
RECURRING_PEAK_RATIO = 1.5         # bin median vs overall median
RECURRING_PEAK_MIN_W = 50.0        # absolute floor for the window
FORECAST_HIGH_RATIO = 1.5
BIN_MINUTES = 30

# Stage 7 peak above baseline worth acting on
PEAK_REDUCE_ABSOLUTE_W = 500.0

PRIORITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

RECOMMENDATION_TEXT = {
    "SHIFT_NON_CRITICAL_LOAD": "Shift non-critical loads outside the recurring peak window.",
    "REDUCE_IDLE_CONSUMPTION": "Reduce unnecessary idle/standby consumption.",
    "IMPROVE_POWER_FACTOR": "Improve power factor (correction capacitors / reschedule reactive loads).",
    "REDUCE_HIGH_POWER": "Investigate and reduce abnormal high-power operation.",
    "INVESTIGATE_HIGH_CURRENT": "Investigate repeated high-current periods.",
    "RESPOND_PREDICTABLE_HIGH_LOAD": "Prepare for / shift load ahead of the predicted high-load window.",
    "REDUCE_PEAK_LOAD": "Reduce repeated peak-load usage (shed or shift loads during peaks).",
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    pzem_number: Optional[int]            # None => SYSTEM-wide
    timestamp: int
    recommendation_type: str
    priority: str                        # LOW | MEDIUM | HIGH
    recommendation: str
    reason: str
    supporting_metrics: Dict[str, Any] = field(default_factory=dict)
    evidence_window: Optional[str] = None
    potential_saving_kwh: Optional[float] = None
    potential_cost_saving: Optional[float] = None
    estimated_percent_reduction: Optional[float] = None
    source_stages: List[str] = field(default_factory=list)


@dataclass
class MeterEvidence:
    """Everything Stage 11 needs for ONE meter (or SYSTEM). Each field is
    optional; detectors skip themselves when their inputs are absent."""
    pzem_number: Optional[int]
    feature_frame: Optional[pd.DataFrame] = None
    peak_result: Optional[Any] = None
    risk_result: Optional[Any] = None
    forecast_result: Optional[Any] = None


# ---------------------------------------------------------------------------
# Small deterministic stats helpers
# ---------------------------------------------------------------------------

def _median(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return float("nan")
    return float(np.median(s))


def _max(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return float("nan")
    return float(np.max(s))


def _frac_above(s: pd.Series, thr: float) -> float:
    s = s.dropna()
    if s.empty:
        return 0.0
    return float((s > thr).mean())


def _fmt_ts(ts: Any) -> Optional[str]:
    if ts is None or not np.isfinite(float(ts)):
        return None
    try:
        return pd.to_datetime(int(ts), unit="s", utc=True).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _window_label(frame: pd.DataFrame) -> Optional[str]:
    if frame is None or "timestamp" not in frame or frame.empty:
        return None
    ts = frame["timestamp"].dropna()
    if ts.empty:
        return None
    return f"{_fmt_ts(ts.min())} -> {_fmt_ts(ts.max())} UTC"


def _recurring_high_window(frame: pd.DataFrame, power_col: str = "power",
                           ratio: float = RECURRING_PEAK_RATIO,
                           min_abs: float = RECURRING_PEAK_MIN_W) -> Optional[dict]:
    """Find a recurring high-demand window from (timestamp, power). Returns a
    dict with label / peak_median / overall_median, or None when no recurring
    window is evident. Deterministic."""
    if frame is None or power_col not in frame:
        return None
    df = frame[["timestamp", power_col]].dropna()
    if len(df) < MIN_EVIDENCE_SAMPLES:
        return None
    ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    hod = ts.dt.hour * 60 + ts.dt.minute
    bins = (hod // BIN_MINUTES).astype(int)
    grp = df.groupby(bins)[power_col].median()
    overall = float(df[power_col].median())
    if overall <= 0:
        return None
    mask = grp > ratio * overall
    if not mask.any():
        return None
    cand = grp[mask]
    max_bin = int(cand.idxmax())
    region = sorted(int(b) for b in cand.index if abs(int(b) - max_bin) <= 2)
    if not region:
        region = [max_bin]
    start_min = min(region) * BIN_MINUTES
    end_min = (max(region) + 1) * BIN_MINUTES
    label = f"{_fmt_hm(start_min)}-{_fmt_hm(end_min)}"
    peak_median = float(cand.max())
    if peak_median < min_abs:
        return None
    return {"label": label, "peak_median": peak_median, "overall_median": overall}


def _fmt_hm(minutes: int) -> str:
    minutes = int(minutes) % (24 * 60)
    h, m = divmod(minutes, 60)
    return f"{h:02d}:{m:02d}"


# ---------------------------------------------------------------------------
# Detectors — each returns a Recommendation or None
# ---------------------------------------------------------------------------

def detect_idle(ev: MeterEvidence, rate: float, ts: int) -> Optional[Recommendation]:
    frame = ev.feature_frame
    if frame is None or "power" not in frame:
        return None
    p = frame["power"].dropna()
    if len(p) < MIN_EVIDENCE_SAMPLES:
        return None
    idle = float(np.percentile(p, IDLE_PERCENTILE))
    operating = float(np.percentile(p, 90))   # high-tail operating level
    if not np.isfinite(idle) or not np.isfinite(operating) or operating <= 0:
        return None
    if idle <= IDLE_THRESHOLD_W:
        return None
    # A flat/constant load (idle ~ operating) is NOT standby waste.
    if idle >= IDLE_OPERATING_RATIO * operating:
        return None
    mean_w = float(p.mean())
    total_kwh_month = mean_w / 1000.0 * 24 * 30
    low_frac = float((p <= idle * 2).mean())
    saving_kwh = idle / 1000.0 * 24 * 30 * IDLE_REDUCTION_FRACTION * low_frac
    cost = saving_kwh * rate if rate and rate > 0 else None
    pct = (saving_kwh / total_kwh_month * 100.0) if total_kwh_month > 0 else None
    priority = "HIGH" if idle > 200 else "MEDIUM" if idle > 50 else "LOW"
    return Recommendation(
        pzem_number=ev.pzem_number, timestamp=ts,
        recommendation_type="REDUCE_IDLE_CONSUMPTION", priority=priority,
        recommendation=RECOMMENDATION_TEXT["REDUCE_IDLE_CONSUMPTION"],
        reason=f"Minimum observed power (idle/standby baseline) is {idle:.0f} W, "
               f"indicating an always-on load. Curtailing ~{int(IDLE_REDUCTION_FRACTION*100)}% "
               f"of it is a realistic saving.",
        supporting_metrics={
            "idle_baseline_w": round(idle, 2),
            "mean_power_w": round(mean_w, 2),
            "low_power_fraction": round(low_frac, 3),
        },
        evidence_window=_window_label(frame),
        potential_saving_kwh=round(saving_kwh, 3),
        potential_cost_saving=round(cost, 2) if cost is not None else None,
        estimated_percent_reduction=round(pct, 2) if pct is not None else None,
        source_stages=["stage1/history", "stage2/preprocessing"],
    )


def detect_pf(ev: MeterEvidence, rate: float, ts: int) -> Optional[Recommendation]:
    frame = ev.feature_frame
    if frame is None or "pf" not in frame:
        return None
    pf = frame["pf"].dropna()
    if len(pf) < MIN_EVIDENCE_SAMPLES:
        return None
    med = float(pf.median())
    if not np.isfinite(med) or med >= PF_POOR_THRESHOLD:
        return None
    priority = "HIGH" if med < PF_CRITICAL_THRESHOLD else "MEDIUM"
    return Recommendation(
        pzem_number=ev.pzem_number, timestamp=ts,
        recommendation_type="IMPROVE_POWER_FACTOR", priority=priority,
        recommendation=RECOMMENDATION_TEXT["IMPROVE_POWER_FACTOR"],
        reason=f"Median power factor is {med:.2f}, below {PF_POOR_THRESHOLD:.2f}. "
               f"Correction reduces reactive/apparent load (and demand charges).",
        supporting_metrics={"median_pf": round(med, 3),
                            "threshold": PF_POOR_THRESHOLD},
        evidence_window=_window_label(frame),
        potential_saving_kwh=None,           # not a kWh reduction on a kWh tariff
        potential_cost_saving=None,
        estimated_percent_reduction=None,
        source_stages=["stage1/history", "stage2/preprocessing"],
    )


def detect_high_power(ev: MeterEvidence, rate: float, ts: int) -> Optional[Recommendation]:
    frame = ev.feature_frame
    if frame is None or "power" not in frame:
        return None
    p = frame["power"].dropna()
    if len(p) < MIN_EVIDENCE_SAMPLES:
        return None
    med = _median(p)
    mx = _max(p)
    if not np.isfinite(med) or med <= 0 or mx <= HIGH_POWER_MULTIPLE * med:
        return None
    excess_frac = _frac_above(p, HIGH_POWER_MULTIPLE * med)
    if excess_frac <= 0:
        return None
    if excess_frac < ABNORMAL_FRACTION:
        return None
    if (mx - med) < HIGH_POWER_MIN_EXCESS_W:
        return None
    priority = "HIGH" if (mx > 5 * med and mx > 1000) else "MEDIUM" if mx > 2 * med else "LOW"
    return Recommendation(
        pzem_number=ev.pzem_number, timestamp=ts,
        recommendation_type="REDUCE_HIGH_POWER", priority=priority,
        recommendation=RECOMMENDATION_TEXT["REDUCE_HIGH_POWER"],
        reason=f"Repeated power excursions up to {mx:.0f} W (median {med:.0f} W) in "
               f"{excess_frac*100:.0f}% of samples.",
        supporting_metrics={"max_power_w": round(mx, 2), "median_power_w": round(med, 2),
                            "high_power_fraction": round(excess_frac, 3)},
        evidence_window=_window_label(frame),
        potential_saving_kwh=None,
        potential_cost_saving=None,
        estimated_percent_reduction=None,
        source_stages=["stage1/history", "stage2/preprocessing"],
    )


def detect_high_current(ev: MeterEvidence, rate: float, ts: int) -> Optional[Recommendation]:
    frame = ev.feature_frame
    if frame is None or "current" not in frame:
        return None
    c = frame["current"].dropna()
    if len(c) < MIN_EVIDENCE_SAMPLES:
        return None
    med = _median(c)
    mx = _max(c)
    if not np.isfinite(med) or med <= 0 or mx <= HIGH_CURRENT_MULTIPLE * med:
        return None
    frac = _frac_above(c, HIGH_CURRENT_MULTIPLE * med)
    if frac < ABNORMAL_FRACTION:
        return None
    priority = "HIGH" if mx > 5 * med else "MEDIUM"
    return Recommendation(
        pzem_number=ev.pzem_number, timestamp=ts,
        recommendation_type="INVESTIGATE_HIGH_CURRENT", priority=priority,
        recommendation=RECOMMENDATION_TEXT["INVESTIGATE_HIGH_CURRENT"],
        reason=f"Repeated high-current periods up to {mx:.2f} A (median {med:.2f} A) in "
               f"{frac*100:.0f}% of samples.",
        supporting_metrics={"max_current_a": round(mx, 3), "median_current_a": round(med, 3),
                            "high_current_fraction": round(frac, 3)},
        evidence_window=_window_label(frame),
        potential_saving_kwh=None,
        potential_cost_saving=None,
        estimated_percent_reduction=None,
        source_stages=["stage1/history", "stage2/preprocessing"],
    )


def _get_horizon(fc_result: Any, key: str) -> Optional[dict]:
    if fc_result is None:
        return None
    h = None
    if hasattr(fc_result, key):
        h = getattr(fc_result, key)
    elif isinstance(fc_result, dict):
        h = fc_result.get(key)
    if isinstance(h, dict):
        return h
    if h is not None and hasattr(h, "status"):   # wrapped object
        return h
    return None


def detect_forecast_high_load(ev: MeterEvidence, rate: float, ts: int) -> Optional[Recommendation]:
    fc = ev.forecast_result
    h = _get_horizon(fc, "forecast_24h") or _get_horizon(fc, "forecast_7d")
    if not h or h.get("status") != "FORECAST":
        return None
    power = h.get("forecast_power_w")
    start = h.get("start_ts")
    if not power or start is None or len(power) == 0:
        return None
    slot = 300  # 5-minute cadence (24h=288, 7d=2016)
    f = pd.DataFrame({
        "timestamp": [int(start) + i * slot for i in range(len(power))],
        "power": list(power),
    })
    res = _recurring_high_window(f, "power", ratio=FORECAST_HIGH_RATIO)
    if not res:
        return None
    return Recommendation(
        pzem_number=ev.pzem_number, timestamp=ts,
        recommendation_type="RESPOND_PREDICTABLE_HIGH_LOAD", priority="MEDIUM",
        recommendation=RECOMMENDATION_TEXT["RESPOND_PREDICTABLE_HIGH_LOAD"],
        reason=f"Forecast shows a predictable high-load window {res['label']} "
               f"(median {res['peak_median']:.0f} W vs typical {res['overall_median']:.0f} W).",
        supporting_metrics={
            "window": res["label"],
            "forecast_peak_median_w": round(res["peak_median"], 2),
            "forecast_typical_median_w": round(res["overall_median"], 2),
            "confidence": h.get("confidence"),
        },
        evidence_window=f"{_fmt_ts(start)} (forecast)",
        potential_saving_kwh=None,
        potential_cost_saving=None,
        estimated_percent_reduction=None,
        source_stages=["stage9/forecast"],
    )


def detect_recurring_peak(ev: MeterEvidence, rate: float, ts: int) -> Optional[Recommendation]:
    frame = ev.feature_frame
    if frame is None:
        return None
    res = _recurring_high_window(frame, "power")
    if not res:
        return None
    priority = "HIGH" if res["peak_median"] > 2 * res["overall_median"] else "MEDIUM"
    return Recommendation(
        pzem_number=ev.pzem_number, timestamp=ts,
        recommendation_type="SHIFT_NON_CRITICAL_LOAD", priority=priority,
        recommendation=RECOMMENDATION_TEXT["SHIFT_NON_CRITICAL_LOAD"],
        reason=f"Recurring high-demand window {res['label']} "
               f"(median {res['peak_median']:.0f} W vs typical {res['overall_median']:.0f} W).",
        supporting_metrics={
            "window": res["label"],
            "peak_median_w": round(res["peak_median"], 2),
            "typical_median_w": round(res["overall_median"], 2),
        },
        evidence_window=_window_label(frame),
        potential_saving_kwh=None,
        potential_cost_saving=None,
        estimated_percent_reduction=None,
        source_stages=["stage1/history", "stage2/preprocessing"],
    )


def detect_peak_from_stage7(ev: MeterEvidence, rate: float, ts: int) -> Optional[Recommendation]:
    peak = ev.peak_result
    if peak is None:
        return None
    status = getattr(peak, "status", None)
    if status is None and isinstance(peak, dict):
        status = peak.get("status")
    if status != "PEAK_FOUND":
        return None
    above = getattr(peak, "peak_above_baseline_w", None)
    if above is None and isinstance(peak, dict):
        above = peak.get("peak_above_baseline_w")
    if above is None or above < PEAK_REDUCE_ABSOLUTE_W:
        return None
    pk = getattr(peak, "peak_power_w", None)
    if pk is None and isinstance(peak, dict):
        pk = peak.get("peak_power_w")
    base = getattr(peak, "baseline_power_w", None)
    if base is None and isinstance(peak, dict):
        base = peak.get("baseline_power_w")
    priority = "HIGH" if above > 2 * PEAK_REDUCE_ABSOLUTE_W else "MEDIUM"
    return Recommendation(
        pzem_number=ev.pzem_number, timestamp=ts,
        recommendation_type="REDUCE_PEAK_LOAD", priority=priority,
        recommendation=RECOMMENDATION_TEXT["REDUCE_PEAK_LOAD"],
        reason=f"Observed peak of {pk:.0f} W is {above:.0f} W above the {base:.0f} W baseline.",
        supporting_metrics={
            "peak_power_w": round(float(pk), 2) if pk is not None else None,
            "baseline_power_w": round(float(base), 2) if base is not None else None,
            "peak_above_baseline_w": round(float(above), 2),
        },
        evidence_window=None,
        potential_saving_kwh=None,
        potential_cost_saving=None,
        estimated_percent_reduction=None,
        source_stages=["stage7/peak_detection"],
    )


# ---------------------------------------------------------------------------
# Per-meter analysis + fleet assembly
# ---------------------------------------------------------------------------

def analyze_meter(ev: MeterEvidence, rate: float, ts: int) -> List[Recommendation]:
    """Run all detectors for ONE meter, de-duplicating the peak event so the
    Stage-7 peak and the history-derived recurring-peak don't double-count."""
    recs: List[Recommendation] = []
    is_system = ev.pzem_number is None

    if not is_system:
        r = detect_idle(ev, rate, ts)
        if r:
            recs.append(r)
        r = detect_pf(ev, rate, ts)
        if r:
            recs.append(r)
        r = detect_high_current(ev, rate, ts)
        if r:
            recs.append(r)

    r = detect_high_power(ev, rate, ts)
    if r:
        recs.append(r)

    peak7 = detect_peak_from_stage7(ev, rate, ts)
    recurring = detect_recurring_peak(ev, rate, ts)
    if peak7 and recurring:
        recs.append(peak7)          # concrete Stage-7 event wins
    elif recurring:
        recs.append(recurring)
    elif peak7:
        recs.append(peak7)

    r = detect_forecast_high_load(ev, rate, ts)
    if r:
        recs.append(r)
    return recs


def build_system_evidence(meters: Dict[int, MeterEvidence]) -> Optional[MeterEvidence]:
    """Aggregate ONLY valid PZEM power data into a SYSTEM feature_frame.
    Voltage/pf/current are not physically summable, so the system frame keeps
    power only; per-meter quantities stay in their own MeterEvidence."""
    frames = [ev.feature_frame for ev in meters.values()
              if ev.feature_frame is not None and not ev.feature_frame.empty
              and "timestamp" in ev.feature_frame and "power" in ev.feature_frame]
    if not frames:
        return None
    merged: Optional[pd.DataFrame] = None
    for i, f in enumerate(frames):
        sub = f[["timestamp", "power"]].rename(columns={"power": f"p{i}"})
        merged = sub if merged is None else merged.merge(sub, on="timestamp", how="outer")
    merged["power"] = merged.drop(columns=["timestamp"]).sum(axis=1, min_count=1)
    out = merged[["timestamp", "power"]].dropna(subset=["power"]).reset_index(drop=True)
    if out.empty:
        return None
    return MeterEvidence(pzem_number=None, feature_frame=out)


def compute_anchor(meters: Dict[int, MeterEvidence]) -> int:
    best: Optional[int] = None
    for ev in meters.values():
        if ev.feature_frame is not None and not ev.feature_frame.empty \
                and "timestamp" in ev.feature_frame:
            mx = ev.feature_frame["timestamp"].max()
            if pd.notna(mx):
                best = int(mx) if best is None else max(best, int(mx))
    return best if best is not None else int(time.time())


def generate_recommendations(meters: Dict[int, MeterEvidence],
                             rate: float = 0.0,
                             anchor_ts: Optional[int] = None) -> List[Recommendation]:
    """Generate per-PZEM and SYSTEM recommendations. Deterministic: identical
    input -> identical ordered output."""
    if anchor_ts is None:
        anchor_ts = compute_anchor(meters)

    recs: List[Recommendation] = []
    for n in sorted(meters):
        recs.extend(analyze_meter(meters[n], rate, anchor_ts))

    system_ev = build_system_evidence(meters)
    if system_ev is not None:
        recs.extend(analyze_meter(system_ev, rate, anchor_ts))

    # De-duplicate exact (meter, type) pairs, keep first.
    seen = set()
    out: List[Recommendation] = []
    for r in recs:
        key = (r.pzem_number, r.recommendation_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    out.sort(key=lambda r: (
        -PRIORITY_RANK.get(r.priority, 0),
        999 if r.pzem_number is None else r.pzem_number,
        r.recommendation_type,
    ))
    return out


# ---------------------------------------------------------------------------
# Firebase persistence (dedicated /ai/energy_saving hierarchy)
# ---------------------------------------------------------------------------

_firebase_ref_override = None


def set_firebase_ref_for_test(ref) -> None:
    """Swap the Firebase ref factory for a fake (tests only)."""
    global _firebase_ref_override
    _firebase_ref_override = ref


def _db_ref(path: str):
    if _firebase_ref_override is not None:
        return _firebase_ref_override(path)
    from firebase_admin import db
    _init_firebase()
    return db.reference(path)


_firebase_app = None


def _init_firebase():
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app
    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("firebase-admin is not installed.") from exc
    settings = get_settings()
    cred_path = settings.firebase_service_account_path
    from pathlib import Path
    if not Path(cred_path).exists():
        raise RuntimeError(f"Service account file not found at {cred_path}.")
    cred = credentials.Certificate(cred_path)
    _firebase_app = firebase_admin.initialize_app(
        cred, {"databaseURL": settings.firebase_database_url}
    )
    return _firebase_app


def build_energy_saving_payload(recs: List[Recommendation], anchor_ts: int,
                                rate: float = 0.0) -> dict:
    items = []
    for r in recs:
        items.append({
            "pzem_number": r.pzem_number,
            "timestamp": int(anchor_ts),
            "recommendation_type": r.recommendation_type,
            "priority": r.priority,
            "recommendation": r.recommendation,
            "reason": r.reason,
            "evidence": r.supporting_metrics,
            "potential_saving_kwh": (
                None if r.potential_saving_kwh is None else round(float(r.potential_saving_kwh), 4)
            ),
            "potential_cost_saving": (
                None if r.potential_cost_saving is None else round(float(r.potential_cost_saving), 4)
            ),
            "estimated_percent_reduction": (
                None if r.estimated_percent_reduction is None
                else round(float(r.estimated_percent_reduction), 2)
            ),
            "evidence_window": r.evidence_window,
            "source_stages": sorted(r.source_stages),
        })
    sources = sorted({s for r in recs for s in r.source_stages}) or ["stage11/energy_saving"]
    return {
        "timestamp": int(anchor_ts),
        "status": "RECOMMENDATIONS" if recs else "NO_RECOMMENDATION",
        "source_stages": sources,
        "rate_per_kwh": float(rate) if rate else 0.0,
        "recommendation_count": len(items),
        "recommendations": items,
        "note": ("Savings are ESTIMATES from historical/AI data, not guaranteed. "
                 "Validation against the real 30-day dataset is pending."),
    }


def write_energy_saving(recs: List[Recommendation], anchor_ts: int,
                        rate: float = 0.0, force: bool = False) -> dict:
    """Writes /ai/energy_saving/<anchor-ts>. Idempotent: same anchor key is
    skipped. Returns a status dict; NEVER raises on Firebase failure."""
    if anchor_ts is None:
        return {"written": False, "reason": "no_anchor"}
    key = str(int(anchor_ts))
    try:
        ref = _db_ref("ai/energy_saving")
        if ref.child(key).get() is not None and not force:
            return {"written": False, "reason": "exists", "key": key}
        payload = build_energy_saving_payload(recs, int(anchor_ts), rate)
        ref.child(key).set(payload)
        return {"written": True, "key": key, "count": len(recs)}
    except Exception as exc:  # noqa: BLE001 - pipeline resilience contract
        logger.error("Firebase write failed for ai/energy_saving/%s: %s", key, exc)
        return {"written": False, "reason": "firebase_error", "error": str(exc)}


def run_stage_11_pipeline(meters: Dict[int, MeterEvidence], rate: float = 0.0,
                          force: bool = False,
                          anchor_ts: Optional[int] = None) -> dict:
    """Full Stage 11 flow: generate + persist. Returns recommendations and the
    persist status. Designed to run AFTER Stages 7/8/9/10 in the existing flow."""
    recs = generate_recommendations(meters, rate=rate, anchor_ts=anchor_ts)
    if anchor_ts is None:
        anchor_ts = compute_anchor(meters)
    persist = write_energy_saving(recs, anchor_ts, rate=rate, force=force)
    return {"recommendations": recs, "anchor_timestamp": anchor_ts, "persist": persist}
