"""
ai/report_generator.py
-----------------------
STAGE 13: Automated MONTHLY Energy Report (PDF only).

Builds professional technical reports that SUMMARIZE the existing system data
and AI results (Stages 1-11). It creates NO new data pipeline: it consumes the
same Stage 2 feature frames and the existing Stage 3/4/7/8/9/10/11 result
objects that every other stage already produces.

Inputs consumed (all optional — missing data is omitted, never fabricated):
    - Stage 1/2 feature frames      (historical power/energy per PZEM)
    - Stage 3 anomalies             (list of records)
    - Stage 4 faults                (list of records, with severity)
    - Stage 7 peaks                 (PeakResult per PZEM)
    - Stage 8 maintenance risk      (RiskResult per PZEM)
    - Stage 9 forecast              (ForecastResult per PZEM)
    - Stage 10 bill prediction      (dict per PZEM / system)
    - Stage 11 energy-saving recs    (Recommendation list)

Outputs (deterministic filenames, written under reports/monthly, plus
latest.pdf copy):
    - PDF report only (print/PDF, minimal dependency-free writer; no HTML)

All statistics are computed from REAL inputs only; NaN/Infinity/empty values
are rendered as "—" / "No data available". No scheduler is implemented here
(Stage 14 will invoke these functions).
"""

from __future__ import annotations

import datetime
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from types import SimpleNamespace

import numpy as np
import pandas as pd

logger = logging.getLogger("ai.report_generator")

UTC = datetime.timezone.utc
DEFAULT_REPORT_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Dashboard Smart-Monitoring-System", "reports",
    )
)


# ---------------------------------------------------------------------------
# Safe numeric helpers (never emit NaN / Infinity)
# ---------------------------------------------------------------------------

def _safe(x: Any, nd: int = 2) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd)


def _fmt(x: Any, nd: int = 2, default: str = "—") -> str:
    s = _safe(x, nd)
    return default if s is None else f"{s:.{nd}f}"


def _get_ts(item: Any) -> Optional[int]:
    if isinstance(item, dict):
        ts = item.get("timestamp")
    else:
        ts = getattr(item, "timestamp", None)
    if ts is None:
        return None
    try:
        return int(ts)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Input container
# ---------------------------------------------------------------------------

@dataclass
class ReportInput:
    """Everything Stage 13 needs. All fields optional; absence => 'No data'."""
    pzem_count: int = 9
    frames: Dict[int, pd.DataFrame] = field(default_factory=dict)      # feature_frame
    anomalies: Dict[int, List[dict]] = field(default_factory=dict)    # records w/ timestamp
    faults: Dict[int, List[dict]] = field(default_factory=dict)       # records w/ severity
    peaks: Dict[int, Any] = field(default_factory=dict)               # PeakResult | dict
    risks: Dict[int, Any] = field(default_factory=dict)               # RiskResult | dict
    forecasts: Dict[int, Any] = field(default_factory=dict)           # ForecastResult | dict
    bills: Dict[int, dict] = field(default_factory=dict)              # per-PZEM bill dict
    system_bill: Optional[dict] = None
    recommendations: List[Any] = field(default_factory=list)           # Stage 11 Recommendation
    rate: float = 0.0
    ai_status: Dict[int, Any] = field(default_factory=dict)            # AI monitoring status per PZEM


# ---------------------------------------------------------------------------
# Period extraction
# ---------------------------------------------------------------------------

def _period_frame(frame: Optional[pd.DataFrame], start: int, end: int) -> Optional[pd.DataFrame]:
    if frame is None or frame.empty or "timestamp" not in frame:
        return None
    ts = frame["timestamp"]
    mask = (ts >= start) & (ts <= end)
    sub = frame[mask]
    return sub if not sub.empty else None


def _system_power(frames: Dict[int, pd.DataFrame], start: int, end: int) -> Optional[pd.Series]:
    rows = []
    for f in frames.values():
        pf = _period_frame(f, start, end)
        if pf is not None and not pf.empty and "power" in pf:
            rows.append(pf[["timestamp", "power"]])
    if not rows:
        return None
    allf = pd.concat(rows)
    return allf.groupby("timestamp")["power"].sum(min_count=1)


def _energy_kwh(series: Optional[pd.Series]) -> Optional[float]:
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    ts = np.array(sorted(s.index), dtype="float64")
    slot = float(np.median(np.diff(ts))) if len(ts) > 1 else 300.0
    wh = float(np.nansum(s.values)) / 1000.0 * (slot / 3600.0)
    return _safe(wh, 3)


def _pzem_stats(frame: Optional[pd.DataFrame], start: int, end: int) -> Optional[dict]:
    pf = _period_frame(frame, start, end)
    if pf is None or pf.empty or "power" not in pf:
        return None
    power = pf["power"].dropna()
    if power.empty:
        return None
    peak = float(power.max())
    peak_ts = int(pf.loc[power.idxmax(), "timestamp"])
    return {
        "energy_kwh": _energy_kwh(pf.set_index("timestamp")["power"]),
        "avg_power_w": _safe(float(power.mean()), 2),
        "peak_power_w": _safe(peak, 2),
        "peak_ts": peak_ts,
        "samples": int(power.shape[0]),
    }


def _count_in_period(items: Optional[List[dict]], start: int, end: int) -> Tuple[int, List[dict]]:
    count = 0
    listed: List[dict] = []
    for it in (items or []):
        ts = _get_ts(it)
        if ts is None:
            continue
        if start <= ts <= end:
            count += 1
            listed.append(it if isinstance(it, dict) else vars(it))
    return count, listed


def _peak_in_period(peak: Any, start: int, end: int) -> Optional[dict]:
    if peak is None:
        return None
    status = getattr(peak, "status", None) or (peak.get("status") if isinstance(peak, dict) else None)
    if status != "PEAK_FOUND":
        return None
    ts = getattr(peak, "peak_timestamp", None) or (peak.get("peak_timestamp") if isinstance(peak, dict) else None)
    if ts is None:
        return None
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return None
    if not (start <= ts <= end):
        return None
    return {
        "peak_power_w": _safe(getattr(peak, "peak_power_w", None), 2),
        "baseline_power_w": _safe(getattr(peak, "baseline_power_w", None), 2),
        "peak_above_baseline_w": _safe(getattr(peak, "peak_above_baseline_w", None), 2),
        "peak_ts": ts,
    }


def _risk_level(risk: Any) -> Optional[str]:
    if risk is None:
        return None
    lvl = getattr(risk, "risk_level", None) or (risk.get("risk_level") if isinstance(risk, dict) else None)
    return lvl


def _forecast_summary(fc: Any) -> dict:
    out = {"status": "NO_FORECAST", "confidence": None, "forecast_energy_kwh": None}
    if fc is None:
        return out
    h = None
    if hasattr(fc, "forecast_24h"):
        h = getattr(fc, "forecast_24h")
    elif isinstance(fc, dict):
        h = fc.get("forecast_24h")
    if isinstance(h, dict):
        out["status"] = h.get("status", "NO_FORECAST")
        out["confidence"] = h.get("confidence")
        pw = h.get("forecast_power_w")
        if isinstance(pw, (list, tuple)) and len(pw):
            slot = 300  # 5-min cadence
            out["forecast_energy_kwh"] = _safe(sum(pw) / 1000.0 * (slot / 3600.0), 3)
    return out


def _bill_summary(b: Optional[dict]) -> dict:
    if not b:
        return {"status": "NO_BILL", "estimated_bill": None,
                "estimated_total_energy_kwh": None, "actual_energy_kwh": None}
    return {
        "status": b.get("status", "UNKNOWN"),
        "estimated_bill": _safe(b.get("estimated_bill")),
        "estimated_total_energy_kwh": _safe(b.get("estimated_total_energy_kwh")),
        "actual_energy_kwh": _safe(b.get("actual_energy_kwh")),
        "forecast_energy_kwh": _safe(b.get("forecast_energy_kwh")),
        "reason": b.get("reason"),
    }


# ---------------------------------------------------------------------------
# Human-readable label mappings
# ---------------------------------------------------------------------------

RECOMMENDATION_LABELS = {
    "RESPOND_PREDICTABLE_HIGH_LOAD": "Predictable High-Usage Period",
    "SHIFT_NON_CRITICAL_LOAD": "Consider Shifting Non-Critical Loads",
    "REDUCE_IDLE_CONSUMPTION": "Reduce Unnecessary Standby Consumption",
    "IMPROVE_POWER_FACTOR": "Improve Power Factor",
    "REDUCE_HIGH_POWER": "Investigate High-Power Operation",
    "INVESTIGATE_HIGH_CURRENT": "Investigate Repeated High Current",
    "REDUCE_PEAK_LOAD": "Reduce Repeated Peak-Load Usage",
}

FAULT_TYPE_LABELS = {
    "OVER_VOLTAGE": "Over-voltage condition detected",
    "UNDER_VOLTAGE": "Under-voltage condition detected",
    "OVER_CURRENT": "Over-current condition detected",
    "POWER_FACTOR_DROP": "Power-factor drop detected",
    "FREQUENCY_DEVIATION": "Frequency deviation detected",
    "HIGH_POWER": "Abnormally high power detected",
    "COMMUNICATION_DEGRADED": "Communication degraded",
}

RISK_LABELS = {"HIGH": "High maintenance risk", "WATCH": "Watch", "LOW": "Low"}

SEVERITY_LABELS = {"EMERGENCY": "Emergency", "WARNING": "Warning", "ANOMALY": "Anomaly"}

PRIORITY_LABELS = {"HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}


def _recommendation_label(rtype: str) -> str:
    return RECOMMENDATION_LABELS.get(rtype, rtype)


def _fault_type_label(ftype: str) -> str:
    return FAULT_TYPE_LABELS.get(ftype, ftype)


def _risk_label(risk: str) -> str:
    return RISK_LABELS.get(risk, risk)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(data: ReportInput, start_ts: int, end_ts: int, kind: str) -> dict:
    """Build the structured report dict for a period [start_ts, end_ts].
    kind = 'daily' | 'monthly'. No Firebase / network access."""
    period_power = _system_power(data.frames, start_ts, end_ts)
    sys_stats = None
    if period_power is not None and not period_power.dropna().empty:
        p = period_power.dropna()
        sys_stats = {
            "energy_kwh": _energy_kwh(p),
            "avg_power_w": _safe(float(p.mean()), 2),
            "peak_power_w": _safe(float(p.max()), 2),
            "peak_ts": int(p.idxmax()),
            "_baseline_power_w": _safe(float(p.median()), 2),
        }

    pzem_rows = []
    active = 0
    for n in range(1, data.pzem_count + 1):
        st = _pzem_stats(data.frames.get(n), start_ts, end_ts)
        if st is None:
            pzem_rows.append({"pzem": n, "available": False})
            continue
        active += 1
        a_count, _ = _count_in_period(data.anomalies.get(n), start_ts, end_ts)
        f_count, f_list = _count_in_period(data.faults.get(n), start_ts, end_ts)
        emer = sum(1 for it in f_list
                   if (it.get("severity") if isinstance(it, dict) else getattr(it, "severity", None)) == "EMERGENCY")
        pzem_rows.append({
            "pzem": n,
            "available": True,
            "energy_kwh": st["energy_kwh"],
            "avg_power_w": st["avg_power_w"],
            "peak_power_w": st["peak_power_w"],
            "peak_ts": st["peak_ts"],
            "anomalies": a_count,
            "faults": f_count,
            "emergencies": emer,
            "risk_level": _risk_level(data.risks.get(n)),
            "peak": _peak_in_period(data.peaks.get(n), start_ts, end_ts),
            "forecast": _forecast_summary(data.forecasts.get(n)),
            "bill": _bill_summary(data.bills.get(n)),
        })

    a_total, a_list = _count_in_period(
        [a for lst in data.anomalies.values() for a in (lst or [])], start_ts, end_ts)
    f_total, f_list = _count_in_period(
        [fl for lst in data.faults.values() for fl in (lst or [])], start_ts, end_ts)
    emer_total = sum(1 for it in f_list
                     if (it.get("severity") if isinstance(it, dict) else getattr(it, "severity", None)) == "EMERGENCY")

    peaks_in_period = []
    for n in range(1, data.pzem_count + 1):
        pk = _peak_in_period(data.peaks.get(n), start_ts, end_ts)
        if pk:
            peaks_in_period.append({"pzem": n, **pk})

    risk_summary = {}
    for n in range(1, data.pzem_count + 1):
        lvl = _risk_level(data.risks.get(n))
        if lvl:
            risk_summary[lvl] = risk_summary.get(lvl, 0) + 1

    # AI insights
    ai = {
        "anomaly_count": a_total,
        "fault_count": f_total,
        "emergency_count": emer_total,
        "peaks": peaks_in_period,
        "risk_summary": risk_summary,
        "forecast": _forecast_summary(data.forecasts.get(0)) if data.forecasts.get(0) else None,
        "system_bill": _bill_summary(data.system_bill) if data.system_bill else None,
        "recommendations": [
            {
                "pzem_number": r.pzem_number,
                "priority": r.priority,
                "recommendation_type": r.recommendation_type,
                "reason": r.reason,
                "potential_saving_kwh": r.potential_saving_kwh,
                "potential_cost_saving": r.potential_cost_saving,
            }
            for r in (data.recommendations or [])
        ],
    }

    # Alert summary
    alerts = []
    for it in f_list:
        d = it if isinstance(it, dict) else vars(it)
        alerts.append({
            "pzem": d.get("pzem_number"),
            "severity": d.get("severity"),
            "type": d.get("fault_type") or d.get("type"),
            "timestamp": _get_ts(d),
        })
    alerts.sort(key=lambda x: (x.get("severity") != "EMERGENCY", x.get("timestamp") or 0), reverse=False)

    report = {
        "kind": kind,
        "period_start": start_ts,
        "period_end": end_ts,
        "generated_at": int(datetime.datetime.now(UTC).timestamp()),
        "rate": data.rate,
        "system": sys_stats,
        "active_pzem": active,
        "total_pzem": data.pzem_count,
        "pzem_rows": pzem_rows,
        "ai_insights": ai,
        "ai_status": data.ai_status,
        "alerts": alerts,
        "charts": _build_chart_data(data, start_ts, end_ts, kind),
    }
    return report


def _build_chart_data(data: ReportInput, start_ts: int, end_ts: int, kind: str) -> dict:
    charts: Dict[str, Any] = {}
    # Daily / monthly: system power (or energy) trend
    sp = _system_power(data.frames, start_ts, end_ts)
    if sp is not None and not sp.dropna().empty:
        s = sp.dropna().sort_index()
        if kind == "daily":
            charts["power_trend"] = {
                "labels": [str(datetime.datetime.fromtimestamp(int(t), UTC).strftime("%H:%M")) for t in s.index],
                "values": [float(v) for v in s.values],
                "title": "System Power Trend",
                "x_label": "Time",
                "y_label": "Power (kW)",
                "unit": "kW",
                "convert": lambda v: v / 1000.0,
            }
        else:
            # aggregate per calendar day
            day_tot = {}
            for t, v in s.items():
                d = datetime.datetime.fromtimestamp(int(t), UTC).strftime("%Y-%m-%d")
                day_tot[d] = day_tot.get(d, 0.0) + float(v)
            # energy per day estimate (slot 300s)
            day_energy = {d: e / 1000.0 * (300.0 / 3600.0) for d, e in day_tot.items()}
            charts["daily_energy_trend"] = {
                "labels": list(day_energy.keys()),
                "values": [round(v, 3) for v in day_energy.values()],
                "title": "Daily Energy Consumption",
                "x_label": "Date",
                "y_label": "Energy (kWh)",
                "unit": "kWh",
                "convert": lambda v: v,
            }
            # peak trend per day
            day_peak = {}
            for t, v in s.items():
                d = datetime.datetime.fromtimestamp(int(t), UTC).strftime("%Y-%m-%d")
                day_peak[d] = max(day_peak.get(d, 0.0), float(v))
            charts["peak_trend"] = {
                "labels": list(day_peak.keys()),
                "values": [round(v / 1000.0, 2) for v in day_peak.values()],
                "title": "Daily Peak Power",
                "x_label": "Date",
                "y_label": "Peak Power (kW)",
                "unit": "kW",
                "convert": lambda v: v / 1000.0,
            }

    # PZEM energy comparison (period)
    pzem_energy = []
    for n in range(1, data.pzem_count + 1):
        st = _pzem_stats(data.frames.get(n), start_ts, end_ts)
        if st and st["energy_kwh"] is not None:
            pzem_energy.append((f"PZEM {n}", st["energy_kwh"]))
    if pzem_energy:
        pzem_energy.sort(key=lambda x: -x[1])
        charts["pzem_energy"] = {
            "labels": [x[0] for x in pzem_energy],
            "values": [round(x[1], 3) for x in pzem_energy],
            "title": "Energy Consumption by Meter",
            "x_label": "Meter",
            "y_label": "Energy (kWh)",
            "unit": "kWh",
            "convert": lambda v: v,
        }

    # anomaly / fault summary per meter
    pzem_a = {}
    pzem_f = {}
    for n in range(1, data.pzem_count + 1):
        pzem_a[n] = len([a for a in (data.anomalies.get(n) or []) if _get_ts(a) and start_ts <= _get_ts(a) <= end_ts])
        pzem_f[n] = len([fl for fl in (data.faults.get(n) or []) if _get_ts(fl) and start_ts <= _get_ts(fl) <= end_ts])
    a_total = sum(pzem_a.values())
    f_total = sum(pzem_f.values())
    if a_total or f_total:
        # Only include meters with data
        active = [n for n in range(1, data.pzem_count + 1) if pzem_a.get(n, 0) > 0 or pzem_f.get(n, 0) > 0]
        if not active:
            active = list(range(1, min(data.pzem_count + 1, 6)))
        charts["event_summary"] = {
            "labels": [f"PZEM {n}" for n in active],
            "series": {
                "Anomalies": [pzem_a[n] for n in active],
                "Faults": [pzem_f[n] for n in active],
            },
            "title": "Anomalies and Faults by Meter",
            "x_label": "Meter",
            "y_label": "Count",
            "unit": "count",
            "convert": lambda v: v,
        }
    return charts


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Minimal dependency-free PDF writer
# ---------------------------------------------------------------------------

class _PDF:
    def __init__(self):
        self.pages: List[List[str]] = []
        self.cur: List[str] = []
        self.W, self.H, self.M = 612, 792, 50
        self.y = self.H - self.M

    @staticmethod
    def _s(text: str) -> str:
        repl = {"—": "-", "–": "-", "₹": "Rs", "·": "*", "≥": ">=", "≤": "<=",
                "’": "'", "‘": "'", "“": '"', "”": '"', "•": "-", "…": "..."}
        for k, v in repl.items():
            text = text.replace(k, v)
        return text.encode("latin-1", "replace").decode("latin-1")

    @staticmethod
    def _wrap(text: str, cpl: int) -> List[str]:
        text = _PDF._s(text)
        words = str(text).split()
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= cpl or not cur:
                cur = (cur + " " + w).strip()
            else:
                lines.append(cur); cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    def _space(self, h: float):
        if self.y - h < self.M:
            self.pages.append(self.cur); self.cur = []; self.y = self.H - self.M

    def text(self, s: str, size: int = 10, bold: bool = False, indent: int = 0):
        cpl = max(8, int((self.W - 2 * self.M - indent) / (size * 0.5)))
        for ln in self._wrap(s, cpl):
            self._space(size * 1.35)
            self.y -= size * 1.2
            esc = ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            font = "F2" if bold else "F1"
            self.cur.append(f"BT /{font} {size} Tf {self.M + indent} {self.y:.1f} Td ({esc}) Tj ET")

    def heading(self, s: str, size: int = 14):
        self._space(size * 2)
        self.y -= size
        self.text(s, size=size, bold=True)

    def spacer(self, h: int = 6):
        self.y -= h

    def table(self, headers: List[str], rows: List[List[str]], widths: List[int]):
        self._space(14)
        self.y -= 11
        head = "  ".join(_PDF._s(h).ljust(w)[:w] for h, w in zip(headers, widths))
        self.cur.append(f"BT /F2 9 Tf {self.M} {self.y:.1f} Td ({head.replace('(', '\\(').replace(')', '\\)')}) Tj ET")
        self.cur.append(f"BT /F1 9 Tf {self.M} {self.y-12:.1f} Td ( ) Tj ET")
        for row in rows:
            cells = [_PDF._s(str(c))[:w] for c, w in zip(row, widths)]
            line = "  ".join(c.ljust(w) for c, w in zip(cells, widths))
            self._space(11)
            self.y -= 11
            esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            self.cur.append(f"BT /F1 9 Tf {self.M} {self.y:.1f} Td ({esc}) Tj ET")

    # ---- Advanced chart rendering ----

    def chart(self, chart: dict):
        """Render a chart dict with title, labels, values, and optional axis labels.
        Supports bar and line kinds, single and multi-series."""
        title = chart.get("title", "")
        labels = chart.get("labels", [])
        values = chart.get("values", [])
        series = chart.get("series")
        x_label = chart.get("x_label", "")
        y_label = chart.get("y_label", "")
        unit = chart.get("unit", "")
        convert = chart.get("convert", lambda v: v)
        kind = "line" if "trend" in title.lower() or "peak" in title.lower() else "bar"

        # Determine if multi-series
        is_multi = series is not None and len(series) > 0
        if is_multi:
            series_labels = list(series.keys())
            series_data = list(series.values())
            n = len(labels)
        else:
            series_labels = ["Value"]
            series_data = [values]
            n = len(values)

        if n == 0:
            return

        # Chart layout constants
        page_w = self.W - 2 * self.M  # 512
        page_h = 220  # chart height
        top_margin = 28  # for title
        bottom_margin = 40  # for x-axis labels
        left_margin = 50  # for y-axis labels
        right_margin = 10  # extra
        left_label_w = 45  # y-axis label area
        legend_h = 18  # legend height

        chart_top = self.y - top_margin
        chart_bottom = chart_top - page_h
        x_axis_y = chart_bottom + 5
        left_x = self.M + left_label_w
        right_x = self.M + page_w - right_margin
        chart_w = right_x - left_x
        chart_h = chart_bottom - x_axis_y

        self._space(top_margin + page_h + bottom_margin + 10)

        # Draw title
        t = _PDF._s(title)
        self.cur.append(f"BT /F2 11 Tf {self.M} {chart_top + 2:.1f} Td ({t}) Tj ET")
        self.y = chart_bottom - 8

        # Draw axes
        self.cur.append("0 0 0 RG 0.5 w")
        # Y-axis
        self.cur.append(f"{left_x:.1f} {x_axis_y:.1f} m {left_x:.1f} {chart_bottom:.1f} l S")
        # X-axis
        self.cur.append(f"{left_x:.1f} {x_axis_y:.1f} m {right_x:.1f} {x_axis_y:.1f} l S")
        # Y-axis arrow
        self.cur.append(f"{left_x:.1f} {chart_bottom:.1f} m {left_x-3:.1f} {chart_bottom-4:.1f} l S")
        self.cur.append(f"{left_x:.1f} {chart_bottom:.1f} m {left_x+3:.1f} {chart_bottom-4:.1f} l S")
        # X-axis arrow
        self.cur.append(f"{right_x:.1f} {x_axis_y:.1f} m {right_x-4:.1f} {x_axis_y+3:.1f} l S")
        self.cur.append(f"{right_x:.1f} {x_axis_y:.1f} m {right_x-4:.1f} {x_axis_y-3:.1f} l S")

        # Y-axis tick marks and labels
        n_ticks = 5
        all_vals = []
        for sdata in series_data:
            all_vals.extend(sdata)
        vmax = max(all_vals) if all_vals else 1.0
        vmin = 0
        if vmax <= 0:
            vmax = 1.0
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            val = vmin + (vmax - vmin) * frac
            y_pos = x_axis_y + chart_h * frac
            # tick mark
            self.cur.append(f"{left_x:.1f} {y_pos:.1f} m {left_x-3:.1f} {y_pos:.1f} l S")
            # label
            label_val = convert(val)
            label_str = f"{label_val:.2f}"
            escaped = _PDF._s(label_str)
            self.cur.append(f"BT /F1 7 Tf {left_x - 8:.1f} {y_pos + 2:.1f} Td ({escaped}) Tj ET")

        # Y-axis label
        y_label_s = _PDF._s(y_label)
        self.cur.append(f"BT /F1 8 Tf {self.M} {(chart_top + chart_bottom) / 2:.1f} Td ({y_label_s}) Tj ET")

        # X-axis tick labels
        tick_step = max(1, n // 8)  # limit number of labels
        for i in range(n):
            if i % tick_step != 0 and i != n - 1:
                continue
            x_pos = left_x + (i + 0.5) * chart_w / n
            label_s = _PDF._s(str(labels[i])[:8])
            self.cur.append(f"BT /F1 7 Tf {x_pos - 10:.1f} {x_axis_y + 10:.1f} Td ({label_s}) Tj ET")
        # X-axis label
        x_label_s = _PDF._s(x_label)
        self.cur.append(f"BT /F1 8 Tf {self.M + page_w / 2 - 20:.1f} {x_axis_y + 20:.1f} Td ({x_label_s}) Tj ET")

        # Draw data
        colors = ["0.14 0.38 0.92", "0.92 0.38 0.14", "0.38 0.92 0.14", "0.92 0.14 0.38", "0.14 0.92 0.38"]
        if is_multi:
            # Grouped bar chart
            n_series = len(series_data)
            group_w = chart_w / n
            bar_w = group_w / (n_series * 1.5)
            for si, (sname, sdata) in enumerate(zip(series_labels, series_data)):
                color = colors[si % len(colors)]
                self.cur.append(f"{color} rg")
                for i, v in enumerate(sdata):
                    bh = chart_h * (v / vmax) if vmax > 0 else 0
                    x = left_x + i * group_w + group_w * 0.1 + si * bar_w
                    self.cur.append(f"{x:.1f} {x_axis_y:.1f} {bar_w:.1f} {bh:.1f} re f")
                    # Value label
                    if v > 0:
                        val_str = _PDF._s(f"{v:.1f}")
                        self.cur.append(f"BT /F1 6 Tf {x + bar_w/2:.1f} {x_axis_y + bh + 8:.1f} Td ({val_str}) Tj ET")
        else:
            # Single series
            color = colors[0]
            self.cur.append(f"{color} rg")
            for i, v in enumerate(series_data[0]):
                bh = chart_h * (convert(v) / vmax) if vmax > 0 else 0
                x = left_x + (i + 0.5) * chart_w / n - (chart_w / n) * 0.35
                bw = chart_w / n * 0.7
                self.cur.append(f"{x:.1f} {x_axis_y:.1f} {bw:.1f} {bh:.1f} re f")
                # Value label
                val = convert(v)
                if val > 0:
                    val_str = _PDF._s(f"{val:.2f}")
                    self.cur.append(f"BT /F1 6 Tf {x + bw/2:.1f} {x_axis_y + bh + 8:.1f} Td ({val_str}) Tj ET")

        # Legend for multi-series
        if is_multi:
            leg_y = chart_bottom + 25
            leg_x = right_x - 120
            self.cur.append(f"BT /F1 8 Tf {leg_x:.1f} {leg_y:.1f} Td (Legend:) Tj ET")
            for si, sname in enumerate(series_labels):
                color = colors[si % len(colors)]
                cx = leg_x + 45 + si * 60
                self.cur.append(f"{color} rg")
                self.cur.append(f"{cx:.1f} {leg_y + 2:.1f} m {cx + 6:.1f} {leg_y + 2:.1f} l {cx + 6:.1f} {leg_y - 4:.1f} l {cx:.1f} {leg_y - 4:.1f} l S")
                self.cur.append(f"BT /F1 7 Tf {cx + 10:.1f} {leg_y - 2:.1f} Td ({_PDF._s(sname)}) Tj ET")

    def save(self, path: str):
        if self.cur:
            self.pages.append(self.cur); self.cur = []
        catalog, pages_id, f1, f2 = 1, 2, 3, 4
        page_ids, content_ids = [], []
        nxt = 5
        for _ in self.pages:
            page_ids.append(nxt); nxt += 1
            content_ids.append(nxt); nxt += 1
        objs: Dict[int, str] = {}
        objs[catalog] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>"
        kids = " ".join(f"{p} 0 R" for p in page_ids)
        objs[pages_id] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"
        objs[f1] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        objs[f2] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"
        for i, pg in enumerate(self.pages):
            stream = "\n".join(pg)
            cid = content_ids[i]
            pid = page_ids[i]
            objs[cid] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
            objs[pid] = (f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
                         f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> >> "
                         f"/Contents {cid} 0 R >>")
        data = "%PDF-1.4\n"
        offsets = {0: 0}
        for oid in range(1, nxt):
            s = objs.get(oid, "")
            objstr = f"{oid} 0 obj\n{s}\nendobj\n"
            offsets[oid] = len(data)
            data += objstr
        xref_pos = len(data)
        xref = f"xref\n0 {nxt}\n0000000000 65535 f \n"
        for oid in range(1, nxt):
            xref += f"{offsets[oid]:010d} 00000 n \n"
        data += xref + f"trailer\n<< /Size {nxt} /Root {catalog} 0 R >>\nstartxref\n{xref_pos}\n%%EOF"
        with open(path, "wb") as f:
            f.write(data.encode("latin-1"))


def render_pdf(report: dict, path: str) -> None:
    period = datetime.datetime.fromtimestamp(report["period_start"], UTC).strftime("%B %Y")
    gen = datetime.datetime.fromtimestamp(report["generated_at"], UTC).strftime("%Y-%m-%d %H:%M UTC")
    rate = report["rate"]
    pdf = _PDF()
    pdf.heading("MONTHLY ENERGY PERFORMANCE REPORT", size=18)
    pdf.spacer(4)
    pdf.text(f"Month: {period}", size=11, bold=True)
    pdf.text(f"Report period: {datetime.datetime.fromtimestamp(report['period_start'], UTC).strftime('%d %B %Y')} – {datetime.datetime.fromtimestamp(report['period_end'] - 1, UTC).strftime('%d %B %Y')}", size=10)
    pdf.text(f"Generated on: {gen}", size=10)
    pdf.text(f"Electricity rate: \u20b9{_fmt(rate)} / kWh", size=10)
    pdf.spacer(6)

    sys = report["system"]
    ai = report["ai_insights"]
    charts = report["charts"]

    # ---- Executive Summary ----
    pdf.heading("Executive Summary", size=14)
    if sys:
        pdf.text(f"Total Energy Used: {_fmt(sys['energy_kwh'])} kWh", size=11, bold=True)
        pdf.text(f"Average Power: {_fmt(sys['avg_power_w'] / 1000.0)} kW", size=10)
        pdf.text(f"Peak Power: {_fmt(sys['peak_power_w'] / 1000.0)} kW", size=10)
        pk_ts = sys['peak_ts']
        pdf.text(f"Peak Time: {datetime.datetime.fromtimestamp(pk_ts, UTC).strftime('%d %B %Y, %H:%M')} UTC", size=10)
    else:
        pdf.text("Total Energy Used: No data", size=11)
    pdf.text(f"Monitored Meters: {report['active_pzem']} / {report['total_pzem']}", size=10)
    pdf.text(f"Alerts: {ai['anomaly_count'] + ai['fault_count']}  Anomalies: {ai['anomaly_count']}  Faults: {ai['fault_count']}", size=10)
    risk_lines = []
    for lvl, cnt in ai['risk_summary'].items():
        risk_lines.append(f"{cnt} {_risk_label(lvl).lower()}")
    pdf.text(f"High-Risk Meters: {ai['risk_summary'].get('HIGH', 0)}", size=10)
    pdf.spacer(6)

    # ---- What Happened This Month? ----
    pdf.heading("What Happened This Month?", size=14)
    if sys:
        total_kwh = _fmt(sys['energy_kwh'])
        peak_kw = _fmt(sys['peak_power_w'] / 1000.0)
        anom = ai['anomaly_count']
        fault = ai['fault_count']
        high_risk = ai['risk_summary'].get('HIGH', 0)
        watch_risk = ai['risk_summary'].get('WATCH', 0)
        parts = [f"During {period}, the system consumed {total_kwh} kWh of electricity.",
                 f"The highest recorded demand was {peak_kw} kW.",
                 f"{anom} anomalies and {fault} faults were detected."]
        if high_risk > 0 or watch_risk > 0:
            risk_parts = []
            if high_risk > 0:
                risk_parts.append(f"{high_risk} meters classified as high maintenance risk")
            if watch_risk > 0:
                risk_parts.append(f"{watch_risk} meters classified as watch")
            parts.append(f"{'Two meters' if len(risk_parts) == 2 else 'Meters'} were classified as {'high' if high_risk > 0 else 'watch'} maintenance risk: {' and '.join(risk_parts)}.")
        narrative = " ".join(parts)
        pdf.text(narrative, size=10)
    else:
        pdf.text("No system data was available for this report period.", size=10)
    pdf.spacer(8)

    # ---- System Performance ----
    pdf.heading("System Performance", size=14)
    headers = ["Metric", "Value", "Explanation"]
    widths = [28, 20, 28]
    rows = []
    if sys:
        rows.append(["Total Energy Used", f"{_fmt(sys['energy_kwh'])} kWh", "Total electrical energy consumed across all meters"])
        rows.append(["Average Power", f"{_fmt(sys['avg_power_w'] / 1000.0)} kW", "Mean power draw over the reporting period"])
        rows.append(["Peak Power", f"{_fmt(sys['peak_power_w'] / 1000.0)} kW", "Highest instantaneous power recorded"])
        pk_ts = sys['peak_ts']
        rows.append(["Peak Time", datetime.datetime.fromtimestamp(pk_ts, UTC).strftime('%d %B %Y, %H:%M UTC'), "Timestamp of the peak demand event"])
    else:
        rows.append(["Total Energy Used", "\u2014", "No data available"])
        rows.append(["Average Power", "\u2014", "No data available"])
        rows.append(["Peak Power", "\u2014", "No data available"])
        rows.append(["Peak Time", "\u2014", "No data available"])
    rows.append(["Active Meters", f"{report['active_pzem']} / {report['total_pzem']}", f"PZEM meters reporting data out of {report['total_pzem']} configured"])
    pdf.table(headers, rows, widths)
    pdf.spacer(8)

    # ---- Meter-wise Energy Summary ----
    pdf.heading("Meter-wise Energy Consumption", size=14)
    headers = ["Meter", "Energy Used (kWh)", "Peak Power (kW)", "Anomalies", "Faults", "Maintenance Risk", "Forecast", "Bill Status"]
    widths = [12, 18, 18, 12, 10, 18, 12, 12]
    rows = []
    for r in report["pzem_rows"]:
        if r["available"]:
            peak_kw = _fmt(r['peak_power_w'] / 1000.0) if r['peak_power_w'] is not None else "\u2014"
            risk = _risk_label(r['risk_level']) if r['risk_level'] else "\u2014"
            rows.append([f"PZEM {r['pzem']}", _fmt(r['energy_kwh']), peak_kw,
                         str(r['anomalies']), str(r['faults']), risk,
                         str(r['forecast']['status']), str(r['bill']['status'])])
        else:
            rows.append([f"PZEM {r['pzem']}", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
    pdf.table(headers, rows, widths)
    pdf.spacer(8)

    # ---- Peak Demand Analysis ----
    # System peak MUST come from sys_stats (simultaneous aligned PZEM samples)
    # NOT from ai['peaks'] which are per-PZEM individual maxima
    pdf.heading("Peak Demand Analysis", size=14)
    if sys:
        pdf.text(f"Maximum system power: {_fmt(sys['peak_power_w'] / 1000.0)} kW", size=10)
        pk_ts = sys['peak_ts']
        pdf.text(f"Peak date: {datetime.datetime.fromtimestamp(pk_ts, UTC).strftime('%d %B %Y')}", size=10)
        pdf.text(f"Peak time: {datetime.datetime.fromtimestamp(pk_ts, UTC).strftime('%H:%M')} UTC", size=10)
        # Baseline is the median of the aligned system power series
        # This comes from sys_stats computation in build_report
        baseline_w = None
        if sys.get("_baseline_power_w") is not None:
            baseline_w = sys["_baseline_power_w"]
        above_baseline = (sys['peak_power_w'] - baseline_w) if baseline_w else None
        pdf.text(f"Peak above baseline: {_fmt(above_baseline / 1000.0) if above_baseline is not None else '\u2014'} kW", size=10)
        pdf.text(f"Baseline (median) power: {_fmt(baseline_w / 1000.0) if baseline_w else '\u2014'} kW", size=10)
    else:
        pdf.text("No peak demand data available for this period.", size=10)
    pdf.text("System peak is computed from timestamp-aligned simultaneous PZEM readings (Stage 3.1 aggregation).", size=9)
    pdf.spacer(8)

    # ---- Alerts & Abnormal Events ----
    pdf.heading("Alerts & Abnormal Events", size=14)
    pdf.text(f"Summary: {ai['anomaly_count']} anomalies, {ai['fault_count']} faults, {ai['emergency_count']} emergencies", size=10)
    pdf.spacer(4)
    if report["alerts"]:
        for a in report["alerts"]:
            ft = _fault_type_label(a.get('type', '') or '')
            sev = SEVERITY_LABELS.get(a.get('severity', ''), a.get('severity', ''))
            t = datetime.datetime.fromtimestamp(a["timestamp"], UTC).strftime('%d %B %Y') if a["timestamp"] else "\u2014"
            who = f"PZEM {a['pzem']}" if a['pzem'] else "SYSTEM"
            pdf.text(f"{who} \u2013 {ft}", size=10, bold=True)
            pdf.text(f"Date: {t}", size=9, indent=6)
            pdf.text(f"Status: {sev}", size=9, indent=6)
    else:
        pdf.text("No anomalies or faults were detected in this period.", size=10)
    pdf.spacer(8)

    # ---- Maintenance Risk ----
    pdf.heading("Maintenance Risk", size=14)
    risk_summary = ai['risk_summary']
    if risk_summary:
        for lvl in ["HIGH", "WATCH", "LOW"]:
            cnt = risk_summary.get(lvl, 0)
            if cnt > 0:
                pdf.text(f"- {cnt} meters: {_risk_label(lvl)}", size=10)
        high_meters = [r['pzem'] for r in report["pzem_rows"] if r.get("risk_level") == "HIGH"]
        watch_meters = [r['pzem'] for r in report["pzem_rows"] if r.get("risk_level") == "WATCH"]
        if high_meters:
            pdf.text(f"High-risk meters: {', '.join(f'PZEM {m}' for m in high_meters)}", size=9, indent=6)
        if watch_meters:
            pdf.text(f"Watch meters: {', '.join(f'PZEM {m}' for m in watch_meters)}", size=9, indent=6)
    else:
        pdf.text("No maintenance risk data available for this period.", size=10)
    pdf.spacer(8)

    # ---- Energy-Saving Recommendations ----
    pdf.heading("Energy-Saving Recommendations", size=14)
    if ai['recommendations']:
        for r in ai['recommendations']:
            label = _recommendation_label(r['recommendation_type'])
            who = "SYSTEM" if r['pzem_number'] is None else f"PZEM {r['pzem_number']}"
            pdf.text(f"{who} \u2013 {label}", size=11, bold=True)
            pdf.text(f"Priority: {PRIORITY_LABELS.get(r['priority'], r['priority'])}", size=9, indent=6)
            pdf.text(f"Observation: {r['reason']}", size=9, indent=6)
            if r['potential_saving_kwh'] is not None:
                pdf.text(f"Estimated saving: {_fmt(r['potential_saving_kwh'])} kWh", size=9, indent=6)
            if r['potential_cost_saving'] is not None and rate > 0:
                pdf.text(f"Estimated cost impact: \u20b9{_fmt(r['potential_cost_saving'])}", size=9, indent=6)
            pdf.spacer(2)
    else:
        pdf.text("No energy-saving recommendations available for this period.", size=10)
    pdf.spacer(8)

    # ---- Energy Cost ----
    # Bill must use same semantics as dashboard: actual_energy × rate
    # estimated_total_energy includes forecast; actual_energy is what's consumed
    pdf.heading("Energy Cost", size=14)
    pdf.text(f"Electricity rate: \u20b9{_fmt(rate)} / kWh", size=10)
    if sys and sys['energy_kwh'] is not None:
        pdf.text(f"Energy consumed (actual): {_fmt(sys['energy_kwh'])} kWh", size=10)
        actual_bill = sys['energy_kwh'] * rate if rate > 0 else None
        pdf.text(f"Estimated cost (actual): \u20b9{_fmt(actual_bill)}", size=10)
    if ai['system_bill'] and ai['system_bill']['estimated_bill'] is not None:
        sb = ai['system_bill']
        pdf.text(f"Estimated total energy (actual + forecast): {_fmt(sb['estimated_total_energy_kwh'])} kWh", size=10)
        pdf.text(f"Forecast energy: {_fmt(sb.get('forecast_energy_kwh', 0))} kWh", size=10)
        pdf.text(f"Estimated bill (total): \u20b9{_fmt(sb['estimated_bill'])}", size=10)
        pdf.text(f"Bill status: {sb.get('status', 'UNKNOWN')}", size=10)
        if sb.get('reason'):
            pdf.text(f"Bill reason: {sb['reason']}", size=9)
    else:
        pdf.text("Bill prediction: Not available for this period.", size=10)
    pdf.spacer(8)

    # ---- Energy Forecast ----
    pdf.heading("Energy Forecast", size=14)
    if ai['forecast']:
        fc = ai['forecast']
        if fc.get('status') == 'FORECAST':
            pdf.text(f"Forecast status: {fc['status']}", size=10)
            pdf.text(f"Confidence: {fc.get('confidence', '\u2014')}", size=10)
            if fc.get('forecast_energy_kwh') is not None:
                pdf.text(f"Forecast energy: {_fmt(fc['forecast_energy_kwh'])} kWh", size=10)
        else:
            pdf.text("Forecast unavailable", size=10)
            pdf.text("Insufficient forecast data was available for the system during this report period.", size=9)
    else:
        pdf.text("Forecast unavailable", size=10)
        pdf.text("Insufficient forecast data was available for the system during this report period.", size=9)
    pdf.spacer(8)

    # ---- Charts ----
    if charts:
        pdf.heading("Charts", size=14)
        for c in charts.values():
            pdf.chart(c)
            # Interpretation text
            title = c.get("title", "")
            labels = c.get("labels", [])
            values = c.get("values", [])
            series = c.get("series")
            unit = c.get("unit", "")
            if "energy" in title.lower() and values:
                max_val = max(values)
                max_idx = values.index(max_val)
                pdf.text(f"Highest daily energy consumption: {_fmt(max_val)} {unit} on {labels[max_idx]}.", size=9)
            elif "peak" in title.lower() and values:
                max_val = max(values)
                max_idx = values.index(max_val)
                pdf.text(f"Highest daily peak: {_fmt(max_val)} {unit} on {labels[max_idx]}.", size=9)
            elif "by meter" in title.lower() and values:
                sorted_items = sorted(zip(labels, values), key=lambda x: -x[1])
                top2 = sorted_items[:2]
                pdf.text("Higher energy consumption was recorded by:", size=9)
                for meter, val in top2:
                    pdf.text(f"  {meter} - {_fmt(val)} {unit}", size=9, indent=6)
            elif "event" in title.lower() and series:
                for sname, sdata in series.items():
                    for i, v in enumerate(sdata):
                        if v > 0:
                            pdf.text(f"{labels[i]}: {sname} = {v}", size=9)
            pdf.spacer(4)
            # Key values table
            if "by meter" in title.lower() and values and len(values) > 0:
                sorted_items = sorted(zip(labels, values), key=lambda x: -x[1])[:5]
                pdf.text("Top Energy Consumers:", size=9, bold=True)
                pdf.table(["Meter", "Energy"],
                          [[m, f"{_fmt(v)} {unit}"] for m, v in sorted_items],
                          [14, 14])
            elif "peak" in title.lower() and values and len(values) > 0:
                max_val = max(values)
                max_idx = values.index(max_val)
                pdf.text(f"Key value: {labels[max_idx]} = {_fmt(max_val)} {unit}", size=9)
            pdf.spacer(4)
    pdf.spacer(8)

    # ---- Technical Details / Appendix ----
    pdf.heading("Technical Details / Appendix", size=14)
    pdf.text("Internal recommendation codes (traceability):", size=9)
    for r in ai['recommendations']:
        code = r['recommendation_type']
        who = "SYSTEM" if r['pzem_number'] is None else f"PZEM {r['pzem_number']}"
        pdf.text(f"  {code} - {who} - {PRIORITY_LABELS.get(r['priority'], r['priority'])}", size=8, indent=6)
    pdf.text("Fault types (internal codes):", size=9)
    for a in report["alerts"]:
        pdf.text(f"  {a.get('type', '\u2014')} - PZEM {a['pzem']} - {a.get('severity', '\u2014')}", size=8, indent=6)
    pdf.text("Report generated by Smart Energy Monitoring System.", size=8)
    pdf.save(path)


# ---------------------------------------------------------------------------
# Orchestration / filenames
# ---------------------------------------------------------------------------

def _month_bounds(year: int, month: int) -> Tuple[int, int]:
    start = datetime.datetime(year, month, 1, tzinfo=UTC).timestamp()
    if month == 12:
        nxt = datetime.datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        nxt = datetime.datetime(year, month + 1, 1, tzinfo=UTC)
    return int(start), int(nxt.timestamp())


def _file_stub(start_ts: int) -> str:
    d = datetime.datetime.fromtimestamp(start_ts, UTC)
    return f"report-{d.strftime('%Y-%m')}"


def generate_monthly_report(data: Optional[ReportInput] = None,
                            year: Optional[int] = None, month: Optional[int] = None,
                            output_dir: Optional[str] = None,
                            rate: float = 0.0) -> dict:
    """Generate the MONTHLY PDF report only (HTML is no longer produced).
    Stage 14 will call this on a schedule."""
    if data is None:
        data = demo_input()
    if rate:
        data.rate = rate
    now = datetime.datetime.now(UTC)
    year = year or now.year
    month = month or now.month
    start, end = _month_bounds(year, month)
    report = build_report(data, start, end, "monthly")
    out_dir = os.path.join(output_dir or DEFAULT_REPORT_ROOT, "monthly")
    os.makedirs(out_dir, exist_ok=True)
    stub = _file_stub(start)
    pdf_path = os.path.join(out_dir, stub + ".pdf")
    render_pdf(report, pdf_path)
    shutil.copyfile(pdf_path, os.path.join(out_dir, "latest.pdf"))
    return {"report": report, "pdf": pdf_path, "stub": stub}


# ---------------------------------------------------------------------------
# Deterministic demo input (used for CLI fallback & offline demonstration)
# ---------------------------------------------------------------------------

def build_report_input_from_pipelines(settings=None) -> ReportInput:
    """Reuse the EXISTING AI pipelines to build a report input (no new load).
    Raises on any failure so the caller can fall back to demo_input()."""
    from ai import (anomaly_detection, preprocessing, forecast, peak_detection,
                    maintenance_risk, fault_diagnosis, energy_saving)
    from ai.bill_prediction import (compute_actual_energy_from_history,
                                    predict_bill_from_record)

    rate = float(os.environ.get("BILL_RATE_PER_KWH", "0.0") or "0.0")
    preprocess_results = preprocessing.run_preprocessing_pipeline(settings=settings)
    anomaly_results = anomaly_detection.run_anomaly_detection_pipeline(
        settings=settings, preprocess_results=preprocess_results)
    fault_results = fault_diagnosis.run_fault_diagnosis_pipeline(
        preprocess_results=preprocess_results, settings=settings)
    peaks, _ = peak_detection.run_peak_detection_pipeline(
        settings=settings, preprocess_results=preprocess_results)
    risks, _ = maintenance_risk.run_maintenance_risk_pipeline(
        settings=settings, preprocess_results=preprocess_results,
        anomaly_results=anomaly_results, fault_events=fault_results, peak_results=peaks)
    fc_per, fc_system = forecast.run_forecast_pipeline(
        settings=settings, preprocess_results=preprocess_results)

    frames = {n: r.feature_frame for n, r in preprocess_results.items()
              if getattr(r, "feature_frame", None) is not None}

    actual_kwh = compute_actual_energy_from_history(settings=settings)
    system_bill = predict_bill_from_record(actual_kwh, fc_system, horizon="forecast_24h",
                                           rate=rate, billing_period="30d")
    bills = {n: predict_bill_from_record(actual_kwh, fc, horizon="forecast_24h",
                                         rate=rate, billing_period="30d")
             for n, fc in fc_per.items()}

    anomalies: Dict[int, List[dict]] = {}
    for n, ar in anomaly_results.items():
        rf = getattr(ar, "result_frame", None)
        lst: List[dict] = []
        if rf is not None and hasattr(rf, "columns"):
            cols = list(rf.columns)
            acol = next((c for c in cols if "anomal" in c.lower()), None)
            if acol and "timestamp" in cols:
                sub = rf[rf[acol].astype(bool)] if rf[acol].dtype == bool else rf[rf[acol].notna()]
                for _, row in sub.iterrows():
                    lst.append({"pzem_number": n, "timestamp": int(row["timestamp"]),
                                "type": "anomaly", "severity": "ANOMALY"})
        anomalies[n] = lst

    faults: Dict[int, List[dict]] = {}
    for n, evs in fault_results.items():
        faults[n] = [vars(e) if not isinstance(e, dict) else e for e in (evs or [])]

    meters = {n: energy_saving.MeterEvidence(
        n, feature_frame=frames.get(n), peak_result=peaks.get(n),
        risk_result=risks.get(n), forecast_result=fc_per.get(n))
        for n in preprocess_results}
    recs = energy_saving.generate_recommendations(meters, rate=rate)

    # Stage 16: AI status for report
    ai_status: Dict[int, Any] = {}
    try:
        from ai.ai_status import compute_all_ai_status
        ai_status = compute_all_ai_status(
            anomaly_results=anomaly_results or {},
            fault_results_map=faults or {},
            last_pipeline_run=None,
        )
    except Exception:
        pass

    return ReportInput(
        pzem_count=settings.pzem_count, frames=frames, anomalies=anomalies,
        faults=faults, peaks=peaks, risks=risks, forecasts=fc_per,
        bills=bills, system_bill=system_bill, recommendations=recs, rate=rate,
        ai_status=ai_status,
    )


def demo_input(pzem_count: int = 9, seed: int = 42) -> ReportInput:
    rng = np.random.RandomState(seed)
    frames: Dict[int, pd.DataFrame] = {}
    anomalies: Dict[int, List[dict]] = {}
    faults: Dict[int, List[dict]] = {}
    peaks: Dict[int, Any] = {}
    risks: Dict[int, Any] = {}
    forecasts: Dict[int, Any] = {}
    bills: Dict[int, dict] = {}
    base = datetime.datetime.now(UTC) - datetime.timedelta(days=9)

    for n in range(1, pzem_count + 1):
        # ~10 days of 5-min data
        n_pts = 10 * 288
        ts = np.arange(int(base.timestamp()), int(base.timestamp()) + n_pts * 300, 300)
        tod = (ts % 86400) // 60
        power = 200.0 + 150.0 * np.sin(tod / 240.0) + rng.normal(0, 20, n_pts)
        if n % 4 == 0:
            power *= 1.8  # higher consumers
        power = np.clip(power, 5, None)
        frames[n] = pd.DataFrame({
            "timestamp": ts,
            "voltage": 230.0 + rng.normal(0, 2, n_pts),
            "current": power / 230.0,
            "power": power,
            "energy": np.cumsum(power) * 300 / 3600.0,
            "frequency": 50.0,
            "pf": 0.95,
        })
        anomalies[n] = [
            {"pzem_number": n, "timestamp": int(ts[1000 + n * 50]),
             "anomaly_severity_provisional": "ANOMALY", "type": "power_spike"}
        ] if n % 3 == 0 else []
        faults[n] = [
            {"pzem_number": n, "timestamp": int(ts[2000 + n * 30]),
             "fault_type": "OVER_VOLTAGE", "severity": "WARNING"}
        ] if n % 5 == 0 else []
        peaks[n] = SimpleNamespace(status="PEAK_FOUND", peak_power_w=float(power.max()),
                                   peak_timestamp=int(ts[int(power.argmax())]),
                                   baseline_power_w=float(np.median(power)),
                                   peak_above_baseline_w=float(power.max() - np.median(power)))
        risks[n] = SimpleNamespace(status="RISK_ASSESSED", risk_level=("HIGH" if n % 4 == 0 else "WATCH"),
                                   risk_score=70 if n % 4 == 0 else 30)
        fc_power = (power[:288] * (1.0 + 0.05 * rng.randn(288))).clip(5).tolist()
        forecasts[n] = SimpleNamespace(
            forecast_24h={"status": "FORECAST", "start_ts": int(ts[0]),
                          "count": 288, "confidence": "high", "forecast_power_w": fc_power},
            forecast_7d={"status": "NO_FORECAST", "reason": "n/a"})
        bills[n] = {"status": "OK", "actual_energy_kwh": round(float(power.sum()) / 1000 * 300 / 3600, 3),
                    "forecast_energy_kwh": 50.0, "estimated_total_energy_kwh": 120.0,
                    "estimated_bill": 840.0, "reason": None}

    from .energy_saving import generate_recommendations, MeterEvidence
    meters = {
        n: MeterEvidence(n, feature_frame=frames[n], peak_result=peaks[n],
                         risk_result=risks[n], forecast_result=forecasts[n])
        for n in range(1, pzem_count + 1)
    }
    recs = generate_recommendations(meters, rate=7.0)
    return ReportInput(
        pzem_count=pzem_count, frames=frames, anomalies=anomalies, faults=faults,
        peaks=peaks, risks=risks, forecasts=forecasts, bills=bills,
        system_bill={"status": "OK", "actual_energy_kwh": 900.0, "forecast_energy_kwh": 450.0,
                     "estimated_total_energy_kwh": 1350.0, "estimated_bill": 9450.0},
        recommendations=recs, rate=7.0,
    )
