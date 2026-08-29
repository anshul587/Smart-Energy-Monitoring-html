"""
ai/bob_tools.py
---------------
Stage 16 (enhanced): the tool registry that makes Ask BOB a data-aware agent.

Each tool is a small, *validated* wrapper around the Stage 15 read layer
(ai.api_store) — the exact same data path the Stage 15 REST API uses to serve
its endpoints. This keeps a single canonical Firebase access seam and means the
LLM/agent never touches Firebase, SQL, the filesystem (except the local monthly
PDFs, which the API also serves), or arbitrary URLs.

The agent can only invoke the registered tools below. Every parameter is
validated before any read. Unknown tools or bad parameters raise ToolError and
are never executed.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Optional

from . import api_store
from .config import get_settings

logger = logging.getLogger("ai.bob_tools")

MAX_LIMIT = 500
DEFAULT_LIMIT = 15  # bounded; tools never pull unlimited history

_ALLOWED_HORIZONS = {"24h", "7d", "both"}


class ToolError(Exception):
    """Raised when a tool call is invalid or its backing data is unavailable."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


# ---------------------------------------------------------------------------
# Validation (every tool parameter is checked before use)
# ---------------------------------------------------------------------------

def _pzem_count() -> int:
    try:
        return get_settings().pzem_count
    except Exception:  # config missing -> safe fallback, never crash the agent
        return 9


def _check_pzem(n: Any) -> int:
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise ToolError("invalid_pzem", "pzem_number must be an integer")
    if not (1 <= n <= _pzem_count()):
        raise ToolError("invalid_pzem", f"pzem_number must be between 1 and {_pzem_count()}")
    return n


def _check_limit(limit: Any) -> int:
    if limit is None:
        return DEFAULT_LIMIT
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise ToolError("invalid_limit", "limit must be an integer")
    if limit < 0:
        raise ToolError("invalid_limit", "limit must be >= 0")
    if limit > MAX_LIMIT:
        raise ToolError("limit_too_large", f"limit exceeds maximum of {MAX_LIMIT}")
    return limit


def _check_ts(ts: Any) -> Optional[int]:
    if ts is None:
        return None
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        raise ToolError("invalid_timestamp", "timestamp must be an integer (unix seconds)")
    if ts < 0:
        raise ToolError("invalid_timestamp", "timestamp must be >= 0")
    return ts


def _check_horizon(h: Any) -> str:
    if h is None:
        return "both"
    h = str(h).lower()
    if h not in _ALLOWED_HORIZONS:
        raise ToolError("invalid_horizon", f"horizon must be one of {sorted(_ALLOWED_HORIZONS)}")
    return h


def _check_str(name: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    if not re.fullmatch(r"[A-Za-z0-9 _\-]+", s):
        raise ToolError(f"invalid_{name}", f"{name} contains disallowed characters")
    return s


# ---------------------------------------------------------------------------
# Shared normalisation (mirrors the Stage 15 REST API shapes)
# ---------------------------------------------------------------------------

def _meter_entry(n: int, m: Any) -> Optional[dict]:
    if not isinstance(m, dict):
        return None
    online, age = api_store.meter_online(m)
    return {
        "pzem_number": n,
        "online": online,
        "voltage": m.get("voltage"),
        "current": m.get("current"),
        "power": m.get("power") if online else None,
        "energy": m.get("energy"),
        "power_factor": m.get("pf"),
        "frequency": m.get("frequency") if online else None,
        "last_seen": m.get("lastSeen", m.get("timestamp")),
        "age_ms": age,
    }


def _filter(records, pzem, start, end, severity, severity_key, risk, priority) -> list:
    out = []
    for r in records or []:
        if not isinstance(r, dict):
            continue
        ts = api_store.record_timestamp(r)
        pz = api_store.record_pzem(r)
        if pzem is not None:
            if pz is None:
                recs = r.get("recommendations", []) or []
                if not any(int(x.get("pzem_number", 0)) == pzem for x in recs if isinstance(x, dict)):
                    continue
            elif pz != pzem:
                continue
        if start is not None and ts is not None and ts < start:
            continue
        if end is not None and ts is not None and ts > end:
            continue
        if severity is not None and str(r.get(severity_key, "")).lower() != severity.lower():
            continue
        if risk is not None and str(r.get("risk_level", "")).lower() != risk.lower():
            continue
        if priority is not None:
            recs = r.get("recommendations", []) or []
            if not any(str(x.get("priority", "")).lower() == priority.lower() for x in recs if isinstance(x, dict)):
                continue
        out.append(r)
    out.sort(key=lambda r: api_store.record_timestamp(r) or 0, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Tools (each calls only the Stage 15 read layer)
# ---------------------------------------------------------------------------

def get_system_summary() -> dict:
    return dict(api_store.build_summary() or {})


def get_meters() -> list:
    raw = api_store.read_all_meters() or {}
    out = []
    for n in sorted(raw.keys()):
        e = _meter_entry(n, raw[n])
        if e:
            out.append(e)
    return out


def get_meter(pzem_number: int) -> dict:
    n = _check_pzem(pzem_number)
    e = _meter_entry(n, api_store.read_meter(n))
    if e is None:
        raise ToolError("not_found", f"No data for PZEM {n}")
    return e


def get_anomalies(pzem_number=None, start=None, end=None, limit=None, severity=None) -> list:
    pz = None if pzem_number is None else _check_pzem(pzem_number)
    lim = _check_limit(limit)
    recs = _filter(api_store.read_anomalies(), pz, _check_ts(start), _check_ts(end),
                   _check_str("severity", severity), "anomaly_severity_provisional", None, None)
    return recs[:lim]


def get_faults(pzem_number=None, start=None, end=None, limit=None, severity=None) -> list:
    pz = None if pzem_number is None else _check_pzem(pzem_number)
    lim = _check_limit(limit)
    recs = _filter(api_store.read_faults(), pz, _check_ts(start), _check_ts(end),
                   _check_str("severity", severity), "severity", None, None)
    return recs[:lim]


def get_peaks(pzem_number=None, start=None, end=None, limit=None) -> list:
    pz = None if pzem_number is None else _check_pzem(pzem_number)
    lim = _check_limit(limit)
    recs = _filter(api_store.read_peaks(), pz, _check_ts(start), _check_ts(end), None, "severity", None, None)
    return recs[:lim]


def get_maintenance(pzem_number=None, start=None, end=None, limit=None, risk=None) -> list:
    pz = None if pzem_number is None else _check_pzem(pzem_number)
    lim = _check_limit(limit)
    recs = _filter(api_store.read_maintenance(), pz, _check_ts(start), _check_ts(end), None,
                   "risk_level", _check_str("risk", risk), None)
    return recs[:lim]


def get_forecast(pzem_number=None, horizon=None, start=None, end=None, limit=None) -> list:
    pz = None if pzem_number is None else _check_pzem(pzem_number)
    h = _check_horizon(horizon)
    lim = _check_limit(limit)
    recs = _filter(api_store.read_forecast(), pz, _check_ts(start), _check_ts(end), None, "severity", None, None)
    if h in ("24h", "7d"):
        strip = "forecast_7d" if h == "24h" else "forecast_24h"
        for r in recs:
            r.pop(strip, None)
    return recs[:lim]


def get_bill_prediction(limit=None) -> list:
    lim = _check_limit(limit)
    recs = api_store.read_bill_prediction() or []
    recs.sort(key=lambda r: api_store.record_timestamp(r) or 0, reverse=True)
    return recs[:lim]


def get_energy_saving(pzem_number=None, priority=None, start=None, end=None, limit=None) -> list:
    pz = None if pzem_number is None else _check_pzem(pzem_number)
    lim = _check_limit(limit)
    recs = _filter(api_store.read_energy_saving(), pz, _check_ts(start), _check_ts(end), None,
                   "severity", None, _check_str("priority", priority))
    return recs[:lim]


# --- Monthly reports: local disk only (same source the API serves) -----------

_MONTHLY_FILE_RE = re.compile(r"^report-\d{4}-\d{2}\.pdf$")


def _monthly_dir() -> str:
    root = os.environ.get("REPORTS_DIR")
    if root:
        return os.path.join(root, "monthly")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.normpath(os.path.join(here, "Dashboard Smart-Monitoring-System", "reports", "monthly"))


def get_monthly_reports() -> list:
    d = _monthly_dir()
    try:
        entries = os.listdir(d) if os.path.isdir(d) else []
    except OSError:
        entries = []
    files = []
    for fn in entries:
        if fn != "latest.pdf" and not _MONTHLY_FILE_RE.match(fn):
            continue
        path = os.path.join(d, fn)
        m = _MONTHLY_FILE_RE.match(fn)
        year = month = None
        if m:
            year, month = (int(x) for x in m.group(0)[len("report-"):-len(".pdf")].split("-"))
        try:
            size = os.path.getsize(path)
        except OSError:
            size = None
        files.append({
            "filename": fn, "year": year, "month": month, "size_bytes": size,
            "url": f"/api/v1/reports/monthly/{fn}",
        })
    files.sort(key=lambda f: (f["year"] or 0, f["month"] or 0), reverse=True)
    return files


# ---------------------------------------------------------------------------
# Registry + dispatch (the ONLY way to invoke a tool)
# ---------------------------------------------------------------------------

_TOOL_FUNCS: dict[str, Callable] = {
    "get_system_summary": get_system_summary,
    "get_meters": get_meters,
    "get_meter": get_meter,
    "get_anomalies": get_anomalies,
    "get_faults": get_faults,
    "get_peaks": get_peaks,
    "get_maintenance": get_maintenance,
    "get_forecast": get_forecast,
    "get_bill_prediction": get_bill_prediction,
    "get_energy_saving": get_energy_saving,
    "get_monthly_reports": get_monthly_reports,
}

# Accepted parameters per tool (used for safe auto-binding + param filtering).
_TOOL_PARAMS: dict[str, tuple] = {
    "get_system_summary": (),
    "get_meters": (),
    "get_meter": ("pzem_number",),
    "get_anomalies": ("pzem_number", "start", "end", "limit", "severity"),
    "get_faults": ("pzem_number", "start", "end", "limit", "severity"),
    "get_peaks": ("pzem_number", "start", "end", "limit"),
    "get_maintenance": ("pzem_number", "start", "end", "limit", "risk"),
    "get_forecast": ("pzem_number", "horizon", "start", "end", "limit"),
    "get_bill_prediction": ("limit",),
    "get_energy_saving": ("pzem_number", "priority", "start", "end", "limit"),
    "get_monthly_reports": (),
}


def available_tools() -> list[str]:
    return sorted(_TOOL_FUNCS.keys())


def run_tool(name: str, **params: Any) -> Any:
    """Execute a registered tool with validated parameters only.

    Unknown tools or unknown/disallowed parameters are rejected — the agent
    (including any LLM) can never reach an unregistered or unsafe call.
    """
    fn = _TOOL_FUNCS.get(name)
    if fn is None:
        raise ToolError("unknown_tool", f"Tool '{name}' is not registered")
    sig = _TOOL_PARAMS[name]
    kwargs = {k: v for k, v in params.items() if v is not None and k in sig}
    return fn(**kwargs)


class ToolContext:
    """Per-request cache so identical tool calls aren't repeated within one ask."""

    def __init__(self) -> None:
        self._cache: dict = {}

    def call(self, name: str, **params: Any) -> dict:
        key = (name, tuple(sorted((k, str(v)) for k, v in params.items() if v is not None)))
        if key in self._cache:
            return self._cache[key]
        result = {
            "tool": name,
            "params": {k: v for k, v in params.items() if v is not None},
            "ok": True,
            "data": None,
            "error": None,
        }
        try:
            result["data"] = run_tool(name, **params)
        except ToolError as exc:
            result["ok"] = False
            result["error"] = {"code": exc.code, "message": exc.message}
        except Exception as exc:  # noqa: BLE001 - never let a tool crash the agent
            logger.warning("Tool %s failed: %s", name, exc)
            result["ok"] = False
            result["error"] = {"code": "tool_error", "message": str(exc)}
        self._cache[key] = result
        return result
