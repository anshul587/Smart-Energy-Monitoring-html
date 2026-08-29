"""
ai/api_store.py
---------------
Stage 15: READ-ONLY data access for the REST API.

This module only READS existing data:
  - live meter readings:        meters/pzem_N
  - persisted AI results:       /ai/anomalies, /ai/faults, /ai/peaks,
                                /ai/maintenance, /ai/forecast,
                                /ai/bill_prediction, /ai/energy_saving

It never writes and never runs any AI model. All Firebase access goes through
a single injectable seam (`_db_get`) so tests can run without credentials, and
through a small TTL cache so the API does not hammer Firebase on every request.

Reuses ai.data_loader._db_ref for the real Firebase read path (same SDK /
credentials / database the rest of the backend already uses).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from .config import get_settings

logger = logging.getLogger("ai.api_store")

# ---------------------------------------------------------------------------
# Injectable read seam — tests monkeypatch ai.api_store._db_get
# ---------------------------------------------------------------------------

_db_get: Optional[Callable[[str], Any]] = None

_CACHE_TTL_SECONDS = 5
_cache: dict[str, tuple[float, Any]] = {}


def _default_db_get(path: str) -> Any:
    from .data_loader import _db_ref

    return _db_ref(path).get()


def db_get(path: str) -> Any:
    fn = _db_get or _default_db_get
    return fn(path)


def set_db_get(fn: Optional[Callable[[str], Any]]) -> None:
    """Override the low-level Firebase reader (used by tests)."""
    global _db_get
    _db_get = fn


def clear_cache() -> None:
    _cache.clear()


def _cached(key: str, getter: Callable[[], Any], ttl: int = _CACHE_TTL_SECONDS) -> Any:
    now = time.time()
    entry = _cache.get(key)
    if entry is not None and (now - entry[0]) < ttl:
        return entry[1]
    value = getter()
    _cache[key] = (now, value)
    return value


# ---------------------------------------------------------------------------
# Freshness (mirrors Dashboard script.js FRESHNESS_TIMEOUT_MS = 30s)
# ---------------------------------------------------------------------------

ONLINE_THRESHOLD_S = 30


def meter_online(meter: Any) -> tuple[bool, Optional[int]]:
    """Returns (online, age_ms). Uses lastSeen ?? timestamp (ms), matching the
    dashboard's live/offline rule. Never fabricates a value."""
    if not isinstance(meter, dict):
        return False, None
    raw = meter.get("lastSeen", meter.get("timestamp"))
    if raw is None:
        return False, None
    try:
        last_ms = int(raw)
    except (TypeError, ValueError):
        return False, None
    age_ms = int(time.time() * 1000) - last_ms
    if age_ms < 0:  # clock skew / future timestamp -> treat as fresh, not offline
        age_ms = 0
    return age_ms <= ONLINE_THRESHOLD_S * 1000, age_ms


# ---------------------------------------------------------------------------
# Generic readers
# ---------------------------------------------------------------------------

def read_meter(pzem_number: int) -> Optional[dict]:
    return db_get(f"meters/pzem_{pzem_number}")


def read_all_meters() -> dict[int, Optional[dict]]:
    settings = get_settings()
    out: dict[int, Optional[dict]] = {}
    for n in range(1, settings.pzem_count + 1):
        out[n] = db_get(f"meters/pzem_{n}")
    return out


def _read_pzem_collection(base: str, include_system: bool) -> list[dict]:
    settings = get_settings()
    records: list[dict] = []
    for n in range(1, settings.pzem_count + 1):
        raw = db_get(f"{base}/pzem_{n}") or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, dict):
                    rec = dict(v)
                    rec.setdefault("pzem_number", n)
                    rec["_key"] = k
                    records.append(rec)
    if include_system:
        raw = db_get(f"{base}/system") or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, dict):
                    rec = dict(v)
                    rec.setdefault("pzem_number", None)
                    rec["_key"] = k
                    records.append(rec)
    return records


def _read_namespace(ns: str) -> list[dict]:
    raw = db_get(ns) or {}
    out: list[dict] = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                rec = dict(v)
                rec["_key"] = k
                out.append(rec)
    return out


def read_anomalies() -> list[dict]:
    return _cached("ai/anomalies", lambda: _read_pzem_collection("ai/anomalies", False))


def read_faults() -> list[dict]:
    return _cached("ai/faults", lambda: _read_pzem_collection("ai/faults", False))


def read_peaks() -> list[dict]:
    return _cached("ai/peaks", lambda: _read_pzem_collection("ai/peaks", True))


def read_maintenance() -> list[dict]:
    return _cached("ai/maintenance", lambda: _read_pzem_collection("ai/maintenance", True))


def read_forecast() -> list[dict]:
    return _cached("ai/forecast", lambda: _read_pzem_collection("ai/forecast", True))


def read_bill_prediction() -> list[dict]:
    return _cached("ai/bill_prediction", lambda: _read_namespace("ai/bill_prediction"))


def read_energy_saving() -> list[dict]:
    return _cached("ai/energy_saving", lambda: _read_namespace("ai/energy_saving"))


# ---------------------------------------------------------------------------
# Normalisation helpers (used by filters + summary)
# ---------------------------------------------------------------------------

def record_timestamp(rec: dict) -> Optional[int]:
    for key in ("timestamp", "anchor_timestamp"):
        val = rec.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return None


def record_pzem(rec: dict) -> Optional[int]:
    pz = rec.get("pzem_number")
    return int(pz) if isinstance(pz, int) else None


def _latest(records: list[dict]) -> Optional[dict]:
    if not records:
        return None
    return max(records, key=lambda r: record_timestamp(r) or 0)


# ---------------------------------------------------------------------------
# System summary
# ---------------------------------------------------------------------------

def build_summary() -> dict:
    settings = get_settings()
    meters = read_all_meters()

    online = 0
    total_power = 0.0
    total_energy = 0.0
    voltages: list[float] = []
    frequencies: list[float] = []
    for n in range(1, settings.pzem_count + 1):
        m = meters.get(n)
        if not isinstance(m, dict):
            continue
        on, _ = meter_online(m)
        try:
            total_energy += float(m.get("energy", 0) or 0)
        except (TypeError, ValueError):
            pass
        if on:
            online += 1
            try:
                total_power += float(m.get("power", 0) or 0)
            except (TypeError, ValueError):
                pass
            try:
                voltages.append(float(m.get("voltage", 0) or 0))
            except (TypeError, ValueError):
                pass
            try:
                frequencies.append(float(m.get("frequency", 0) or 0))
            except (TypeError, ValueError):
                pass

    anomalies = read_anomalies()
    active_anomalies = sum(
        1 for a in anomalies
        if str(a.get("anomaly_label", "")) not in ("", "NORMAL", "NOT_SCORED")
        and a.get("model_status") == "READY"
    )

    faults = read_faults()

    maintenance = read_maintenance()
    m_sys = _latest([m for m in maintenance if m.get("pzem_number") is None])
    maintenance_summary = None
    if m_sys:
        maintenance_summary = {
            "timestamp": record_timestamp(m_sys),
            "high_risk_meters": len(m_sys.get("high_risk_meters", []) or []),
            "watch_meters": len(m_sys.get("watch_meters", []) or []),
            "normal_meters": len(m_sys.get("normal_meters", []) or []),
            "highest_risk_pzem": m_sys.get("highest_risk_pzem"),
            "highest_risk_score": m_sys.get("highest_risk_score"),
        }

    peaks = read_peaks()
    peak_sys = _latest([p for p in peaks if p.get("pzem_number") is None])
    latest_peak = None
    if peak_sys:
        latest_peak = {
            "timestamp": record_timestamp(peak_sys),
            "total_peak_power_w": peak_sys.get("total_peak_power_w"),
            "dominant_pzems": peak_sys.get("dominant_pzems"),
        }

    forecast = read_forecast()
    forecast_available = any(f.get("status") == "FORECAST" for f in forecast)

    bills = read_bill_prediction()
    latest_bill = None
    if bills:
        b = _latest(bills)
        latest_bill = {
            "anchor_timestamp": record_timestamp(b),
            "status": b.get("status"),
            "estimated_bill": b.get("estimated_bill"),
            "estimated_total_energy_kwh": b.get("estimated_total_energy_kwh"),
        }

    savings = read_energy_saving()
    latest_saving = None
    if savings:
        s = _latest(savings)
        latest_saving = {
            "timestamp": record_timestamp(s),
            "status": s.get("status"),
            "recommendation_count": s.get("recommendation_count"),
        }

    return {
        "system_status": "online" if online > 0 else "offline",
        "online_meter_count": online,
        "total_meter_count": settings.pzem_count,
        "total_power_w": round(total_power, 3),
        "total_energy_kwh": round(total_energy, 3),
        "average_voltage_v": round(sum(voltages) / len(voltages), 3) if voltages else None,
        "frequency_hz": round(sum(frequencies) / len(frequencies), 3) if frequencies else None,
        "active_anomaly_count": active_anomalies,
        "active_fault_count": len(faults),
        "maintenance_risk": maintenance_summary,
        "latest_peak": latest_peak,
        "forecast_available": forecast_available,
        "latest_bill_prediction": latest_bill,
        "energy_saving": latest_saving,
    }
