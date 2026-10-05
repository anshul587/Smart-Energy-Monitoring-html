"""PZEM load identity mapping - single source of truth.

Source of truth is Firebase RTDB ``config/pzem_mapping/<pzemKey>``
(``{"loadName": ..., "location": ...}``), which the dashboard edit modal
writes. ``DEFAULT_PZEM_LOAD_MAPPING`` below is only the fallback used when
Firebase is unavailable or the path has no entry yet, so display code never
breaks and never invents a name.
"""

from __future__ import annotations

from typing import Any, Optional

FIREBASE_MAPPING_PATH = "config/pzem_mapping"

DEFAULT_PZEM_LOAD_MAPPING: dict[str, dict[str, str]] = {
    "pzem_1": {"load_name": "PZEM-1", "location": "Unassigned"},
    "pzem_2": {"load_name": "PZEM-2", "location": "Unassigned"},
    "pzem_3": {"load_name": "PZEM-3", "location": "Unassigned"},
    "pzem_4": {"load_name": "PZEM-4", "location": "Unassigned"},
    "pzem_5": {"load_name": "PZEM-5", "location": "Unassigned"},
    "pzem_6": {"load_name": "PZEM-6", "location": "Unassigned"},
    "pzem_7": {"load_name": "PZEM-7", "location": "Unassigned"},
    "pzem_8": {"load_name": "PZEM-8", "location": "Unassigned"},
    "pzem_9": {"load_name": "PZEM-9", "location": "Unassigned"},
}

_REMOTE: dict[str, dict[str, str]] = {}
_REMOTE_LOADED = False


def _load_remote() -> dict[str, dict[str, str]]:
    """Read config/pzem_mapping once. Any failure -> empty (defaults apply)."""
    global _REMOTE, _REMOTE_LOADED
    if _REMOTE_LOADED:
        return _REMOTE
    _REMOTE_LOADED = True
    _REMOTE = {}
    try:
        from .api_store import db_get

        raw: Optional[Any] = db_get(FIREBASE_MAPPING_PATH)
    except Exception:
        return _REMOTE
    if not isinstance(raw, dict):
        return _REMOTE
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        name = str(value.get("loadName") or value.get("load_name") or "").strip()
        loc = str(value.get("location") or "").strip()
        _REMOTE[str(key)] = {"load_name": name, "location": loc}
    return _REMOTE


def refresh_pzem_load_mapping() -> None:
    """Force the next lookup to re-read Firebase (call after a save)."""
    global _REMOTE_LOADED
    _REMOTE_LOADED = False
    _load_remote()


def get_pzem_load_mapping() -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {
        k: dict(v) for k, v in DEFAULT_PZEM_LOAD_MAPPING.items()
    }
    for key, value in _load_remote().items():
        merged.setdefault(key, {"load_name": "", "location": ""}).update(
            {k: v for k, v in value.items() if v}
        )
    return merged


def get_load_name(pzem_key: str, default: str | None = None) -> str:
    if not pzem_key:
        return default or "Unknown"
    if default is None:
        default = pzem_key.upper().replace("_", "-")
    try:
        entry = _load_remote().get(pzem_key) or {}
        name = entry.get("load_name")
        if not name:
            name = DEFAULT_PZEM_LOAD_MAPPING.get(pzem_key, {}).get("load_name")
        if name and name.strip():
            return name.strip()
    except Exception:
        pass
    return default


def get_load_location(pzem_key: str, default: str = "Unassigned") -> str:
    if not pzem_key:
        return default
    try:
        entry = _load_remote().get(pzem_key) or {}
        loc = entry.get("location")
        if not loc:
            loc = DEFAULT_PZEM_LOAD_MAPPING.get(pzem_key, {}).get("location")
        if loc and loc.strip():
            return loc.strip()
    except Exception:
        pass
    return default


def get_pzem_display_id(pzem_key: str) -> str:
    num = pzem_key.replace("pzem_", "").replace("PZEM_", "")
    return f"PZEM-{num}"


def meter_label(pzem_number) -> str:
    """Display name only: 'Fan 1'. For charts/tables."""
    key = pzem_number if isinstance(pzem_number, str) and pzem_number.startswith("pzem_") \
        else f"pzem_{pzem_number}"
    return get_load_name(key) or get_pzem_display_id(key)


def meter_label_with_id(pzem_number) -> str:
    """'Fan 1 (PZEM-1)' - keeps the immutable ID visible for traceability."""
    key = pzem_number if isinstance(pzem_number, str) and pzem_number.startswith("pzem_") \
        else f"pzem_{pzem_number}"
    name = get_load_name(key)
    display_id = get_pzem_display_id(key)
    return display_id if not name or name == display_id else f"{name} ({display_id})"
