"""
ai/data_loader.py
------------------
STAGE 1: Firebase historical-data loader.

Reads the EXISTING Firebase paths written by SmartEnergyMonitor.ino —
    meters/pzem_1 .. meters/pzem_9          (live)
    history/pzem_1/<unix-seconds> .. pzem_9 (5-minute slots, ~60 days)

and turns them into clean pandas DataFrames for the ML stages that follow.

This module NEVER writes to meters/ or history/ — it is read-only against
those two paths. (Stage 6 will introduce a separate ai/ path for AI
results; this file doesn't touch that either.)

Design goals for this stage specifically:
  1. Don't re-download 60 days of history from Firebase on every run —
     cache per-meter data on disk (Parquet) and only fetch what's new
     since the last cached timestamp (incremental fetch).
  2. Never invent data. Malformed/missing rows are dropped and reported,
     not filled with fabricated values.
  3. Degrade gracefully: if Firebase is unreachable, fall back to
     whatever is already cached and say so explicitly, rather than
     crashing the whole analysis pipeline.
  4. If less than the configured retention window is available (e.g. the
     system has only been running 12 days), report the ACTUAL available
     duration — never claim 60 days of coverage that doesn't exist.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import Settings, get_settings

logger = logging.getLogger("ai.data_loader")

# The exact fields the firmware writes per reading (readingJson() in the
# .ino). "lastSeen" only exists on meters/, not history/, so it's handled
# separately rather than listed here.
READING_FIELDS = ["voltage", "current", "power", "energy", "frequency", "pf"]


class FirebaseUnavailableError(RuntimeError):
    """Raised when the Firebase RTDB can't be reached at all (network,
    auth, or the database itself down) — distinct from "reachable but
    this meter simply has no data", which is not an error."""


# ---------------------------------------------------------------------------
# Firebase Admin SDK bootstrap
# ---------------------------------------------------------------------------

_firebase_app = None


def _init_firebase(settings: Settings):
    """Initializes the Firebase Admin SDK exactly once per process, using a
    SERVER-SIDE service-account key — never the ESP32's device
    email/password from config.h, and never exposed to any frontend code.
    """
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

    cred_path = Path(settings.firebase_service_account_path)
    if not cred_path.exists():
        raise ConfigFileMissingError(
            f"Service account file not found at {cred_path}. "
            f"Download it from Firebase console > Project settings > "
            f"Service accounts, and point FIREBASE_SERVICE_ACCOUNT_PATH at it. "
            f"Never commit this file."
        )

    cred = credentials.Certificate(str(cred_path))
    _firebase_app = firebase_admin.initialize_app(
        cred, {"databaseURL": settings.firebase_database_url}
    )
    logger.info("Firebase Admin SDK initialized against %s", settings.firebase_database_url)
    return _firebase_app


class ConfigFileMissingError(RuntimeError):
    pass


def _db_ref(path: str):
    from firebase_admin import db

    _init_firebase(get_settings())
    return db.reference(path)


# ---------------------------------------------------------------------------
# Historical reads (history/pzem_N/<unix-seconds>)
# ---------------------------------------------------------------------------

@dataclass
class HistoryLoadResult:
    """What callers actually need: the cleaned data PLUS honest metadata
    about how much of it there really is and where it came from, so
    downstream ML/report code never has to guess."""

    pzem_number: int
    frame: pd.DataFrame           # columns: timestamp (unix s), voltage, current, power, energy, frequency, pf
    available_days: float         # actual span of data available, not the configured target
    requested_days: int           # what was asked for (e.g. 60)
    served_from_cache_only: bool  # True if Firebase was unreachable and this is stale cache
    dropped_rows: int             # malformed/unparseable rows that were discarded
    duplicate_keys_collapsed: int


def _cache_path(settings: Settings, pzem_number: int) -> Path:
    return settings.cache_dir / f"history_pzem_{pzem_number}.parquet"


def _parse_history_snapshot(raw: dict, pzem_number: int) -> tuple[pd.DataFrame, int, int]:
    """Turns Firebase's {"<unix-seconds-as-string>": {...}, ...} object into
    a validated DataFrame. Returns (frame, dropped_rows, duplicate_keys).

    Mirrors script.js's normalizePowerWatts() fallback: a row may be either
    the current object format or (for old/legacy rows) a bare number, per
    the project's own history — but a bare number has no voltage/current/
    energy/etc., so it's still dropped here since this loader needs the
    full reading, not just power. That's a deliberate difference from the
    dashboard's live-card display, which only needs power.
    """
    if not raw:
        return pd.DataFrame(columns=["timestamp", *READING_FIELDS]), 0, 0

    rows = []
    dropped = 0
    seen_timestamps: set[int] = set()
    duplicates = 0

    for key, value in raw.items():
        # Key must be a valid Unix-seconds integer (matches how the firmware
        # writes history/pzem_N/<slot> and how cleanupHistory() parses keys
        # numerically rather than trusting lexicographic order).
        try:
            ts = int(key)
        except (TypeError, ValueError):
            dropped += 1
            continue

        if not isinstance(value, dict):
            # Legacy bare-number row, or otherwise malformed — can't recover
            # the other five fields from it.
            dropped += 1
            continue

        row = {"timestamp": ts}
        valid = True
        for field_name in READING_FIELDS:
            v = value.get(field_name)
            try:
                v = float(v)
            except (TypeError, ValueError):
                valid = False
                break
            row[field_name] = v

        if not valid:
            dropped += 1
            continue

        if ts in seen_timestamps:
            duplicates += 1
            # Keep the later-seen value (dict iteration order from Firebase
            # is insertion order for numeric-ish keys; either way this is a
            # deliberate "last one wins" rather than silently keeping
            # whichever happened to parse first).
            rows = [r for r in rows if r["timestamp"] != ts]
        seen_timestamps.add(ts)
        rows.append(row)

    frame = pd.DataFrame(rows, columns=["timestamp", *READING_FIELDS])
    if not frame.empty:
        frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame, dropped, duplicates


def _load_cache(settings: Settings, pzem_number: int) -> pd.DataFrame:
    path = _cache_path(settings, pzem_number)
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", *READING_FIELDS])
    try:
        return pd.read_parquet(path)
    except Exception:
        logger.exception("Cache file for PZEM %d is corrupt; ignoring it", pzem_number)
        return pd.DataFrame(columns=["timestamp", *READING_FIELDS])


def _save_cache(settings: Settings, pzem_number: int, frame: pd.DataFrame) -> None:
    path = _cache_path(settings, pzem_number)
    tmp_path = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)  # atomic on POSIX — avoids a half-written cache file


def fetch_meter_history(
    pzem_number: int,
    settings: Optional[Settings] = None,
    force_full_refresh: bool = False,
) -> HistoryLoadResult:
    """Loads up to `settings.history_retention_days` of history for one
    PZEM, incrementally: only the slots newer than what's already cached
    are actually fetched from Firebase.

    If Firebase is unreachable, falls back to the on-disk cache (if any)
    and marks the result as served_from_cache_only=True rather than
    raising — callers decide whether that's acceptable for their use case.
    """
    settings = settings or get_settings()
    _validate_pzem_number(pzem_number, settings)

    cached = pd.DataFrame(columns=["timestamp", *READING_FIELDS]) if force_full_refresh else _load_cache(settings, pzem_number)

    now = int(time.time())
    retention_cutoff = now - settings.history_retention_days * 86400

    # Only ask Firebase for slots after the newest one we already have
    # cached (or the full retention window, on first run / forced refresh).
    fetch_start = int(cached["timestamp"].max()) + 1 if not cached.empty else retention_cutoff

    served_from_cache_only = False
    new_frame = pd.DataFrame(columns=["timestamp", *READING_FIELDS])
    dropped = 0
    duplicates = 0

    if fetch_start <= now:
        try:
            ref = _db_ref(f"history/pzem_{pzem_number}")
            raw = (
                ref.order_by_key()
                .start_at(str(fetch_start))
                .get()
            )
            new_frame, dropped, duplicates = _parse_history_snapshot(raw, pzem_number)
        except FirebaseUnavailableError:
            logger.warning(
                "Firebase unreachable while loading history for PZEM %d; "
                "falling back to cached data only.",
                pzem_number,
            )
            served_from_cache_only = True
        except Exception as exc:
            logger.warning(
                "Firebase read failed for PZEM %d (%s); falling back to cache.",
                pzem_number,
                exc,
            )
            served_from_cache_only = True

    combined = pd.concat([cached, new_frame], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(subset="timestamp", keep="last")
        combined = combined[combined["timestamp"] >= retention_cutoff]
        combined = combined.sort_values("timestamp").reset_index(drop=True)

    # Persist the trimmed, deduped result as the new cache baseline —
    # keeps the cache file bounded to the retention window instead of
    # growing forever.
    if not combined.empty:
        _save_cache(settings, pzem_number, combined)

    if combined.empty:
        available_days = 0.0
    else:
        span_seconds = combined["timestamp"].iloc[-1] - combined["timestamp"].iloc[0]
        available_days = round(span_seconds / 86400, 2)

    return HistoryLoadResult(
        pzem_number=pzem_number,
        frame=combined,
        available_days=available_days,
        requested_days=settings.history_retention_days,
        served_from_cache_only=served_from_cache_only,
        dropped_rows=dropped,
        duplicate_keys_collapsed=duplicates,
    )


def fetch_all_history(
    settings: Optional[Settings] = None,
    force_full_refresh: bool = False,
) -> dict[int, HistoryLoadResult]:
    """Loads history for every configured PZEM (1..PZEM_COUNT). One
    Firebase-unreachable meter doesn't block the others — each is
    independent, so a partial outage still returns everything it could."""
    settings = settings or get_settings()
    results: dict[int, HistoryLoadResult] = {}
    for n in range(1, settings.pzem_count + 1):
        results[n] = fetch_meter_history(n, settings=settings, force_full_refresh=force_full_refresh)
    return results


def _validate_pzem_number(pzem_number: int, settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    if not (1 <= pzem_number <= settings.pzem_count):
        raise ValueError(
            f"pzem_number must be between 1 and {settings.pzem_count}, got {pzem_number}"
        )
