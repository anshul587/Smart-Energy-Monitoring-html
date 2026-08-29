"""
ai/config.py
------------
Centralized configuration for the AI backend, loaded entirely from
environment variables (see .env.example). No credentials are hardcoded
here or anywhere else in this package.

This mirrors config.h's role in the firmware: one place that defines the
constants everything else depends on, so later stages (anomaly detection,
forecasting, Claude integration) import FROM here rather than re-reading
os.environ themselves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Loads a local .env if present (developer machine / server). In real
# deployment (e.g. a container), these are normally injected directly as
# environment variables and .env may not exist at all — that's fine, this
# is a no-op if the file isn't there.
load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"Required environment variable {name} is not set. "
            f"Copy .env.example to .env and fill it in, or set it in your "
            f"deployment environment."
        )
    return value


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value < 0:
        raise ConfigError(f"{name} must be >= 0, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    # --- Firebase ---
    # Path to a Firebase service-account JSON key file (Admin SDK). This is
    # a SERVER-SIDE credential with full read/write access to the RTDB, and
    # is intentionally different from the device email/password in the
    # ESP32's config.h — the backend never reuses the device's credentials.
    firebase_service_account_path: str = field(
        default_factory=lambda: _require("FIREBASE_SERVICE_ACCOUNT_PATH")
    )
    # Same RTDB instance the ESP32 and dashboard already use
    # (FIREBASE_DATABASE_URL in config.h / firebaseConfig.databaseURL in
    # script.js). The AI backend reads history/ and meters/ from here — it
    # does not create a second database.
    firebase_database_url: str = field(
        default_factory=lambda: _require("FIREBASE_DATABASE_URL")
    )

    # --- PZEM fleet (matches config.h PZEM_COUNT) ---
    pzem_count: int = field(default_factory=lambda: _int("PZEM_COUNT", 9))

    # --- History retention/analysis window ---
    # Matches config.h HISTORY_RETENTION_SECONDS (60 days), expressed in
    # days for the Python side. This is the AI analysis window — NOT the
    # dashboard's 1d/7d/30d visualization window, which is a display-only
    # filter script.js applies on top of the same history/ data.
    history_retention_days: int = field(
        default_factory=lambda: _int("HISTORY_RETENTION_DAYS", 60)
    )

    # --- Local cache ---
    # Where per-meter history is cached on disk so we don't re-download 60
    # days of readings from Firebase on every run. See data_loader.py.
    cache_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("AI_CACHE_DIR", ".cache"))
    )

    # --- Stage 7: peak-load threshold ---
    # ANNOTATION ONLY — never gates detection or persistence, and NOT a
    # Stage 4 fault threshold (FAULT_HIGH_POWER_W in fault_diagnosis.py is
    # separate and untouched). Default 0.0 disables the annotation: peaks
    # are then reported purely as observed maxima. Set e.g. 3000 (watts)
    # to have each peak record flag exceeds_threshold / margin above it.
    peak_power_threshold_w: float = field(
        default_factory=lambda: _float("PEAK_POWER_THRESHOLD_W", 0.0)
    )

    # --- Anthropic (used starting Stage 16 — declared now so later stages
    # don't need another config pass) ---
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )

    def __post_init__(self) -> None:
        if self.pzem_count < 1:
            raise ConfigError("PZEM_COUNT must be >= 1")
        if self.history_retention_days < 1:
            raise ConfigError("HISTORY_RETENTION_DAYS must be >= 1")
        self.cache_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Lazily-constructed singleton so importing this module never fails
    just because env vars aren't set yet (e.g. during unit tests that patch
    things before calling get_settings())."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
