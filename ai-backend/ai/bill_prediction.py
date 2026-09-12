"""
ai/bill_prediction.py
----------------------
STAGE 10: AI Bill Prediction + Dashboard Integration (backend half).

Combines EXISTING data sources to estimate the upcoming electricity bill:

    * actual energy consumed so far in the billing period
      -> reused from the SAME 30-day cumulative-energy history the manual
      Bill Calculator (script.js) already reads (no second history pipeline;
      compute_actual_energy_from_history() goes through ai.data_loader,
      Stage 1's loader, exactly like everything else).
    * forecasted future energy
      -> taken from the Stage 9 power forecast (/ai/forecast, already
      produced by ai.forecast), converted from average power (W) over the
      5-minute slot to energy (kWh):  kWh = SUM(power_W) * (300 / 3_600_000).
    * an ESTIMATED total and PREDICTED bill under the EXISTING flat-rate
      model (the same rate the manual calculator uses).

Nothing here invents measurements. When the forecast is unavailable or
insufficient, the result is INSUFFICIENT with a reason and no fabricated
bill. All numeric outputs are guarded against NaN/Inf/negative energy/bill.

PERSISTENCE / IDEMPOTENCY
  /ai/bill_prediction/<anchor-timestamp>   one record per run

The child key IS the deterministic Stage 9 anchor timestamp, checked before
writing (get-then-set), so re-running identical input is a no-op. This module
only writes to /ai/bill_prediction and never touches /meters, /history,
/alerts, /ai/anomalies, /ai/faults, /ai/peaks, /ai/maintenance or /ai/forecast.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from .config import Settings, get_settings
from .preprocessing import HISTORY_SLOT_SECONDS

logger = logging.getLogger("ai.bill_prediction")

# Energy conversion: each forecast point is the AVERAGE power over one
# 5-minute (HISTORY_SLOT_SECONDS) slot. Energy for that slot in kWh =
# power_W / 1000 (kW) * (slot_s / 3600) (h). Factor collapses to
# slot_s / 3_600_000.
WATT_SECONDS_PER_KWH = 3_600_000.0
_SLOT_ENERGY_FACTOR = HISTORY_SLOT_SECONDS / WATT_SECONDS_PER_KWH

SOURCE_STAGE = "stage10/bill_prediction"


# ---------------------------------------------------------------------------
# Pure helpers (fully deterministic, no Firebase)
# ---------------------------------------------------------------------------

def forecast_energy_kwh(horizon_payload: Optional[dict]) -> float:
    """Converts a Stage 9 forecast horizon payload's power series (W) into
    forecasted energy (kWh) using the 5-minute slot.

    Safe: ignores non-FORECAST payloads, drops NaN/None/non-numeric values,
    clamps negative power to 0 (energy is non-negative). Never returns NaN,
    Inf or a negative number.
    """
    if not isinstance(horizon_payload, dict):
        return 0.0
    if horizon_payload.get("status") != "FORECAST":
        return 0.0
    powers = horizon_payload.get("forecast_power_w")
    if not powers:
        return 0.0

    total_w = 0.0
    for p in powers:
        try:
            v = float(p)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        if v < 0:
            v = 0.0
        total_w += v

    kwh = total_w * _SLOT_ENERGY_FACTOR
    if not math.isfinite(kwh) or kwh < 0:
        return 0.0
    return kwh


def predict_bill(
    actual_energy_kwh,
    forecast_horizon_payload: Optional[dict],
    rate: float = 0.0,
    billing_period: str = "30d",
    forecast_confidence: Optional[str] = None,
) -> dict:
    """Combines actual energy + Stage 9 forecast into a bill prediction.

    Returns a dict with status "OK" or "INSUFFICIENT". On INSUFFICIENT the
    energy/bill fields are None (never fabricated). On OK every numeric
    field is finite and non-negative.
    """
    # --- actual energy (must be finite & non-negative) ---
    try:
        actual = float(actual_energy_kwh)
    except (TypeError, ValueError):
        actual = float("nan")
    if not math.isfinite(actual) or actual < 0:
        return _insufficient("actual energy consumption unavailable or invalid")

    # --- forecast (must be a valid FORECAST payload) ---
    if not isinstance(forecast_horizon_payload, dict) \
            or forecast_horizon_payload.get("status") != "FORECAST":
        return _insufficient("power forecast unavailable")

    fc_kwh = forecast_energy_kwh(forecast_horizon_payload)

    # --- rate (existing flat-rate model; >0 required for a bill number) ---
    try:
        r = float(rate)
    except (TypeError, ValueError):
        r = float("nan")
    rate_ok = math.isfinite(r) and r > 0
    rate_field = round(r, 6) if math.isfinite(r) else 0.0

    actual = max(0.0, actual)
    fc = max(0.0, fc_kwh)
    estimated_total = actual + fc

    if rate_ok:
        estimated_bill = estimated_total * r
        current_bill = actual * r
        predicted_difference = estimated_bill - current_bill  # == fc * r
    else:
        estimated_bill = None
        predicted_difference = None

    conf = forecast_confidence or forecast_horizon_payload.get("confidence") or "low"

    return {
        "status": "OK",
        "reason": None,
        "actual_energy_kwh": round(actual, 6),
        "forecast_energy_kwh": round(fc, 6),
        "estimated_total_energy_kwh": round(estimated_total, 6),
        "rate": rate_field,
        "estimated_bill": round(estimated_bill, 4) if estimated_bill is not None else None,
        "predicted_difference": round(predicted_difference, 4) if predicted_difference is not None else None,
        "forecast_confidence": conf,
        "billing_period": billing_period,
    }


def predict_bill_from_record(
    actual_energy_kwh,
    forecast_record,
    horizon: str = "forecast_24h",
    rate: float = 0.0,
    billing_period: str = "30d",
) -> dict:
    """Convenience wrapper that pulls the horizon payload + anchor + confidence
    out of either a Stage 9 ForecastResult or a persisted forecast dict."""
    if forecast_record is None:
        return _insufficient("power forecast unavailable")

    if hasattr(forecast_record, horizon):  # ForecastResult
        hp = getattr(forecast_record, horizon)
        anchor = getattr(forecast_record, "anchor_timestamp", None)
        conf = hp.get("confidence") if isinstance(hp, dict) else None
    elif isinstance(forecast_record, dict):
        hp = forecast_record.get(horizon)
        anchor = forecast_record.get("anchor_timestamp")
        conf = forecast_record.get("confidence")
    else:
        return _insufficient("power forecast unavailable")

    result = predict_bill(
        actual_energy_kwh, hp, rate=rate,
        billing_period=billing_period, forecast_confidence=conf,
    )
    result["anchor_timestamp"] = int(anchor) if anchor else None
    return result


def _insufficient(reason: str) -> dict:
    return {
        "status": "INSUFFICIENT",
        "reason": reason,
        "actual_energy_kwh": None,
        "forecast_energy_kwh": None,
        "estimated_total_energy_kwh": None,
        "rate": 0.0,
        "estimated_bill": None,
        "predicted_difference": None,
        "forecast_confidence": None,
        "billing_period": None,
    }


# ---------------------------------------------------------------------------
# Actual energy from existing history (reuses Stage 1 loader)
# ---------------------------------------------------------------------------

def compute_actual_energy_from_history(settings: Optional[Settings] = None) -> float:
    """Sums per-PZEM cumulative-energy deltas over the available history window,
    reusing ai.data_loader (Stage 1) — the SAME history/ data the dashboard's
    manual Bill Calculator reads. No second pipeline is created.

    A per-PZEM delta requires >= 2 energy readings (start & end); a meter
    reset (counter decrease) is clamped to 0, never producing negative energy.
    """
    from . import data_loader

    settings = settings or get_settings()
    results = data_loader.fetch_all_history(settings=settings)
    total = 0.0
    for _n, hlr in results.items():
        frame = getattr(hlr, "frame", None)
        if frame is None or getattr(frame, "empty", True):
            continue
        if "energy" not in frame.columns:
            continue
        energy = frame["energy"]
        energy = energy if hasattr(energy, "iloc") else None
        if energy is None:
            continue
        vals = energy.dropna()
        if len(vals) < 2:
            continue
        try:
            start_e = float(vals.iloc[0])
            end_e = float(vals.iloc[-1])
        except (TypeError, ValueError):
            continue
        consumption = end_e - start_e
        if consumption < 0:
            consumption = 0.0
        if math.isfinite(consumption):
            total += consumption
    return round(max(0.0, total), 6)


# ---------------------------------------------------------------------------
# Firebase payload + persistence (dedicated /ai/bill_prediction namespace)
# ---------------------------------------------------------------------------

def build_bill_prediction_payload(
    result: dict,
    anchor_timestamp: int,
    source_stage: str = SOURCE_STAGE,
) -> dict:
    """JSON-safe Firebase payload (no NaN). Mirrors the spec's required fields."""
    return {
        "timestamp": int(anchor_timestamp),
        "anchor_timestamp": int(anchor_timestamp),
        "source_stage": source_stage,
        "status": result.get("status"),
        "reason": result.get("reason"),
        "actual_energy_kwh": result.get("actual_energy_kwh"),
        "forecast_energy_kwh": result.get("forecast_energy_kwh"),
        "estimated_total_energy_kwh": result.get("estimated_total_energy_kwh"),
        "rate": result.get("rate"),
        "estimated_bill": result.get("estimated_bill"),
        "predicted_difference": result.get("predicted_difference"),
        "forecast_confidence": result.get("forecast_confidence"),
        "billing_period": result.get("billing_period"),
    }


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
    from pathlib import Path
    cred_path = Path(settings.firebase_service_account_path)
    if not cred_path.exists():
        raise RuntimeError(
            f"Service account file not found at {cred_path}. Never commit this file."
        )
    cred = credentials.Certificate(str(cred_path))
    _firebase_app = firebase_admin.initialize_app(
        cred, {"databaseURL": settings.firebase_database_url}
    )
    logger.info("Firebase Admin SDK initialized against %s", settings.firebase_database_url)
    return _firebase_app


def _db_ref(path: str):
    from firebase_admin import db
    _init_firebase()
    return db.reference(path)


def write_bill_prediction(result: dict, anchor_timestamp) -> bool:
    """Writes the bill prediction to /ai/bill_prediction/<anchor-timestamp>.

    Idempotent: the child key is the deterministic anchor timestamp and an
    existing key is skipped. Returns True if written OR already present,
    False on failure / nothing-to-write. One failure never raises.
    """
    if result.get("status") != "OK" or not anchor_timestamp:
        logger.debug(
            "Skipping bill-prediction persist (status=%s).", result.get("status")
        )
        return False
    try:
        payload = build_bill_prediction_payload(result, int(anchor_timestamp))
    except (ValueError, TypeError) as exc:
        logger.warning("Invalid bill-prediction payload: %s", exc)
        return False

    key = str(int(anchor_timestamp))
    try:
        ref = _db_ref("ai/bill_prediction")
        if ref.child(key).get() is not None:
            logger.debug("ai/bill_prediction/%s already exists; skipping (idempotent).", key)
            return True
        ref.child(key).set(payload)
        logger.info("Wrote bill prediction to ai/bill_prediction/%s", key)
        return True
    except Exception as exc:  # noqa: BLE001 - pipeline resilience contract
        logger.error("Firebase write failed for ai/bill_prediction/%s: %s", key, exc)
        return False


def run_stage_10_pipeline(
    actual_energy_kwh=None,
    forecast_system_record=None,
    horizon: str = "forecast_24h",
    rate: float = 0.0,
    billing_period: str = "30d",
    settings: Optional[Settings] = None,
) -> dict:
    """Full Stage 10 flow: build the prediction from existing data and persist
    it under /ai/bill_prediction. Designed to run AFTER Stage 9 in the
    existing execution flow.

    `actual_energy_kwh` and `forecast_system_record` are optional: when
    omitted they are derived from the existing Stage 1/2/9 pipelines (no new
    data-loading path is created).
    """
    settings = settings or get_settings()

    if actual_energy_kwh is None:
        actual_energy_kwh = compute_actual_energy_from_history(settings=settings)
    if forecast_system_record is None:
        from . import forecast as fc
        _results, system = fc.run_forecast_pipeline(settings=settings)
        forecast_system_record = system

    result = predict_bill_from_record(
        actual_energy_kwh, forecast_system_record,
        horizon=horizon, rate=rate, billing_period=billing_period,
    )

    anchor = result.get("anchor_timestamp")
    if anchor is None and hasattr(forecast_system_record, "anchor_timestamp"):
        anchor = getattr(forecast_system_record, "anchor_timestamp")
    if anchor is None and isinstance(forecast_system_record, dict):
        anchor = forecast_system_record.get("anchor_timestamp")

    written = False
    if result.get("status") == "OK" and anchor:
        written = write_bill_prediction(result, anchor)

    return {"result": result, "written": written}
