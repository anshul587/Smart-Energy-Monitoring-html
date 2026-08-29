"""
ai/persist_ai_results.py
------------------------
STAGE 5: Persist AI Anomaly and Fault Results to Firebase.

Reads the output of Stages 3 (anomaly detection) and 4 (fault diagnosis)
and writes them to a dedicated /ai hierarchy in Firebase RTDB.

Paths written:
  /ai/anomalies/pzem_N/<timestamp>    ← anomaly results from Stage 3
  /ai/faults/pzem_N/<timestamp>       ← fault results from Stage 4

Does NOT write to /meters, /history, or /alerts.

Idempotency: the timestamp (Unix seconds) is used as the Firebase key.
Re-running the AI pipeline on the same source data will skip records
that already exist for the same PZEM + timestamp + result type, so no
uncontrolled duplicates are created.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ai.config import Settings, get_settings
from ai.preprocessing import PreprocessResult
from ai.fault_diagnosis import FaultEvent, run_fault_diagnosis_pipeline
from ai.anomaly_detection import AnomalyDetectionResult, run_anomaly_detection_pipeline

logger = logging.getLogger("ai.persist_ai_results")


# ---------------------------------------------------------------------------
# Firebase bootstrap (same pattern as data_loader.py)
# ---------------------------------------------------------------------------

_firebase_app = None


def _init_firebase():
    """Initialise the Firebase Admin SDK exactly once per process."""
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
            f"Service account file not found at {cred_path}. "
            "Never commit this file."
        )

    cred = credentials.Certificate(cred_path)
    _firebase_app = firebase_admin.initialize_app(
        cred, {"databaseURL": settings.firebase_database_url}
    )
    logger.info("Firebase Admin SDK initialized against %s", settings.firebase_database_url)
    return _firebase_app


def _db_ref(path: str):
    """Return a Firebase RTDB reference for the given relative path."""
    from firebase_admin import db
    _init_firebase()
    return db.reference(path)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_pzem_number(pzem_number: int, max_count: Optional[int] = None) -> None:
    """Validate that a PZEM number is within the configured range."""
    settings = get_settings()
    limit = max_count or settings.pzem_count
    if not (1 <= pzem_number <= limit):
        raise ValueError(
            f"pzem_number must be between 1 and {limit}, got {pzem_number}"
        )


def _validate_timestamp(ts) -> int:
    """Ensure timestamp is a valid Unix-second integer. Returns int."""
    if ts is None:
        raise ValueError("timestamp is None; cannot write to Firebase without a timestamp.")
    try:
        ts_int = int(ts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timestamp must be convertible to int, got {ts!r}") from exc
    if ts_int < 0:
        raise ValueError(f"timestamp must be a non-negative Unix second, got {ts_int}.")
    return ts_int


def _validate_numeric_value(val, field_name: str) -> Optional[float]:
    """Validate a numeric value is finite. Returns None if val is None/NaN."""
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be numeric, got {val!r}.")
    if not np.isfinite(v):
        raise ValueError(f"{field_name} must be finite, got {v} (NaN/Inf).")
    return v


def _validate_result_payload(
    pzem_number: int,
    timestamp: int,
    anomaly_result: Optional[object] = None,
    fault_result: Optional[object] = None,
) -> None:
    """Validate payload before writing to Firebase."""
    _validate_pzem_number(pzem_number)
    _validate_timestamp(timestamp)

    if anomaly_result is not None and not isinstance(anomaly_result, AnomalyDetectionResult):
        raise TypeError("anomaly_result must be an AnomalyDetectionResult or None.")

    if fault_result is not None and not isinstance(fault_result, FaultEvent):
        raise TypeError("fault_result must be a FaultEvent or None.")


# ---------------------------------------------------------------------------
# Anomaly result persistence
# ---------------------------------------------------------------------------

def _anomaly_payload(result: AnomalyDetectionResult) -> dict:
    """Build the Firebase payload for an anomaly detection result.

    Persists only values that Stage 3 actually produces. Does NOT invent
    new values not produced by Stage 3.
    """
    # Core identifying fields
    pzem = result.pzem_number
    ts = int(result.result_frame["timestamp"].iloc[0]) if result.result_frame is not None and not result.result_frame.empty else int(result.timestamp) if hasattr(result, "timestamp") else None

    # Gather the latest scored row (skip NOT_SCORED)
    scored = result.result_frame[result.result_frame["anomaly_label"] != "NOT_SCORED"] if result.result_frame is not None and not result.result_frame.empty else pd.DataFrame()

    if scored.empty:
        # No scored rows – still persist what we can (metadata only)
        return {
            "pzem_number": pzem,
            "timestamp": ts,
            "anomaly_label": "NOT_SCORED",
            "anomaly_score": None,
            "anomaly_score_normalized": None,
            "anomaly_severity_provisional": "N/A",
            "operating_state": None,
            "model_status": result.model_status,
            "training_rows": result.training_rows,
            "features_used": result.features_used,
            "contamination": result.contamination,
            "random_state": result.random_state,
            "active_rows": result.active_rows,
            "inactive_rows": result.inactive_rows,
            "active_days_represented": result.active_days_represented,
            "source_stage": "stage3/anomaly_detection",
        }

    latest = scored.iloc[-1]

    # anomaly_score / anomaly_score_normalized are float – may be NaN for weird edge cases
    raw_score = float(latest["anomaly_score"]) if not np.isnan(latest["anomaly_score"]) else None
    norm_score = float(latest["anomaly_score_normalized"]) if not np.isnan(latest["anomaly_score_normalized"]) else None

    return {
        "pzem_number": pzem,
        "timestamp": ts,
        "anomaly_score": raw_score,
        "anomaly_score_normalized": norm_score,
        "anomaly_label": str(latest["anomaly_label"]),
        "anomaly_severity_provisional": str(latest["anomaly_severity_provisional"]),
        "operating_state": str(latest["operating_state"]) if "operating_state" in latest.index else None,
        "model_status": result.model_status,
        "training_rows": result.training_rows,
        "features_used": result.features_used,
        "contamination": result.contamination,
        "random_state": result.random_state,
        "active_rows": result.active_rows,
        "inactive_rows": result.inactive_rows,
        "active_days_represented": result.active_days_represented,
        "source_stage": "stage3/anomaly_detection",
    }


def write_anomaly_result(result: AnomalyDetectionResult) -> bool:
    """Write one anomaly detection result to Firebase at /ai/anomalies/pzem_N/<timestamp>.

    Returns True if the write was attempted (even if already existed / skipped),
    False if a fatal configuration error prevented the attempt or the result
    is invalid (e.g. not READY, no scored rows).
    """
    try:
        _validate_result_payload(result.pzem_number, int(result.timestamp) if hasattr(result, "timestamp") else 0, anomaly_result=result)
    except (ValueError, TypeError) as exc:
        logger.warning("Invalid anomaly result for PZEM %d: %s", result.pzem_number, exc)
        return False

    # Reject results that are not READY — no model trained, no scores to persist.
    if result.model_status != "READY":
        logger.debug(
            "Anomaly result for PZEM %d not READY (status=%s); skipping write.",
            result.pzem_number,
            result.model_status,
        )
        return False

    # Reject results where there are no scored rows (all NOT_SCORED).
    # Per the spec: "Do NOT write NOT_SCORED results as anomalies unless there
    # is a clear reason and documented schema for them."
    if result.result_frame is not None and not result.result_frame.empty:
        scored = result.result_frame[result.result_frame["anomaly_label"] != "NOT_SCORED"]
        if scored.empty:
            logger.debug(
                "Anomaly result for PZEM %d has no scored rows (all NOT_SCORED); skipping write.",
                result.pzem_number,
            )
            return False

    payload = _anomaly_payload(result)
    pzem = payload["pzem_number"]
    ts = payload["timestamp"]

    _validate_pzem_number(pzem)

    try:
        ref = _db_ref(f"ai/anomalies/pzem_{pzem}")
        existing = ref.child(str(ts)).get()
        if existing is not None:
            logger.debug(
                "Anomaly result already exists for PZEM %d timestamp %s; skipping (idempotent).",
                pzem,
                ts,
            )
            return True

        ref.child(str(ts)).set(payload)
        logger.info(
            "Wrote anomaly result for PZEM %d timestamp %s to /ai/anomalies/pzem_%d/%s",
            pzem,
            ts,
            pzem,
            ts,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - one meter's Firebase failure must
        # not take down the whole AI pipeline.
        logger.error(
            "Firebase write failed for PZEM %d anomaly timestamp %s: %s",
            pzem,
            ts,
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# Fault result persistence
# ---------------------------------------------------------------------------

def _fault_payload(event: FaultEvent) -> dict:
    """Build the Firebase payload for a fault diagnosis result.

    Persists only values from the FaultEvent. Does NOT duplicate the
    /alerts emergency-alert structure; /ai/faults is a separate namespace.
    """
    base = {
        "pzem_number": event.pzem_number,
        "timestamp": event.timestamp,
        "fault_type": event.fault_type,
        "severity": event.severity,
        "measured_value": event.measured_value,
        "reason": event.reason,
        "confidence": event.confidence,
        "source_stage": "stage4/fault_diagnosis",
    }
    if event.evidence:
        base["evidence"] = event.evidence
    return base


def write_fault_result(event: FaultEvent) -> bool:
    """Write one fault diagnosis result to Firebase at /ai/faults/pzem_N/<timestamp>.

    Returns True if the write was attempted, False if a fatal configuration
    error prevented the attempt.
    """
    try:
        _validate_pzem_number(event.pzem_number)
    except ValueError as exc:
        logger.warning("Invalid fault event PZEM number: %s", exc)
        return False

    payload = _fault_payload(event)
    pzem = payload["pzem_number"]
    ts = payload["timestamp"]

    try:
        ref = _db_ref(f"ai/faults/pzem_{pzem}")
        # Idempotency: check whether a record with this exact timestamp
        # already exists under this PZEM's fault path.
        existing = ref.child(str(ts)).get()
        if existing is not None:
            logger.debug(
                "Fault result already exists for PZEM %d timestamp %s; skipping (idempotent).",
                pzem,
                ts,
            )
            return True

        ref.child(str(ts)).set(payload)
        logger.info(
            "Wrote fault result for PZEM %d timestamp %s to /ai/faults/pzem_%d/%s",
            pzem,
            ts,
            pzem,
            ts,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - one meter's Firebase failure must
        # not take down the whole AI pipeline.
        logger.error(
            "Firebase write failed for PZEM %d fault timestamp %s: %s",
            pzem,
            ts,
            exc,
        )
        return False


# ---------------------------------------------------------------------------
# Pipeline integration: persist AI results after Stages 3 & 4
# ---------------------------------------------------------------------------

def persist_anomaly_results(results: dict[int, AnomalyDetectionResult]) -> dict[int, int]:
    """Persist anomaly results for all PZEMs.

    Returns a dict mapping PZEM number -> count of attempted writes
    (always equals len(results) if all results passed validation;
    actual Firebase successes may be fewer if some meters fail).
    """
    settings = get_settings()
    counts: dict[int, int] = {}
    for pzem_number in range(1, settings.pzem_count + 1):
        result = results.get(pzem_number)
        if result is None or result.model_status != "READY" or result.result_frame is None:
            counts[pzem_number] = 0
            continue
        ok = write_anomaly_result(result)
        counts[pzem_number] = 1 if ok else 0
    return counts


def persist_fault_results(
    preprocess_results: dict[int, PreprocessResult],
) -> dict[int, int]:
    """Run fault diagnosis on all PZEMs and persist the results to Firebase.

    Returns a dict mapping PZEM number -> count of fault events written
    (0 = no events, or all writes failed).
    """
    try:
        fault_events_map = run_fault_diagnosis_pipeline(preprocess_results)
    except Exception as exc:  # noqa: BLE001 - one meter's diagnosis failure
        # must not stop the whole Stage 5 pipeline.
        logger.error("Fault diagnosis pipeline failed: %s", exc)
        return {n: 0 for n in range(1, get_settings().pzem_count + 1)}

    counts: dict[int, int] = {}
    for pzem_number in range(1, get_settings().pzem_count + 1):
        events = fault_events_map.get(pzem_number, [])
        written = 0
        for event in events:
            ok = write_fault_result(event)
            if ok:
                written += 1
        counts[pzem_number] = written
    return counts


# ---------------------------------------------------------------------------
# High-level convenience: run Stages 3-5 end-to-end
# ---------------------------------------------------------------------------

def run_stage_5_pipeline(
    preprocess_results: dict[int, PreprocessResult],
    anomaly_results: Optional[dict[int, AnomalyDetectionResult]] = None,
) -> dict[str, dict[int, int]]:
    """Run Stages 3 + 5 (and implicitly Stage 4 via the fault pipeline).

    Preferred flow (as specified):
      load data
        -> preprocess          (already done via preprocess_results)
        -> anomaly detection   (run if anomaly_results not supplied)
        -> fault diagnosis     (run via persist_fault_results)
        -> persist AI results
        -> existing report/output

    Returns a dict with keys "anomalies" and "faults", each mapping
    PZEM number -> count of writes attempted.
    """
    if anomaly_results is None:
        from ai.anomaly_detection import run_anomaly_detection_pipeline
        anomaly_results = run_anomaly_detection_pipeline(settings=get_settings(), preprocess_results=preprocess_results)

    anomaly_counts = persist_anomaly_results(anomaly_results)
    fault_counts = persist_fault_results(preprocess_results)

    return {"anomalies": anomaly_counts, "faults": fault_counts}