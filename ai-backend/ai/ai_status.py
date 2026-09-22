"""
ai/ai_status.py
---------------
Stage 16: AI Monitoring Status Model.

Provides a truthful, authoritative AI status per PZEM that the
dashboard can consume via the REST API. Distinguishes five states:

  AVAILABLE       - valid anomaly/fault/diagnostic result exists
  NO_EVENT        - AI analysis successfully ran, no anomaly/fault detected
  INSUFFICIENT_DATA - data exists but not enough historical data
  NOT_RUN         - pipeline has not generated results for this meter/period
  ERROR           - pipeline execution failed

The dashboard MUST use this status model instead of inferring from
Firebase child-existence, because "no Firebase record" can mean
ANY of the above states.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

logger = logging.getLogger("ai.ai_status")


class AIStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    NO_EVENT = "NO_EVENT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_RUN = "NOT_RUN"
    ERROR = "ERROR"


@dataclass
class AIStatusEntry:
    pzem_number: int
    status: AIStatus = AIStatus.NOT_RUN
    severity: Optional[str] = None
    anomaly_label: Optional[str] = None
    anomaly_score: Optional[float] = None
    fault_type: Optional[str] = None
    measured_value: Optional[float] = None
    reason: Optional[str] = None
    timestamp: Optional[int] = None
    last_pipeline_run: Optional[int] = None
    model_status: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline-level status tracking
# ---------------------------------------------------------------------------

def compute_ai_status(
    anomaly_result,
    fault_events,
    pipeline_error: Optional[str] = None,
    last_pipeline_run: Optional[int] = None,
    model_status: Optional[str] = None,
) -> AIStatusEntry:
    """Compute the authoritative AI status for one PZEM from pipeline outputs.

    Parameters
    ----------
    anomaly_result : AnomalyDetectionResult or None
        Output of ai.anomaly_detection.run_anomaly_detection_pipeline
    fault_events : list[FaultEvent] or None
        Output of ai.fault_diagnosis.run_fault_diagnosis_pipeline
    pipeline_error : str or None
        Non-None if the pipeline raised an exception for this meter
    last_pipeline_run : int or None
        Unix seconds of the last pipeline run for this meter
    model_status : str or None
        The model_status from the anomaly result (e.g. "READY", "INSUFFICIENT_DATA")
    """
    pzem = getattr(anomaly_result, "pzem_number", 1) if anomaly_result else 1
    entry = AIStatusEntry(pzem_number=pzem, last_pipeline_run=last_pipeline_run)

    # Pipeline error takes highest precedence
    if pipeline_error:
        entry.status = AIStatus.ERROR
        entry.reason = pipeline_error
        return entry

    # If anomaly result exists, derive status from it
    if anomaly_result is not None:
        entry.model_status = getattr(anomaly_result, "model_status", None)
        if entry.model_status == "INSUFFICIENT_DATA":
            entry.status = AIStatus.INSUFFICIENT_DATA
            entry.reason = getattr(anomaly_result, "reason", None)
            return entry

        if entry.model_status == "READY" and anomaly_result.result_frame is not None:
            scored = anomaly_result.result_frame[
                anomaly_result.result_frame["anomaly_label"] != "NOT_SCORED"
            ]
            if scored.empty:
                # Model is ready but no scored rows -> no anomaly event
                entry.status = AIStatus.NO_EVENT
                entry.model_status = "READY"
                return entry

            # Has scored anomalies
            latest_scored = scored.iloc[-1]
            label = str(latest_scored["anomaly_label"])
            entry.anomaly_label = label
            score_val = latest_scored["anomaly_score_normalized"]
            try:
                entry.anomaly_score = float(score_val) if float(score_val) == float(score_val) else None
            except (ValueError, TypeError):
                entry.anomaly_score = None
            entry.severity = str(latest_scored.get("anomaly_severity_provisional", "N/A"))
            entry.timestamp = int(latest_scored["timestamp"])

            if label == "ANOMALY":
                entry.status = AIStatus.AVAILABLE
                entry.reason = f"Anomaly detected: {entry.anomaly_label} (severity: {entry.severity})"
            else:
                entry.status = AIStatus.NO_EVENT
                entry.reason = "AI analysis completed; latest reading is NORMAL"
            return entry

    # If fault events exist, derive status from faults
    if fault_events:
        latest_fault = max(fault_events, key=lambda e: e.timestamp)
        entry.status = AIStatus.AVAILABLE
        entry.fault_type = latest_fault.fault_type
        entry.severity = latest_fault.severity
        entry.measured_value = latest_fault.measured_value
        entry.timestamp = latest_fault.timestamp
        entry.reason = latest_fault.reason
        return entry

    # No anomaly result, no fault events, no pipeline error
    if last_pipeline_run is not None:
        entry.status = AIStatus.NO_EVENT
        entry.reason = "AI analysis completed; no anomaly or fault detected"
    else:
        entry.status = AIStatus.NOT_RUN
        entry.reason = "AI pipeline has not generated results for this meter"

    return entry


def compute_all_ai_status(
    anomaly_results: dict[int, "AnomalyDetectionResult"],
    fault_results_map: dict[int, list],
    pipeline_errors: Optional[dict[int, str]] = None,
    last_pipeline_run: Optional[int] = None,
) -> dict[int, AIStatusEntry]:
    """Compute AI status for all PZEMs from pipeline outputs."""
    pipeline_errors = pipeline_errors or {}
    status_map: dict[int, AIStatusEntry] = {}

    for pzem_number in range(1, max(
        max(anomaly_results.keys(), default=0),
        max(fault_results_map.keys(), default=0),
    ) + 1):
        ar = anomaly_results.get(pzem_number)
        faults = fault_results_map.get(pzem_number, [])
        error = pipeline_errors.get(pzem_number)
        status_map[pzem_number] = compute_ai_status(
            anomaly_result=ar,
            fault_events=faults,
            pipeline_error=error,
            last_pipeline_run=last_pipeline_run,
        )

    return status_map


# ---------------------------------------------------------------------------
# Persistence to Firebase /ai/ai_status/pzem_N/<timestamp>
# ---------------------------------------------------------------------------

_firebase_app = None


def _init_firebase():
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app
    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError:
        raise RuntimeError("firebase-admin is not installed.")
    from ai.config import get_settings
    from pathlib import Path
    settings = get_settings()
    cred_path = Path(settings.firebase_service_account_path)
    cred = credentials.Certificate(str(cred_path))
    try:
        _firebase_app = firebase_admin.initialize_app(
            cred, {"databaseURL": settings.firebase_database_url}
        )
    except ValueError:
        _firebase_app = firebase_admin.get_app()
    logger.info("Firebase Admin SDK initialized against %s", settings.firebase_database_url)
    return _firebase_app


def _db_ref(path: str):
    from firebase_admin import db
    _init_firebase()
    return db.reference(path)


def persist_ai_status(status_entry: AIStatusEntry) -> bool:
    """Persist AI status for one PZEM to Firebase at /ai/ai_status/pzem_N/<timestamp>."""
    try:
        payload = {
            "pzem_number": status_entry.pzem_number,
            "timestamp": status_entry.last_pipeline_run or int(__import__("time").time()),
            "ai_status": status_entry.status.value,
            "severity": status_entry.severity,
            "anomaly_label": status_entry.anomaly_label,
            "anomaly_score": status_entry.anomaly_score,
            "fault_type": status_entry.fault_type,
            "measured_value": status_entry.measured_value,
            "reason": status_entry.reason,
            "model_status": status_entry.model_status,
        }
        ref = _db_ref(f"ai/ai_status/pzem_{status_entry.pzem_number}")
        key = str(status_entry.timestamp or int(__import__("time").time()))
        if ref.child(key).get() is not None:
            return True
        ref.child(key).set(payload)
        logger.info("Persisted AI status for PZEM %d: %s", status_entry.pzem_number, status_entry.status.value)
        return True
    except Exception as exc:
        logger.error("Failed to persist AI status for PZEM %d: %s", status_entry.pzem_number, exc)
        return False


def read_ai_status(pzem_number: int) -> Optional[AIStatusEntry]:
    """Read the latest AI status for one PZEM from Firebase."""
    try:
        ref = _db_ref(f"ai/ai_status/pzem_{pzem_number}")
        raw = ref.get()
        if not raw or not isinstance(raw, dict):
            return None
        # Find the most recent entry
        latest_key = None
        latest_ts = 0
        for k in raw:
            try:
                ts = int(k)
                if ts > latest_ts:
                    latest_ts = ts
                    latest_key = k
            except (TypeError, ValueError):
                continue
        if latest_key is None:
            return None
        rec = raw[latest_key]
        if not isinstance(rec, dict):
            return None
        return AIStatusEntry(
            pzem_number=int(rec.get("pzem_number", pzem_number)),
            status=AIStatus(rec.get("ai_status", "NOT_RUN")),
            severity=rec.get("severity"),
            anomaly_label=rec.get("anomaly_label"),
            anomaly_score=rec.get("anomaly_score"),
            fault_type=rec.get("fault_type"),
            measured_value=rec.get("measured_value"),
            reason=rec.get("reason"),
            timestamp=rec.get("timestamp"),
            last_pipeline_run=rec.get("last_pipeline_run"),
            model_status=rec.get("model_status"),
        )
    except Exception:
        logger.exception("Failed to read AI status for PZEM %d", pzem_number)
        return None


def read_all_ai_status() -> dict[int, AIStatusEntry]:
    """Read AI status for all PZEMs from Firebase."""
    from ai.config import get_settings
    settings = get_settings()
    result: dict[int, AIStatusEntry] = {}
    for n in range(1, settings.pzem_count + 1):
        entry = read_ai_status(n)
        if entry:
            result[n] = entry
        else:
            result[n] = AIStatusEntry(pzem_number=n, status=AIStatus.NOT_RUN)
    return result
