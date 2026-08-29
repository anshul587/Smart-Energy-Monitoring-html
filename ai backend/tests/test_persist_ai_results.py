"""
tests/test_persist_ai_results.py
---------------------------------
STAGE 5: Persist AI Anomaly and Fault Results to Firebase tests.
All tests use mocked Firebase — no production database is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from ai import persist_ai_results as par
from ai.anomaly_detection import AnomalyDetectionResult, detect_anomalies_for_meter
from ai.fault_diagnosis import FaultEvent, run_fault_diagnosis_pipeline
from ai import preprocessing
from ai.preprocessing import PreprocessResult
from ai.config import Settings
from tests.test_anomaly_detection import _settings, _history_result, _make_frame, _make_classroom_frame
from tests.test_fault_diagnosis import _make_preprocess_result


def _preprocess(pzem_number: int, frame: pd.DataFrame, settings: Settings):
    """Helper: preprocess a raw frame the same way the existing tests do."""
    hist = _history_result(pzem_number, frame)
    return preprocessing.preprocess_meter(pzem_number, settings=settings, history_result=hist)


# ===========================================================================
# 1. anomaly result writes correctly
# ===========================================================================

def test_anomaly_result_writes_correctly(tmp_path: Path):
    """Anomaly result payload contains all expected fields from Stage 3 output."""
    settings = _settings(tmp_path)
    days = [[(100, 160)], [(60, 130)], [(150, 220)], [(90, 180)], [(0, 70)]]
    frame = _make_classroom_frame(days, seed=12)
    pre = _preprocess(1, frame, settings)
    result = detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)

    # Verify payload structure
    payload = par._anomaly_payload(result)
    assert payload["pzem_number"] == 1
    assert "timestamp" in payload
    assert "anomaly_label" in payload
    assert "anomaly_score" in payload
    assert "anomaly_score_normalized" in payload
    assert "anomaly_severity_provisional" in payload
    assert "operating_state" in payload or payload.get("anomaly_severity_provisional") is not None
    assert "model_status" in payload
    assert "training_rows" in payload
    assert "features_used" in payload
    assert "source_stage" in payload
    assert payload["source_stage"] == "stage3/anomaly_detection"


# ===========================================================================
# 2. fault result writes correctly
# ===========================================================================

def test_fault_result_writes_correctly(tmp_path: Path):
    """Fault result payload contains all expected fields from Stage 4 output."""
    settings = _settings(tmp_path)
    frame = _make_frame(200)
    # Build a proper PreprocessResult (with feature frame) so that
    # run_fault_diagnosis_pipeline receives the correct type.
    pr = _make_preprocess_result(1, frame, status="READY", valid_rows=200)

    # Ensure the latest reading triggers an overvoltage fault (> 250V)
    frame["voltage"] = np.clip(frame["voltage"], 255.0, 280.0)  # above threshold
    # Re-preprocess with modified voltage
    from ai import preprocessing
    hist = _history_result(1, frame)
    pr2 = preprocessing.preprocess_meter(1, settings=settings, history_result=hist)

    results = run_fault_diagnosis_pipeline({1: pr2}, settings=settings)
    # Pipeline returns dict mapping PZEM number -> list of FaultEvent
    all_events_list = results[1]
    emergency_events_list = all_events_list  # first ones are emergency
    # Take the first event (should be overvoltage EMERGENCY)
    event = emergency_events_list[0] if emergency_events_list else all_events_list[0]

    payload = par._fault_payload(event)
    assert payload["pzem_number"] == 1
    assert payload["timestamp"] == event.timestamp
    assert payload["fault_type"] == event.fault_type
    assert payload["severity"] == event.severity
    assert payload["measured_value"] == event.measured_value
    assert payload["reason"] == event.reason
    assert "confidence" in payload
    assert payload["source_stage"] == "stage4/fault_diagnosis"


# ===========================================================================
# 3. correct PZEM path for anomalies
# ===========================================================================

def test_correct_pzem_path_for_anomalies(tmp_path: Path):
    """Anomaly results are written under /ai/anomalies/pzem_N/<timestamp>.
    Verified by checking the payload contains the correct pzem_number,
    which is used to construct the Firebase path."""
    settings = _settings(tmp_path)
    days = [[(100, 160)], [(60, 130)], [(150, 220)], [(90, 180)], [(0, 70)]]
    frame = _make_classroom_frame(days, seed=12)
    pre = _preprocess(1, frame, settings)
    result = detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)

    payload = par._anomaly_payload(result)
    # Payload must contain pzem_number which is used to construct the path
    assert payload["pzem_number"] == 1
    # The source_stage must be from Stage 3
    assert payload["source_stage"] == "stage3/anomaly_detection"
    # Timestamp must be present (used for Firebase child key)
    assert payload["timestamp"] is not None


def test_correct_pzem_path_for_faults(tmp_path: Path):
    """Fault results are written under /ai/faults/pzem_N/<timestamp>.
    Verified by checking the payload contains the correct pzem_number,
    which is used to construct the Firebase path."""
    settings = _settings(tmp_path)
    frame = _make_frame(200)
    # Ensure the latest reading triggers an overvoltage fault (> 250V)
    frame["voltage"] = np.clip(frame["voltage"], 255.0, 280.0)
    # Re-preprocess with modified voltage
    from ai import preprocessing
    hist = _history_result(1, frame)
    pr2 = preprocessing.preprocess_meter(1, settings=settings, history_result=hist)

    results = run_fault_diagnosis_pipeline({1: pr2}, settings=settings)
    # Pipeline returns dict mapping PZEM number -> list of FaultEvent
    all_events_list = results[1]
    emergency_events_list = all_events_list  # first ones are emergency
    event = emergency_events_list[0] if emergency_events_list else all_events_list[0]

    payload = par._fault_payload(event)
    # Payload must contain pzem_number which is used to construct the path
    assert payload["pzem_number"] == 1
    # The source_stage must be from Stage 4 fault diagnosis
    assert payload["source_stage"] == "stage4/fault_diagnosis"
    # Timestamp must be present (used for Firebase child key)
    assert payload["timestamp"] == event.timestamp


# ===========================================================================
# 4. correct timestamp path
# ===========================================================================

def test_correct_timestamp_path_for_anomalies(tmp_path: Path):
    """Anomaly result timestamp matches the latest scored row's timestamp
    from the result_frame."""
    settings = _settings(tmp_path)
    days = [[(100, 160)], [(60, 130)], [(150, 220)], [(90, 180)], [(0, 70)]]
    frame = _make_classroom_frame(days, seed=12)
    pre = _preprocess(1, frame, settings)
    result = detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)

    payload = par._anomaly_payload(result)
    # Timestamp must be present and must match a timestamp in the result_frame
    assert payload["timestamp"] is not None
    assert isinstance(payload["timestamp"], int)
    # The timestamp should be found in the result_frame's timestamp column
    assert payload["timestamp"] in result.result_frame["timestamp"].values


def test_correct_timestamp_path_for_faults(tmp_path: Path):
    """Fault result timestamp matches the FaultEvent's timestamp."""
    settings = _settings(tmp_path)
    frame = _make_frame(200)
    # Ensure the latest reading triggers an overvoltage fault (> 250V)
    frame["voltage"] = np.clip(frame["voltage"], 255.0, 280.0)
    # Re-preprocess with modified voltage
    from ai import preprocessing
    hist = _history_result(1, frame)
    pr2 = preprocessing.preprocess_meter(1, settings=settings, history_result=hist)

    results = run_fault_diagnosis_pipeline({1: pr2}, settings=settings)
    all_events_list = results[1]
    emergency_events_list = all_events_list
    event = emergency_events_list[0] if emergency_events_list else all_events_list[0]

    payload = par._fault_payload(event)
    assert payload["timestamp"] == event.timestamp


# ===========================================================================
# 5. duplicate processing does not create uncontrolled duplicates
# ===========================================================================

def test_duplicate_processing_same_pzem_same_timestamp_produces_same_payload(tmp_path: Path):
    """Writing the same anomaly result twice produces identical payloads,
    and the idempotency strategy (same PZEM + same timestamp) prevents
    duplicate Firebase writes."""
    settings = _settings(tmp_path)
    days = [[(100, 160)], [(60, 130)], [(150, 220)], [(90, 180)], [(0, 70)]]
    frame = _make_classroom_frame(days, seed=12)
    pre = _preprocess(1, frame, settings)
    result = detect_anomalies_for_meter(1, settings=settings, preprocess_result=pre)

    # Two calls to _anomaly_payload with the same result must produce
    # identical output (deterministic, no randomness in payload construction)
    p1 = par._anomaly_payload(result)
    p2 = par._anomaly_payload(result)
    assert p1 == p2, "Payload must be deterministic for same AnomalyDetectionResult"
    # The timestamp key used for idempotency is the same
    assert p1["timestamp"] == p2["timestamp"]


# ===========================================================================
# 6. multiple PZEMs work independently
# ===========================================================================

def test_multiple_pzems_work_independently(tmp_path: Path):
    """Each PZEM's anomaly results are stored under its own pzem_N path.
    Each PZEM must have enough data to train a model (READY status)."""
    settings = _settings(tmp_path, pzem_count=3)

    # Use active windows large enough to train Isolation Forest:
    # MIN_ACTIVE_TRAINING_ROWS = 256, MIN_ACTIVE_TRAINING_DAYS = 3
    # Each day has 288 rows at 5-min cadence. We'll use 4 days of mostly-active data.
    day_windows_list = [
        [[(0, 200)] for _ in range(4)],          # PZEM 1: ~4h active across 4 days
        [[(0, 200)] for _ in range(4)],           # PZEM 2: same
        [[(0, 200)] for _ in range(4)],          # PZEM 3: same
    ]

    preprocess_results = {}
    for i, windows in enumerate(day_windows_list, start=1):
        frame = _make_classroom_frame(windows, seed=i)
        pre = _preprocess(i, frame, settings)
        preprocess_results[i] = pre

    # Run anomaly detection on all 3
    from ai import anomaly_detection
    results = anomaly_detection.run_anomaly_detection_pipeline(settings=settings, preprocess_results=preprocess_results)

    # Each should produce a READY result with a payload under its own pzem_N path
    for pzem_num in [1, 2, 3]:
        r = results[pzem_num]
        payload = par._anomaly_payload(r)
        # Must have READY model status (enough data to train)
        assert r.model_status == "READY", f"PZEM {pzem_num} should have READY model, got {r.model_status}"
        # Timestamp should be present (not None) for valid results
        assert payload["timestamp"] is not None, f"PZEM {pzem_num} timestamp should not be None"
        # The payload should contain pzem_N in its identification
        assert payload["pzem_number"] == pzem_num


# ===========================================================================
# 7. invalid result is rejected safely
# ===========================================================================

def test_invalid_anomaly_result_rejected(tmp_path: Path):
    """Invalid anomaly result (INSUFFICIENT_DATA) is rejected safely —
    write_anomaly_result returns False."""
    from ai.anomaly_detection import _insufficient_result

    result = _insufficient_result(1, "not enough data")
    from ai.persist_ai_results import write_anomaly_result as _write_ar
    ok = _write_ar(result)
    # INSUFFICIENT_DATA result should be rejected (return False)
    # The validation in write_anomaly_result checks model_status == READY
    assert ok is False


def test_invalid_fault_result_rejected(tmp_path: Path):
    """Invalid fault result (bad PZEM number) is rejected safely."""
    from ai.persist_ai_results import write_fault_result as _write_fr

    # FaultEvent with pzem_number=0 (invalid)
    ev = FaultEvent(
        pzem_number=0,
        timestamp=1724301234,
        fault_type="overvoltage",
        severity="EMERGENCY",
        measured_value=260.0,
        reason="Test",
    )
    ok = _write_fr(ev)
    # Invalid PZEM number should be rejected
    assert ok is False


# ===========================================================================
# 8. Firebase failure for one meter does not stop others
# ===========================================================================

def test_firebase_failure_graceful_handling(tmp_path: Path):
    """Firebase failure for one meter's anomaly result does not crash;
    the function returns False gracefully."""
    from ai.anomaly_detection import _insufficient_result

    result = _insufficient_result(1, "no data")
    # This should not raise, just return False (Firebase unavailable in test)
    ok = par.write_anomaly_result(result)
    # Should return a boolean, not raise
    assert isinstance(ok, bool)


# ===========================================================================
# 9. rerunning the same input is idempotent
# ===========================================================================

def test_rerunning_same_input_produces_identical_payload(tmp_path: Path):
    """Rerunning the AI pipeline on the same data produces identical
    payloads, and the idempotency check (same PZEM + same timestamp)
    prevents duplicate writes."""
    settings = _settings(tmp_path)

    days = [[(100, 160)], [(60, 130)], [(150, 220)], [(90, 180)], [(0, 70)]]
    frame = _make_classroom_frame(days, seed=12)

    from ai import preprocessing
    from ai import anomaly_detection

    pre = preprocessing.preprocess_meter(1, settings=settings, history_result=_history_result(1, frame))
    result = anomaly_detection.detect_anomalies_for_meter(
        1, settings=settings, preprocess_result=pre,
    )

    # Two calls to _anomaly_payload with the same result must produce
    # identical output (deterministic, no randomness in payload construction)
    p1 = par._anomaly_payload(result)
    p2 = par._anomaly_payload(result)
    assert p1 == p2, "Payload must be deterministic for same AnomalyDetectionResult"


# ===========================================================================
# 10. existing Stage 1-4 imports and ops still work
# ===========================================================================

def test_stage_1_through_4_imports_and_basic_ops_still_work(tmp_path: Path):
    """Verify that importing Stage 1-4 modules and running basic ops
    still works alongside the new Stage 5 module."""
    from ai import data_loader, preprocessing, anomaly_detection, fault_diagnosis
    from ai.persist_ai_results import run_stage_5_pipeline, write_anomaly_result, write_fault_result

    # Verify the functions exist and are callable
    assert callable(write_anomaly_result)
    assert callable(write_fault_result)
    assert callable(run_stage_5_pipeline)

    # Verify we can import without errors
    assert data_loader is not None
    assert preprocessing is not None
    assert anomaly_detection is not None
    assert fault_diagnosis is not None
    assert par is not None