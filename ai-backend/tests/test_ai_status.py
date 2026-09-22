"""
tests/test_ai_status.py — Stage 16 AI Monitoring Status Model tests.
Tests the truthful AI status model that distinguishes AVAILABLE, NO_EVENT,
INSUFFICIENT_DATA, NOT_RUN, ERROR.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from ai.ai_status import (
    AIStatus,
    AIStatusEntry,
    compute_ai_status,
    compute_all_ai_status,
)
from ai.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        firebase_service_account_path="unused-in-tests.json",
        firebase_database_url="https://unused-in-tests.example/",
        pzem_count=9,
        history_retention_days=60,
        cache_dir=tmp_path,
        anthropic_api_key="",
    )


def _make_result_frame(n_pts: int = 5, power: float = 200.0,
                        anomaly_labels: Optional[list] = None) -> pd.DataFrame:
    """Create a minimal anomaly result frame."""
    now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    ts = np.arange(now, now + n_pts * 300, 300)
    if anomaly_labels is None:
        anomaly_labels = ["NORMAL"] * n_pts
    n = min(n_pts, len(anomaly_labels))
    return pd.DataFrame({
        "timestamp": ts[:n],
        "anomaly_label": anomaly_labels[:n],
        "anomaly_score_normalized": [0.1] * n,
        "anomaly_severity_provisional": ["N/A"] * n,
    })


def _make_anomaly_result(pzem_number: int = 1, model_status: str = "READY",
                          frame: Optional[pd.DataFrame] = None,
                          training_rows: int = 10) -> "AnomalyDetectionResult":
    """Create a minimal AnomalyDetectionResult for testing."""
    from ai.anomaly_detection import AnomalyDetectionResult
    if model_status == "INSUFFICIENT_DATA":
        result_frame = None
    elif frame is None:
        result_frame = _make_result_frame()
    else:
        result_frame = frame
    return AnomalyDetectionResult(
        pzem_number=pzem_number,
        model_status=model_status,
        reason=None if model_status == "READY" else "Not enough training data",
        training_rows=training_rows,
        features_used=["power"],
        result_frame=result_frame,
    )


# ===========================================================================
# 1. AIStatus enum values
# ===========================================================================

def test_ai_status_enum_values():
    assert AIStatus.AVAILABLE.value == "AVAILABLE"
    assert AIStatus.NO_EVENT.value == "NO_EVENT"
    assert AIStatus.INSUFFICIENT_DATA.value == "INSUFFICIENT_DATA"
    assert AIStatus.NOT_RUN.value == "NOT_RUN"
    assert AIStatus.ERROR.value == "ERROR"


# ===========================================================================
# 2. AVAILABLE status — anomaly exists
# ===========================================================================

def test_available_with_anomaly():
    """AVAILABLE means a valid anomaly result exists."""
    frame = _make_result_frame(anomaly_labels=["NORMAL", "ANOMALY"])
    result = _make_anomaly_result(frame=frame)
    entry = compute_ai_status(
        anomaly_result=result,
        fault_events=[],
        last_pipeline_run=int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
    )
    assert entry.pzem_number == 1
    assert entry.status == AIStatus.AVAILABLE
    assert entry.anomaly_label == "ANOMALY"


# ===========================================================================
# 3. NO_EVENT status — AI ran, no anomaly found
# ===========================================================================

def test_no_event_status():
    """NO_EVENT means AI ran successfully but found no anomaly."""
    frame = _make_result_frame(anomaly_labels=["NORMAL", "NORMAL"])
    result = _make_anomaly_result(frame=frame)
    entry = compute_ai_status(
        anomaly_result=result,
        fault_events=[],
        last_pipeline_run=int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
    )
    assert entry.status == AIStatus.NO_EVENT


# ===========================================================================
# 4. INSUFFICIENT_DATA status
# ===========================================================================

def test_insufficient_data_status():
    """INSUFFICIENT_DATA means data exists but not enough history."""
    result = _make_anomaly_result(model_status="INSUFFICIENT_DATA", training_rows=5)
    entry = compute_ai_status(
        anomaly_result=result,
        fault_events=[],
        last_pipeline_run=None,
    )
    assert entry.status == AIStatus.INSUFFICIENT_DATA
    assert entry.model_status == "INSUFFICIENT_DATA"
    assert entry.reason is not None


# ===========================================================================
# 5. NOT_RUN status — pipeline has not generated results
# ===========================================================================

def test_not_run_status():
    """NOT_RUN means pipeline has never generated results."""
    entry = compute_ai_status(
        anomaly_result=None,
        fault_events=None,
        last_pipeline_run=None,
    )
    assert entry.status == AIStatus.NOT_RUN
    assert entry.reason == "AI pipeline has not generated results for this meter"


# ===========================================================================
# 6. ERROR status — pipeline execution failed
# ===========================================================================

def test_error_status():
    """ERROR means pipeline execution failed."""
    entry = compute_ai_status(
        anomaly_result=None,
        fault_events=None,
        pipeline_error="IsolationForest training failed: out of memory",
        last_pipeline_run=int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
    )
    assert entry.status == AIStatus.ERROR
    assert entry.reason == "IsolationForest training failed: out of memory"


# ===========================================================================
# 7. NO_EVENT when pipeline ran but no faults
# ===========================================================================

def test_no_event_with_faults():
    """When fault events exist, status is AVAILABLE."""
    from ai.fault_diagnosis import FaultEvent
    now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    faults = [FaultEvent(
        pzem_number=1,
        timestamp=now,
        fault_type="overvoltage",
        severity="EMERGENCY",
        measured_value=260.0,
        reason="Overvoltage detected",
    )]
    entry = compute_ai_status(
        anomaly_result=None,
        fault_events=faults,
        last_pipeline_run=now,
    )
    assert entry.status == AIStatus.AVAILABLE
    assert entry.fault_type == "overvoltage"


# ===========================================================================
# 8. compute_all_ai_status returns entries for all PZEMs
# ===========================================================================

def test_compute_all_ai_status_returns_all_pzems():
    """compute_all_ai_status should return entries for all PZEM numbers found."""
    result1 = _make_anomaly_result(pzem_number=1, frame=_make_result_frame())
    result2 = _make_anomaly_result(pzem_number=2, frame=_make_result_frame())
    status_map = compute_all_ai_status(
        anomaly_results={1: result1, 2: result2},
        fault_results_map={1: [], 2: []},
        last_pipeline_run=int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
    )
    assert len(status_map) >= 2
    for n in (1, 2):
        assert n in status_map
        assert isinstance(status_map[n], AIStatusEntry)
        assert status_map[n].pzem_number == n


def test_fault_results_reach_ai_status():
    """When anomaly_result is NO_EVENT and fault_events exist,
    compute_all_ai_status must propagate fault results so that
    compute_ai_status derives AVAILABLE status from the fault."""
    from ai.fault_diagnosis import FaultEvent
    now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    result = _make_anomaly_result(
        pzem_number=1, model_status="NO_EVENT",
        frame=_make_result_frame(anomaly_labels=["NORMAL", "NORMAL"]),
    )
    fault = FaultEvent(
        pzem_number=1, timestamp=now,
        fault_type="OVER_VOLTAGE", severity="WARNING",
        measured_value=250.0,
        reason="Voltage exceeded 250V threshold",
        confidence=0.95,
    )
    status_map = compute_all_ai_status(
        anomaly_results={1: result},
        fault_results_map={1: [fault]},
        last_pipeline_run=now,
    )
    assert 1 in status_map
    entry = status_map[1]
    assert entry.status == AIStatus.AVAILABLE
    assert entry.fault_type == "OVER_VOLTAGE"
    assert entry.severity == "WARNING"
    assert entry.measured_value == 250.0
    assert entry.timestamp == now


# ===========================================================================
# 9. AIStatusEntry dataclass
# ===========================================================================

def test_ai_status_entry_defaults():
    entry = AIStatusEntry(pzem_number=1)
    assert entry.status == AIStatus.NOT_RUN
    assert entry.severity is None
    assert entry.anomaly_label is None
    assert entry.fault_type is None
    assert entry.reason is None
    assert entry.timestamp is None


# ===========================================================================
# 10. Timestamp units are Unix seconds (not milliseconds)
# ===========================================================================

def test_ai_status_uses_unix_seconds():
    now_s = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    entry = compute_ai_status(
        anomaly_result=None,
        fault_events=None,
        last_pipeline_run=now_s,
    )
    assert entry.last_pipeline_run == now_s
    assert entry.last_pipeline_run < 10**12  # Not milliseconds


# ===========================================================================
# 11. PZEM mapping is correct
# ===========================================================================

def test_pzem_mapping_correct():
    """PZEM number should be preserved correctly."""
    frame = _make_result_frame()
    result = _make_anomaly_result(pzem_number=3, frame=frame)
    entry = compute_ai_status(
        anomaly_result=result,
        fault_events=[],
        last_pipeline_run=int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
    )
    assert entry.pzem_number == 3


# ===========================================================================
# 12. NO_EVENT when pipeline ran but found nothing
# ===========================================================================

def test_no_event_when_ran_but_nothing_found():
    """Pipeline ran but found no anomaly/fault → NO_EVENT, not NOT_RUN."""
    now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    entry = compute_ai_status(
        anomaly_result=None,
        fault_events=None,
        last_pipeline_run=now,
    )
    assert entry.status == AIStatus.NO_EVENT


# ===========================================================================
# 13. Empty anomaly result frame handled
# ===========================================================================

def test_empty_anomaly_result_frame():
    """Empty frame with no scored rows produces NO_EVENT."""
    frame = pd.DataFrame(columns=[
        "timestamp", "anomaly_label", "anomaly_score_normalized",
        "anomaly_severity_provisional"
    ])
    result = _make_anomaly_result(frame=frame)
    entry = compute_ai_status(
        anomaly_result=result,
        fault_events=[],
        last_pipeline_run=int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
    )
    # Empty frame → no scored rows → NO_EVENT
    assert entry.status == AIStatus.NO_EVENT


# ===========================================================================
# 14. Pipeline error takes precedence over all other states
# ===========================================================================

def test_pipeline_error_takes_precedence():
    """Pipeline error status takes precedence over AVAILABLE."""
    frame = _make_result_frame(anomaly_labels=["ANOMALY"])
    result = _make_anomaly_result(frame=frame)
    entry = compute_ai_status(
        anomaly_result=result,
        fault_events=[],
        pipeline_error="Pipeline crashed",
        last_pipeline_run=int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
    )
    assert entry.status == AIStatus.ERROR
    assert entry.reason == "Pipeline crashed"


# ===========================================================================
# 15. NO_EVENT vs NOT_RUN distinction
# ===========================================================================

def test_no_event_vs_not_run_distinction():
    """NO_EVENT means pipeline ran; NOT_RUN means it never ran."""
    now = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    # With last_pipeline_run set → NO_EVENT
    entry_run = compute_ai_status(None, None, last_pipeline_run=now)
    assert entry_run.status == AIStatus.NO_EVENT
    # Without last_pipeline_run → NOT_RUN
    entry_norun = compute_ai_status(None, None, last_pipeline_run=None)
    assert entry_norun.status == AIStatus.NOT_RUN
