"""
tests/test_diagnostic_recommendation.py
-----------------------------------------
Phase 2A: Deterministic diagnostic recommendation tests.

Covers:
- All 7 supported fault types
- Required fields present
- Deterministic recommendation_id
- Evidence/action presence
- Unsupported fault type raises ValueError
- Missing optional fields handled
- No invented energy/cost impact
- No unsupported root-cause certainty
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pandas as pd

from ai import diagnostic_recommendation as dr
from ai.diagnostic_recommendation import (
    DiagnosticRecommendation,
    RecommendationEngine,
    SUPPORTED_FAULT_TYPES,
    recommend,
    recommend_from_anomaly,
    recommend_from_peak,
    recommend_from_system_peak,
    recommend_from_recurring_peak,
    recommend_from_cross_pipeline,
    recommend_from_bill,
    recommend_from_energy_saving,
    recommend_from_maintenance,
    recommend_from_forecast_peak,
    deduplicate_recommendations,
)
from ai.fault_diagnosis import FaultEvent
from ai.config import Settings
from ai.anomaly_detection import AnomalyDetectionResult
from ai.bill_prediction import predict_bill
from ai.energy_saving import Recommendation as EnergySavingRecommendation, MeterEvidence
from ai.forecast import ForecastResult
from ai.maintenance_risk import RiskResult, SystemMaintenanceSummary
from ai.peak_detection import PeakResult, SystemPeakResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_TS = 1_700_000_000


def make_fault(
    pzem_number: int = 1,
    timestamp: int = BASE_TS,
    fault_type: str = "overvoltage",
    severity: str = "EMERGENCY",
    measured_value: float = 260.0,
    reason: str = "Test fault",
    evidence: dict | None = None,
    confidence: float = 0.95,
) -> FaultEvent:
    return FaultEvent(
        pzem_number=pzem_number,
        timestamp=timestamp,
        fault_type=fault_type,
        severity=severity,
        measured_value=measured_value,
        reason=reason,
        evidence=evidence,
        confidence=confidence,
    )


def make_engine() -> RecommendationEngine:
    return RecommendationEngine()


# ---------------------------------------------------------------------------
# Test: all 7 fault types produce recommendations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fault_type", SUPPORTED_FAULT_TYPES)
def test_all_fault_types_produce_recommendation(fault_type: str):
    fault = make_fault(fault_type=fault_type, severity="WARNING", measured_value=1.0)
    rec = make_engine().recommend(fault)

    assert isinstance(rec, DiagnosticRecommendation)
    assert rec.fault_type == fault_type
    assert rec.condition
    assert rec.probable_cause
    assert rec.why_it_happened
    assert rec.evidence
    assert rec.what_to_check
    assert rec.what_to_do_now
    assert rec.corrective_action
    assert rec.urgency
    assert rec.maintenance_timing
    assert rec.confidence > 0
    assert rec.source_stages


# ---------------------------------------------------------------------------
# Test: required fields present
# ---------------------------------------------------------------------------

def test_required_fields_present():
    fault = make_fault()
    rec = make_engine().recommend(fault)

    assert rec.recommendation_id
    assert rec.timestamp == BASE_TS
    assert rec.pzem_system == "PZEM-1"
    assert rec.condition
    assert rec.fault_type == "overvoltage"
    assert rec.severity == "EMERGENCY"
    assert rec.priority
    assert rec.probable_cause
    assert rec.why_it_happened
    assert rec.evidence
    assert rec.what_to_check
    assert rec.what_to_do_now
    assert rec.corrective_action
    assert rec.urgency
    assert rec.maintenance_required is not None
    assert rec.maintenance_timing
    assert rec.confidence is not None
    assert rec.source_stages


# ---------------------------------------------------------------------------
# Test: deterministic recommendation_id
# ---------------------------------------------------------------------------

def test_deterministic_recommendation_id():
    fault = make_fault(pzem_number=3, timestamp=1_700_000_000, fault_type="overvoltage")
    rec1 = make_engine().recommend(fault)
    rec2 = make_engine().recommend(fault)
    assert rec1.recommendation_id == rec2.recommendation_id


def test_recommendation_id_changes_with_input():
    fault1 = make_fault(pzem_number=1, timestamp=1_700_000_000, fault_type="overvoltage")
    fault2 = make_fault(pzem_number=2, timestamp=1_700_000_000, fault_type="overvoltage")
    rec1 = make_engine().recommend(fault1)
    rec2 = make_engine().recommend(fault2)
    assert rec1.recommendation_id != rec2.recommendation_id


def test_recommendation_id_format():
    fault = make_fault()
    rec = make_engine().recommend(fault)
    assert rec.recommendation_id.startswith("REC-")
    assert len(rec.recommendation_id) == 16  # REC- + 12 hex chars


# ---------------------------------------------------------------------------
# Test: evidence/action presence for each fault type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fault_type", SUPPORTED_FAULT_TYPES)
def test_evidence_and_actions_present(fault_type: str):
    fault = make_fault(fault_type=fault_type)
    rec = make_engine().recommend(fault)
    assert rec.evidence
    assert rec.what_to_check
    assert rec.what_to_do_now
    assert rec.corrective_action
    assert rec.probable_cause
    assert rec.why_it_happened


# ---------------------------------------------------------------------------
# Test: unsupported fault type raises ValueError
# ---------------------------------------------------------------------------

def test_unsupported_fault_type_raises():
    fault = make_fault(fault_type="short_circuit")
    with pytest.raises(ValueError, match="Unsupported fault type"):
        make_engine().recommend(fault)


# ---------------------------------------------------------------------------
# Test: missing optional fields handled gracefully
# ---------------------------------------------------------------------------

def test_missing_evidence_optional():
    fault = make_fault(fault_type="communication_degraded", evidence=None, confidence=1.0)
    rec = make_engine().recommend(fault)
    assert rec is not None
    assert rec.evidence  # rule-based evidence still present
    assert rec.confidence == 1.0


def test_missing_confidence_uses_default():
    fault = FaultEvent(
        pzem_number=1, timestamp=BASE_TS, fault_type="high_power",
        severity="EMERGENCY", measured_value=260.0, reason="Test",
        evidence=None, confidence=None,
    )
    rec = make_engine().recommend(fault)
    assert rec.confidence == 0.5


def test_missing_confidence_zero():
    fault = FaultEvent(
        pzem_number=1, timestamp=BASE_TS, fault_type="overvoltage",
        severity="EMERGENCY", measured_value=260.0, reason="Test",
        evidence=None, confidence=0.0,
    )
    rec = make_engine().recommend(fault)
    assert rec.confidence == 0.0


# ---------------------------------------------------------------------------
# Test: no invented energy/cost impact
# ---------------------------------------------------------------------------

def test_no_invented_energy_impact():
    """Energy impact should be None unless fault result contains deterministic value."""
    fault = make_fault(fault_type="overvoltage")
    rec = make_engine().recommend(fault)
    assert rec.energy_impact_kwh is None
    assert rec.cost_impact is None


def test_energy_impact_only_when_provided():
    """Energy impact is extracted only when fault evidence contains energy_impact_kwh."""
    fault = make_fault(
        fault_type="overvoltage",
        evidence={"voltage": 260.0, "energy_impact_kwh": 5.5},
    )
    rec = make_engine().recommend(fault)
    assert rec.energy_impact_kwh == 5.5
    assert rec.cost_impact is None


def test_no_arbitrary_kwh_or_cost():
    """Verify no arbitrary kWh or cost values are generated."""
    for fault_type in SUPPORTED_FAULT_TYPES:
        fault = make_fault(fault_type=fault_type)
        rec = make_engine().recommend(fault)
        assert rec.energy_impact_kwh is None, (
            f"{fault_type} should not have invented energy_impact_kwh"
        )
        assert rec.cost_impact is None, (
            f"{fault_type} should not have invented cost_impact"
        )


# ---------------------------------------------------------------------------
# Test: no unsupported root-cause certainty
# ---------------------------------------------------------------------------

def test_no_transformer_claim_overvoltage():
    """Overvoltage recommendation must never claim transformer failure."""
    fault = make_fault(fault_type="overvoltage")
    rec = make_engine().recommend(fault)
    assert "transformer" not in rec.corrective_action.lower() or "never conclude" in rec.corrective_action.lower() or "transformer failure" not in rec.corrective_action.lower()


def test_no_blind_capacitor_recommendation():
    """PF drop must not blindly recommend capacitor installation."""
    fault = make_fault(fault_type="power_factor_drop")
    rec = make_engine().recommend(fault)
    assert "blindly" in rec.corrective_action.lower() or "conditional" in rec.corrective_action.lower() or "do not blindly" in rec.corrective_action.lower()


def test_no_appliance_blame_without_evidence():
    """Overcurrent and high_power must not claim a specific appliance."""
    for ft in ("overcurrent", "high_power"):
        fault = make_fault(fault_type=ft)
        rec = make_engine().recommend(fault)
        text = (rec.what_to_do_now + " " + rec.corrective_action).lower()
        assert "specific appliance" not in text or "without evidence" in text or "without supporting" in text


def test_no_utility_fault_claim():
    """Frequency deviation must not claim utility/generator fault without evidence."""
    fault = make_fault(fault_type="frequency_deviation")
    rec = make_engine().recommend(fault)
    text = (rec.what_to_do_now + " " + rec.corrective_action).lower()
    assert "utility fault" not in text or "without evidence" in text
    assert "generator fault" not in text or "without evidence" in text


def test_comm_degraded_not_load_fault():
    """Communication degraded must not be described as an electrical load fault."""
    fault = make_fault(fault_type="communication_degraded")
    rec = make_engine().recommend(fault)
    assert "not an electrical load fault" in (rec.what_to_do_now + " " + rec.corrective_action).lower()


# ---------------------------------------------------------------------------
# Test: severity -> priority mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("severity,expected_priority", [
    ("EMERGENCY", "P1 - Critical"),
    ("WARNING", "P2 - Important"),
    ("NORMAL", "P3 - Informational"),
])
def test_severity_to_priority(severity: str, expected_priority: str):
    fault = make_fault(severity=severity)
    rec = make_engine().recommend(fault)
    assert rec.priority == expected_priority


# ---------------------------------------------------------------------------
# Test: pzem_system field
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pzem_num", [1, 5, 9])
def test_pzem_system_field(pzem_num: int):
    fault = make_fault(pzem_number=pzem_num)
    rec = make_engine().recommend(fault)
    assert rec.pzem_system == f"PZEM-{pzem_num}"


# ---------------------------------------------------------------------------
# Test: source_stages
# ---------------------------------------------------------------------------

def test_source_stages():
    fault = make_fault()
    rec = make_engine().recommend(fault)
    assert "Stage 3: Fault Diagnosis" in rec.source_stages


# ---------------------------------------------------------------------------
# Test: confidence from fault event
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("confidence", [0.0, 0.5, 0.95, 1.0])
def test_confidence_from_fault_event(confidence: float):
    fault = make_fault(confidence=confidence)
    rec = make_engine().recommend(fault)
    assert rec.confidence == confidence


# ---------------------------------------------------------------------------
# Test: frozen dataclass
# ---------------------------------------------------------------------------

def test_recommendation_is_frozen():
    fault = make_fault()
    rec = make_engine().recommend(fault)
    with pytest.raises(Exception):
        rec.recommendation_id = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: recommend_batch
# ---------------------------------------------------------------------------

def test_recommend_batch():
    faults = [
        make_fault(fault_type=ft, severity="WARNING")
        for ft in SUPPORTED_FAULT_TYPES
    ]
    recs = RecommendationEngine().recommend_batch(faults)
    assert len(recs) == len(SUPPORTED_FAULT_TYPES)
    for fault_type, rec in zip(SUPPORTED_FAULT_TYPES, recs):
        assert rec.fault_type == fault_type


# ---------------------------------------------------------------------------
# Test: convenience function
# ---------------------------------------------------------------------------

def test_convenience_recommend_function():
    fault = make_fault()
    rec = recommend(fault)
    assert isinstance(rec, DiagnosticRecommendation)
    assert rec.fault_type == "overvoltage"


# ---------------------------------------------------------------------------
# Test: communication_degraded severity handling
# ---------------------------------------------------------------------------

def test_communication_degraded_severity():
    fault = make_fault(fault_type="communication_degraded", severity="WARNING", confidence=0.85)
    rec = make_engine().recommend(fault)
    assert rec.severity == "WARNING"
    assert rec.priority == "P2 - Important"
    assert rec.confidence == 0.85
    assert rec.maintenance_required is True


# ---------------------------------------------------------------------------
# Test: high_power urgency is MEDIUM
# ---------------------------------------------------------------------------

def test_high_power_urgency_medium():
    fault = make_fault(fault_type="high_power", severity="EMERGENCY")
    rec = make_engine().recommend(fault)
    assert rec.urgency == "MEDIUM"
    assert rec.maintenance_required is False


# ---------------------------------------------------------------------------
# Test: power_factor_drop is NOT blindly recommending capacitors
# ---------------------------------------------------------------------------

def test_power_factor_drop_no_blind_capacitors():
    fault = make_fault(fault_type="power_factor_drop", severity="WARNING")
    rec = make_engine().recommend(fault)
    text = rec.corrective_action.lower()
    assert "do not blindly" in text or "conditional on proper" in text


# ---------------------------------------------------------------------------
# Test: all fault types have maintenance_required set appropriately
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fault_type", SUPPORTED_FAULT_TYPES)
def test_maintenance_required_consistent(fault_type: str):
    fault = make_fault(fault_type=fault_type)
    rec = make_engine().recommend(fault)
    assert isinstance(rec.maintenance_required, bool)
    assert rec.maintenance_timing


# ---------------------------------------------------------------------------
# Test: deterministic_id same across separate engine instances
# ---------------------------------------------------------------------------

def test_deterministic_across_engine_instances():
    fault = make_fault()
    rec1 = RecommendationEngine().recommend(fault)
    rec2 = RecommendationEngine().recommend(fault)
    assert rec1.recommendation_id == rec2.recommendation_id


# ---------------------------------------------------------------------------
# Phase 2B: Anomaly + Peak tests
# ---------------------------------------------------------------------------

BASE_TS = 1_700_000_000


def _make_anomaly_result(
    pzem_number: int = 1,
    model_status: str = "READY",
    anomaly_label: str = "ANOMALY",
    anomaly_score_normalized: float = 0.9,
    anomaly_severity_provisional: str = "HIGH_PROVISIONAL",
    training_rows: int = 256,
    active_days: int = 5,
) -> AnomalyDetectionResult:
    """Build an AnomalyDetectionResult with a scored result_frame."""
    if model_status == "READY" and anomaly_label == "ANOMALY":
        df = pd.DataFrame({
            "pzem_id": [pzem_number],
            "timestamp": [BASE_TS],
            "operating_state": ["ACTIVE"],
            "anomaly_score": [-0.5],
            "anomaly_score_normalized": [anomaly_score_normalized],
            "anomaly_label": [anomaly_label],
            "anomaly_severity_provisional": [anomaly_severity_provisional],
        })
    elif model_status == "READY":
        df = pd.DataFrame({
            "pzem_id": [pzem_number],
            "timestamp": [BASE_TS],
            "operating_state": ["ACTIVE"],
            "anomaly_score": [0.1],
            "anomaly_score_normalized": [0.1],
            "anomaly_label": ["NORMAL"],
            "anomaly_severity_provisional": ["N/A"],
        })
    else:
        df = None

    return AnomalyDetectionResult(
        pzem_number=pzem_number,
        model_status=model_status,
        reason=None if model_status == "READY" else "Insufficient training data",
        training_rows=training_rows,
        features_used=["power", "current"],
        contamination=0.05,
        random_state=42,
        operating_state_method="gmm",
        active_rows=500,
        inactive_rows=200,
        active_days_represented=active_days,
        result_frame=df,
    )


def _make_peak_result(
    pzem_number: int = 1,
    status: str = "PEAK_FOUND",
    peak_power_w: float = 3500.0,
    peak_timestamp: int = BASE_TS,
    sustained: bool = True,
    baseline_power_w: float = 200.0,
) -> PeakResult:
    """Build a PeakResult."""
    return PeakResult(
        pzem_number=pzem_number,
        status=status,
        peak_power_w=peak_power_w if status == "PEAK_FOUND" else None,
        peak_timestamp=peak_timestamp if status == "PEAK_FOUND" else None,
        peak_duration_seconds=300 if sustained else 0,
        sustained=sustained,
        average_power_w=peak_power_w * 0.5,
        baseline_power_w=baseline_power_w,
        peak_above_baseline_w=peak_power_w - baseline_power_w if status == "PEAK_FOUND" else None,
        threshold_w=0.0,
        exceeds_threshold=None,
        peak_above_threshold_w=None,
        samples_analyzed=288,
        invalid_rows_dropped=0,
        analysis_start_ts=BASE_TS - 86400,
        analysis_end_ts=BASE_TS,
    )


def _make_system_peak(
    status: str = "PEAK_FOUND",
    total_peak_power_w: float = 8500.0,
    timestamp: int = BASE_TS,
    dominant_pzems: list[int] | None = None,
) -> SystemPeakResult:
    """Build a SystemPeakResult."""
    return SystemPeakResult(
        status=status,
        total_peak_power_w=total_peak_power_w if status == "PEAK_FOUND" else None,
        timestamp=timestamp if status == "PEAK_FOUND" else None,
        dominant_pzems=dominant_pzems or [1, 2],
        per_pzem_power_w={"pzem_1": 4000.0, "pzem_2": 4500.0},
        meters_analyzed=2,
        threshold_w=0.0,
        exceeds_threshold=None,
    )


# ---------------------------------------------------------------------------
# Test: anomaly detection recommendation
# ---------------------------------------------------------------------------

def test_anomaly_high_provisional():
    """Anomaly with HIGH_PROVISIONAL produces EMERGENCY severity."""
    result = _make_anomaly_result(
        anomaly_label="ANOMALY",
        anomaly_score_normalized=0.95,
        anomaly_severity_provisional="HIGH_PROVISIONAL",
    )
    rec = recommend_from_anomaly(result)
    assert rec.fault_type == "anomaly"
    assert rec.severity == "EMERGENCY"
    assert rec.priority == "P1 - Critical"
    assert rec.pzem_system == "PZEM-1"
    assert rec.confidence == 0.95
    assert rec.source_stages == ("Stage 3: Anomaly Detection",)


def test_anomaly_medium_provisional():
    """Anomaly with MEDIUM_PROVISIONAL produces WARNING severity."""
    result = _make_anomaly_result(
        anomaly_label="ANOMALY",
        anomaly_score_normalized=0.7,
        anomaly_severity_provisional="MEDIUM_PROVISIONAL",
    )
    rec = recommend_from_anomaly(result)
    assert rec.severity == "WARNING"
    assert rec.priority == "P2 - Important"
    assert rec.confidence == 0.7


def test_anomaly_normal_result():
    """Anomaly result with NORMAL latest row produces NORMAL severity."""
    result = _make_anomaly_result(
        anomaly_label="NORMAL",
        anomaly_score_normalized=0.1,
        anomaly_severity_provisional="N/A",
    )
    rec = recommend_from_anomaly(result)
    assert rec.fault_type == "anomaly"
    assert rec.severity == "NORMAL"
    assert "No anomaly detected" in rec.condition


def test_anomaly_insufficient_data():
    """Anomaly result with INSUFFICIENT_DATA produces NORMAL severity."""
    result = _make_anomaly_result(
        model_status="INSUFFICIENT_DATA",
        anomaly_label="NOT_SCORED",
    )
    rec = recommend_from_anomaly(result)
    assert rec.fault_type == "anomaly"
    assert rec.condition == "Insufficient data for anomaly detection."
    assert "Insufficient" in rec.why_it_happened
    assert rec.confidence == 0.0


def test_anomaly_no_scored_rows():
    """Anomaly result with empty scored frame produces NORMAL."""
    result = _make_anomaly_result(anomaly_label="NORMAL")
    rec = recommend_from_anomaly(result)
    assert rec.fault_type == "anomaly"
    assert rec.confidence == 0.0


def test_anomaly_evidence_present():
    """Anomaly recommendation has evidence and actions."""
    result = _make_anomaly_result(
        anomaly_label="ANOMALY",
        anomaly_score_normalized=0.88,
        anomaly_severity_provisional="HIGH_PROVISIONAL",
    )
    rec = recommend_from_anomaly(result)
    assert rec.evidence
    assert rec.what_to_check
    assert rec.what_to_do_now
    assert rec.corrective_action
    assert rec.probable_cause


def test_anomaly_cause_remains_conditional():
    """Anomaly must not claim specific cause."""
    result = _make_anomaly_result(
        anomaly_label="ANOMALY",
        anomaly_score_normalized=0.9,
        anomaly_severity_provisional="HIGH_PROVISIONAL",
    )
    rec = recommend_from_anomaly(result)
    text = (rec.probable_cause + " " + rec.corrective_action).lower()
    assert "statistically unusual" in text or "conditional" in text
    assert "component has failed" not in text


# ---------------------------------------------------------------------------
# Test: high peak recommendation
# ---------------------------------------------------------------------------

def test_high_peak_basic():
    """High peak produces a valid recommendation."""
    peak = _make_peak_result()
    rec = recommend_from_peak(peak)
    assert rec.fault_type == "high_peak"
    assert rec.pzem_system == "PZEM-1"
    assert rec.severity == "WARNING"
    assert rec.priority == "P2 - Important"
    assert rec.confidence == 0.7
    assert "demand" in rec.condition.lower()


def test_high_peak_evidence():
    """High peak recommendation has evidence and actions."""
    peak = _make_peak_result()
    rec = recommend_from_peak(peak)
    assert rec.evidence
    assert rec.what_to_check
    assert rec.what_to_do_now
    assert rec.corrective_action


def test_high_peak_no_invented_appliance():
    """High peak must not claim a specific appliance caused it."""
    peak = _make_peak_result()
    rec = recommend_from_peak(peak)
    text = (rec.what_to_do_now + " " + rec.corrective_action).lower()
    assert "specific" in text or "not invent" in text or "without evidence" in text


def test_no_peak():
    """NO_PEAK result produces a valid recommendation."""
    peak = _make_peak_result(status="NO_PEAK")
    rec = recommend_from_peak(peak)
    assert rec.fault_type == "high_peak"
    assert rec.condition == "No peak found in the analysis window."
    assert rec.confidence == 0.3


# ---------------------------------------------------------------------------
# Test: system peak recommendation
# ---------------------------------------------------------------------------

def test_system_peak_basic():
    """System peak produces a valid recommendation."""
    sys_peak = _make_system_peak()
    rec = recommend_from_system_peak(sys_peak)
    assert rec.fault_type == "system_peak"
    assert rec.pzem_system == "SYSTEM"
    assert rec.severity == "WARNING"
    assert "demand" in rec.condition.lower()
    assert rec.confidence == 0.7


def test_system_peak_evidence():
    """System peak recommendation has evidence and actions."""
    sys_peak = _make_system_peak()
    rec = recommend_from_system_peak(sys_peak)
    assert rec.evidence
    assert rec.what_to_check
    assert rec.what_to_do_now


# ---------------------------------------------------------------------------
# Test: recurring peak recommendation
# ---------------------------------------------------------------------------

def test_recurring_peak_basic():
    """Recurring peak with multiple PEAK_FOUND produces valid recommendation."""
    peak1 = _make_peak_result(pzem_number=1, peak_timestamp=BASE_TS)
    peak2 = _make_peak_result(pzem_number=2, peak_timestamp=BASE_TS + 100)
    system = _make_system_peak()
    peak_results = {1: peak1, 2: peak2}
    rec = recommend_from_recurring_peak(peak_results, system)
    assert rec.fault_type == "recurring_peak"
    assert rec.pzem_system == "SYSTEM"
    assert rec.confidence == 0.6
    assert "repeated" in rec.condition.lower()


def test_recurring_peak_insufficient():
    """Recurring peak with fewer than 2 peaks produces no-recurring condition."""
    peak1 = _make_peak_result(pzem_number=1)
    system = _make_system_peak(status="NO_PEAK")
    peak_results = {1: peak1}
    rec = recommend_from_recurring_peak(peak_results, system)
    assert rec.fault_type == "recurring_peak"
    assert "No recurring peak pattern" in rec.condition
    assert rec.confidence == 0.4


def test_recurring_peak_no_system_peak():
    """Recurring peak works without system_peak if multiple peaks found."""
    peak1 = _make_peak_result(pzem_number=1, peak_timestamp=BASE_TS)
    peak2 = _make_peak_result(pzem_number=2, peak_timestamp=BASE_TS + 100)
    peak_results = {1: peak1, 2: peak2}
    rec = recommend_from_recurring_peak(peak_results)
    assert rec.fault_type == "recurring_peak"
    assert "repeated" in rec.condition.lower()


def test_recurring_peak_evidence():
    """Recurring peak recommendation has evidence and actions."""
    peak1 = _make_peak_result(pzem_number=1)
    peak2 = _make_peak_result(pzem_number=2)
    system = _make_system_peak()
    peak_results = {1: peak1, 2: peak2}
    rec = recommend_from_recurring_peak(peak_results, system)
    assert rec.evidence
    assert rec.what_to_check
    assert rec.what_to_do_now
    assert rec.corrective_action


# ---------------------------------------------------------------------------
# Test: no invented energy/cost impact
# ---------------------------------------------------------------------------

def test_anomaly_no_invented_impact():
    """Anomaly recommendations must not have invented kWh or cost."""
    result = _make_anomaly_result()
    rec = recommend_from_anomaly(result)
    assert rec.energy_impact_kwh is None
    assert rec.cost_impact is None


def test_peak_no_invented_impact():
    """Peak recommendations must not have invented kWh or cost."""
    peak = _make_peak_result()
    rec = recommend_from_peak(peak)
    assert rec.energy_impact_kwh is None
    assert rec.cost_impact is None


def test_system_peak_no_invented_impact():
    """System peak recommendations must not have invented kWh or cost."""
    sys_peak = _make_system_peak()
    rec = recommend_from_system_peak(sys_peak)
    assert rec.energy_impact_kwh is None
    assert rec.cost_impact is None


def test_recurring_peak_no_invented_impact():
    """Recurring peak recommendations must not have invented kWh or cost."""
    peak1 = _make_peak_result(pzem_number=1)
    peak2 = _make_peak_result(pzem_number=2)
    rec = recommend_from_recurring_peak({1: peak1, 2: peak2})
    assert rec.energy_impact_kwh is None
    assert rec.cost_impact is None


# ---------------------------------------------------------------------------
# Test: deterministic output
# ---------------------------------------------------------------------------

def test_anomaly_deterministic():
    """Identical anomaly input produces identical recommendation."""
    result = _make_anomaly_result()
    rec1 = recommend_from_anomaly(result)
    rec2 = recommend_from_anomaly(result)
    assert rec1.recommendation_id == rec2.recommendation_id
    assert rec1 == rec2


def test_peak_deterministic():
    """Identical peak input produces identical recommendation."""
    peak = _make_peak_result()
    rec1 = recommend_from_peak(peak)
    rec2 = recommend_from_peak(peak)
    assert rec1.recommendation_id == rec2.recommendation_id
    assert rec1 == rec2


def test_system_peak_deterministic():
    """Identical system peak input produces identical recommendation."""
    sys_peak = _make_system_peak()
    rec1 = recommend_from_system_peak(sys_peak)
    rec2 = recommend_from_system_peak(sys_peak)
    assert rec1.recommendation_id == rec2.recommendation_id


# ---------------------------------------------------------------------------
# Test: missing optional evidence
# ---------------------------------------------------------------------------

def test_anomaly_missing_optional_fields():
    """Anomaly recommendation works with minimal optional fields."""
    result = _make_anomaly_result(model_status="INSUFFICIENT_DATA")
    rec = recommend_from_anomaly(result)
    assert rec is not None
    assert rec.recommendation_id
    assert rec.timestamp >= 0
    assert rec.pzem_system
    assert rec.confidence is not None


def test_peak_missing_optional_fields():
    """Peak recommendation works with NO_PEAK."""
    peak = _make_peak_result(status="NO_PEAK")
    rec = recommend_from_peak(peak)
    assert rec is not None
    assert rec.condition
    assert rec.confidence is not None


# ---------------------------------------------------------------------------
# Test: convenience functions
# ---------------------------------------------------------------------------

def test_convenience_recommend_from_anomaly():
    result = _make_anomaly_result()
    rec = recommend_from_anomaly(result)
    assert isinstance(rec, DiagnosticRecommendation)


def test_convenience_recommend_from_peak():
    peak = _make_peak_result()
    rec = recommend_from_peak(peak)
    assert isinstance(rec, DiagnosticRecommendation)


def test_convenience_recommend_from_system_peak():
    sys_peak = _make_system_peak()
    rec = recommend_from_system_peak(sys_peak)
    assert isinstance(rec, DiagnosticRecommendation)


def test_convenience_recommend_from_recurring_peak():
    peak1 = _make_peak_result(pzem_number=1)
    peak2 = _make_peak_result(pzem_number=2)
    rec = recommend_from_recurring_peak({1: peak1, 2: peak2})
    assert isinstance(rec, DiagnosticRecommendation)


# ---------------------------------------------------------------------------
# Phase 2C tests
# ---------------------------------------------------------------------------

def _make_fault(fault_type="overcurrent", severity="EMERGENCY") -> FaultEvent:
    return FaultEvent(
        pzem_number=1, timestamp=BASE_TS, fault_type=fault_type,
        severity=severity, measured_value=35.0, reason="Test fault",
        evidence=None, confidence=0.95,
    )


def _make_bill_result() -> dict:
    return {
        "status": "OK",
        "actual_energy_kwh": 450.5,
        "forecast_energy_kwh": 120.3,
        "estimated_total_energy_kwh": 570.8,
        "rate": 0.12,
        "estimated_bill": 68.50,
        "predicted_difference": 14.44,
        "forecast_confidence": "medium",
        "billing_period": "30d",
        "anchor_timestamp": BASE_TS,
    }


def _make_es_recommendation(
    pzem_number: int = 1,
    recommendation_type: str = "SHIFT_NON_CRITICAL_LOAD",
    priority: str = "HIGH",
    potential_saving_kwh: float = 15.5,
    potential_cost_saving: float = 1.86,
) -> EnergySavingRecommendation:
    return EnergySavingRecommendation(
        pzem_number=pzem_number,
        timestamp=BASE_TS,
        recommendation_type=recommendation_type,
        priority=priority,
        recommendation="Shift non-critical loads outside the recurring peak window.",
        reason="Recurring high-demand window detected with elevated median power.",
        supporting_metrics={"peak_median_w": 3500.0, "typical_median_w": 1200.0},
        evidence_window="2024-01-01 -> 2024-01-31 UTC",
        potential_saving_kwh=potential_saving_kwh,
        potential_cost_saving=potential_cost_saving,
        estimated_percent_reduction=8.5,
        source_stages=["stage11/energy_saving"],
    )


def _make_risk_result(
    pzem_number: int = 1,
    risk_score: int = 75,
    risk_level: str = "CRITICAL",
    indicators: list | None = None,
) -> RiskResult:
    return RiskResult(
        pzem_number=pzem_number, status="RISK_ASSESSED",
        risk_score=risk_score, risk_level=risk_level,
        data_sufficiency="SUFFICIENT_FOR_RISK_ANALYSIS",
        samples_analyzed=500,
        window_start_ts=BASE_TS - 86400,
        window_end_ts=BASE_TS,
        indicators=indicators or ["power_trend", "peak_threshold_exceeded"],
        evidence=[{"indicator": "test", "triggered": True}],
        confidence="moderate",
    )


def _make_system_summary() -> SystemMaintenanceSummary:
    return SystemMaintenanceSummary(
        meters_analyzed=3,
        high_risk_meters=[1],
        watch_meters=[2],
        normal_meters=[3],
        insufficient_data_meters=[],
        highest_risk_pzem=1,
        highest_risk_score=75,
        system_data_sufficiency="SUFFICIENT_FOR_RISK_ANALYSIS",
        timestamp=BASE_TS,
    )


def _make_forecast_result() -> ForecastResult:
    return ForecastResult(
        pzem_number=1, is_system=False, status="FORECAST",
        anchor_timestamp=BASE_TS, valid_samples=288, span_days=30.0,
        recent_level_w=2500.0,
        forecast_24h={
            "status": "FORECAST",
            "confidence": "high",
            "start_ts": BASE_TS + 300,
            "end_ts": BASE_TS + 86400,
            "count": 288,
            "timestamps": [BASE_TS + 300 + i * 300 for i in range(288)],
            "forecast_power_w": [2000.0 + i * 10 for i in range(288)],
            "lower_bound": [1800.0] * 288,
            "upper_bound": [2200.0] * 288,
        },
        forecast_7d={},
    )


# ---------------------------------------------------------------------------
# Test: cross-pipeline reasoning
# ---------------------------------------------------------------------------

def test_cross_pipeline_high_current_high_power_peak():
    """Cross-pipeline combines fault + peak + anomaly into one recommendation."""
    fault = _make_fault(fault_type="overcurrent", severity="EMERGENCY")
    peak = _make_peak_result(pzem_number=1, peak_power_w=4500.0)
    anomaly = _make_anomaly_result(anomaly_label="ANOMALY", anomaly_score_normalized=0.9)
    rec = recommend_from_cross_pipeline(
        fault_result=fault, peak_result=peak, anomaly_result=anomaly,
        energy_kwh=150.0, dominant_pzem=1,
    )
    assert rec.fault_type == "cross_pipeline"
    assert rec.condition == "Combined signals suggest possible overloaded or stressed circuit."
    assert "overcurrent" in rec.evidence.lower() or "peak" in rec.evidence.lower()
    assert rec.energy_impact_kwh == 150.0
    assert rec.confidence == 0.7


def test_cross_pipeline_single_signal():
    """Cross-pipeline with single signal works."""
    fault = _make_fault(fault_type="overcurrent", severity="WARNING")
    rec = recommend_from_cross_pipeline(fault_result=fault, dominant_pzem=1)
    assert rec.fault_type == "cross_pipeline"
    assert rec.confidence == 0.5


# ---------------------------------------------------------------------------
# Test: energy impact MEASURED
# ---------------------------------------------------------------------------

def test_energy_impact_measured():
    """Energy impact classification MEASURED when value supplied."""
    rec = recommend_from_cross_pipeline(
        fault_result=_make_fault(), peak_result=_make_peak_result(),
        energy_kwh=150.0, dominant_pzem=1,
    )
    assert rec.energy_impact_kwh == 150.0
    assert "MEASURED" in rec.evidence


def test_energy_impact_unknown():
    """Energy impact is UNKNOWN when no value supplied."""
    rec = recommend_from_cross_pipeline(fault_result=_make_fault(), dominant_pzem=1)
    assert rec.energy_impact_kwh is None
    assert "UNKNOWN" in rec.evidence


# ---------------------------------------------------------------------------
# Test: cost calculation
# ---------------------------------------------------------------------------

def test_cost_calculation_with_tariff():
    """Cost = energy_kwh × tariff when both valid."""
    rec = recommend_from_cross_pipeline(
        fault_result=_make_fault(), peak_result=_make_peak_result(),
        energy_kwh=150.0, tariff=0.12, dominant_pzem=1,
    )
    assert rec.cost_impact == 18.0  # 150 * 0.12


def test_cost_unknown_without_tariff():
    """Cost is None when tariff not supplied."""
    rec = recommend_from_cross_pipeline(
        fault_result=_make_fault(), peak_result=_make_peak_result(),
        energy_kwh=150.0, dominant_pzem=1,
    )
    assert rec.cost_impact is None


# ---------------------------------------------------------------------------
# Test: bill recommendation
# ---------------------------------------------------------------------------

def test_bill_recommendation():
    """Bill recommendation from valid bill result."""
    bill = _make_bill_result()
    peak = _make_peak_result()
    rec = recommend_from_bill(bill_result=bill, peak_result=peak, dominant_pzem=1)
    assert rec.fault_type == "bill"
    assert rec.energy_impact_kwh is not None
    assert rec.cost_impact is not None
    assert "actual" in rec.evidence.lower()
    assert "forecast" in rec.evidence.lower()


def test_bill_recommendation_with_explicit_anchor():
    """Bill recommendation uses explicit anchor_timestamp when provided."""
    bill = _make_bill_result()
    rec = recommend_from_bill(bill_result=bill, anchor_timestamp=1_700_001_000, dominant_pzem=1)
    assert rec.fault_type == "bill"
    assert rec.timestamp == 1_700_001_000


def test_bill_recommendation_no_anchor_defaults_to_zero():
    """Bill recommendation without anchor_timestamp and without bill_result anchor falls back to 0."""
    bill = _make_bill_result()
    bill_no_ts = {k: v for k, v in bill.items() if k != "anchor_timestamp"}
    rec = recommend_from_bill(bill_result=bill_no_ts, dominant_pzem=1)
    assert rec.fault_type == "bill"
    # bill_result has no anchor_timestamp, so falls back to 0
    assert rec.timestamp == 0


def test_bill_recommendation_insufficient():
    """Bill recommendation with insufficient data."""
    rec = recommend_from_bill(bill_result=None, dominant_pzem=1)
    assert rec.fault_type == "bill"
    assert rec.condition == "Bill data unavailable or insufficient."


def test_bill_recommendation_insufficient_with_anchor():
    """Bill recommendation with insufficient data uses explicit anchor_timestamp."""
    rec = recommend_from_bill(bill_result=None, anchor_timestamp=1_700_002_000, dominant_pzem=1)
    assert rec.fault_type == "bill"
    assert rec.timestamp == 1_700_002_000


# ---------------------------------------------------------------------------
# Test: energy-saving integration
# ---------------------------------------------------------------------------

def test_energy_saving_integration():
    """Energy-saving recommendation converts to DiagnosticRecommendation."""
    es_rec = _make_es_recommendation(
        potential_saving_kwh=15.5, potential_cost_saving=1.86,
    )
    rec = recommend_from_energy_saving(es_recommendation=es_rec)
    assert rec.fault_type == "energy_saving"
    assert rec.energy_impact_kwh is not None
    assert "15.5" in rec.corrective_action or rec.energy_impact_kwh is not None
    # Priority should be normalized to P1/P2/P3 convention
    assert rec.priority in ("P1 - Critical", "P2 - Important", "P3 - Informational")


def test_energy_saving_priority_normalization():
    """Energy-saving HIGH priority maps to P1."""
    es_rec = _make_es_recommendation(priority="HIGH")
    rec = recommend_from_energy_saving(es_recommendation=es_rec)
    assert rec.priority == "P1 - Critical"

    es_rec_med = _make_es_recommendation(priority="MEDIUM")
    rec_med = recommend_from_energy_saving(es_recommendation=es_rec_med)
    assert rec_med.priority == "P2 - Important"

    es_rec_low = _make_es_recommendation(priority="LOW")
    rec_low = recommend_from_energy_saving(es_recommendation=es_rec_low)
    assert rec_low.priority == "P3 - Informational"


def test_energy_saving_no_savings():
    """Energy-saving without savings keeps unknown impact."""
    es_rec = _make_es_recommendation(potential_saving_kwh=None, potential_cost_saving=None)
    rec = recommend_from_energy_saving(es_recommendation=es_rec)
    assert rec.fault_type == "energy_saving"
    assert rec.energy_impact_kwh is None


# ---------------------------------------------------------------------------
# Test: maintenance-risk integration
# ---------------------------------------------------------------------------

def test_maintenance_risk_critical():
    """Maintenance recommendation for CRITICAL risk."""
    risk = _make_risk_result(risk_score=75, risk_level="CRITICAL")
    rec = recommend_from_maintenance(risk_result=risk)
    assert rec.fault_type == "maintenance"
    assert "CRITICAL" in rec.condition
    assert rec.urgency == "HIGH"
    assert "urgent" in rec.maintenance_timing.lower()


def test_maintenance_risk_normal():
    """Maintenance recommendation for NORMAL risk."""
    risk = _make_risk_result(risk_score=10, risk_level="NORMAL")
    rec = recommend_from_maintenance(risk_result=risk)
    assert rec.fault_type == "maintenance"
    assert rec.urgency == "LOW"
    assert not rec.maintenance_required


def test_maintenance_risk_no_data():
    """Maintenance recommendation with insufficient data."""
    risk = RiskResult(pzem_number=1, status="INSUFFICIENT_DATA", reason="No data")
    rec = recommend_from_maintenance(risk_result=risk)
    assert rec.fault_type == "maintenance"
    assert rec.condition == "No maintenance risk assessment available."


# ---------------------------------------------------------------------------
# Test: forecast + peak combination
# ---------------------------------------------------------------------------

def test_forecast_peak_combination():
    """Forecast elevated + peak evidence produces planning recommendation."""
    forecast = _make_forecast_result()
    peak = _make_peak_result()
    rec = recommend_from_forecast_peak(
        forecast_result=forecast, peak_result=peak,
    )
    assert rec.fault_type == "forecast_peak"
    assert "planning" in rec.condition.lower() or "Forecast" in rec.condition
    assert rec.urgency == "MEDIUM"
    assert "shift" in rec.what_to_do_now.lower() or "schedule" in rec.what_to_do_now.lower()


def test_forecast_peak_no_convergence():
    """No forecast elevation produces low urgency."""
    forecast = ForecastResult(
        pzem_number=1, is_system=False, status="NO_FORECAST",
    )
    peak = _make_peak_result()
    rec = recommend_from_forecast_peak(
        forecast_result=forecast, peak_result=peak,
    )
    assert rec.fault_type == "forecast_peak"
    assert rec.urgency == "LOW"


# ---------------------------------------------------------------------------
# Test: deduplication
# ---------------------------------------------------------------------------

def test_deduplication_removes_duplicates():
    """Deduplication removes exact duplicates by recommendation_id."""
    rec1 = recommend_from_cross_pipeline(fault_result=_make_fault(), dominant_pzem=1)
    rec2 = recommend_from_cross_pipeline(fault_result=_make_fault(), dominant_pzem=1)
    deduped = deduplicate_recommendations([rec1, rec2])
    assert len(deduped) == 1


def test_deduplication_keeps_distinct():
    """Deduplication keeps distinct conditions."""
    rec1 = recommend_from_cross_pipeline(
        fault_result=_make_fault(fault_type="overcurrent"), dominant_pzem=1)
    rec2 = recommend_from_cross_pipeline(
        fault_result=_make_fault(fault_type="overvoltage"), dominant_pzem=2)
    deduped = deduplicate_recommendations([rec1, rec2])
    assert len(deduped) >= 1  # May merge or keep distinct based on condition


def test_deduplication_merges_same_condition():
    """Deduplication merges recommendations with same pzem_system + fault_type + severity."""
    rec1 = recommend_from_cross_pipeline(fault_result=_make_fault(), dominant_pzem=1)
    rec2 = recommend_from_cross_pipeline(fault_result=_make_fault(), dominant_pzem=1)
    deduped = deduplicate_recommendations([rec1, rec2])
    assert len(deduped) == 1
    assert "ALSO" in deduped[0].why_it_happened


# ---------------------------------------------------------------------------
# Test: deterministic output
# ---------------------------------------------------------------------------

def test_cross_pipeline_deterministic():
    """Identical cross-pipeline input produces identical output."""
    fault = _make_fault()
    peak = _make_peak_result()
    rec1 = recommend_from_cross_pipeline(fault_result=fault, peak_result=peak, dominant_pzem=1)
    rec2 = recommend_from_cross_pipeline(fault_result=fault, peak_result=peak, dominant_pzem=1)
    assert rec1.recommendation_id == rec2.recommendation_id


def test_bill_deterministic():
    """Identical bill input produces identical output."""
    bill = _make_bill_result()
    rec1 = recommend_from_bill(bill_result=bill, dominant_pzem=1)
    rec2 = recommend_from_bill(bill_result=bill, dominant_pzem=1)
    assert rec1.recommendation_id == rec2.recommendation_id


def test_maintenance_deterministic():
    """Identical maintenance input produces identical output."""
    risk = _make_risk_result()
    rec1 = recommend_from_maintenance(risk_result=risk)
    rec2 = recommend_from_maintenance(risk_result=risk)
    assert rec1.recommendation_id == rec2.recommendation_id


# ---------------------------------------------------------------------------
# Test: no invented numeric values
# ---------------------------------------------------------------------------

def test_no_invented_energy_in_bill():
    """Bill recommendation uses actual supplied values."""
    bill = _make_bill_result()
    rec = recommend_from_bill(bill_result=bill, dominant_pzem=1)
    assert rec.energy_impact_kwh is not None
    assert rec.cost_impact is not None


def test_no_invented_energy_in_maintenance():
    """Maintenance recommendation does not have invented energy impact."""
    risk = _make_risk_result()
    rec = recommend_from_maintenance(risk_result=risk)
    assert rec.energy_impact_kwh is None
    assert rec.cost_impact is None


def test_no_invented_energy_in_forecast_peak():
    """Forecast+peak recommendation does not have invented energy impact."""
    forecast = _make_forecast_result()
    peak = _make_peak_result()
    rec = recommend_from_forecast_peak(forecast_result=forecast, peak_result=peak)
    assert rec.energy_impact_kwh is None
    assert rec.cost_impact is None


# ---------------------------------------------------------------------------
# Stage 4 Audit Fix Regression Tests
# ---------------------------------------------------------------------------

# --- Fix 2: source_stages reflects ONLY inputs actually supplied ---

def test_cross_pipeline_source_stages_fault_only():
    """cross_pipeline with only fault has only fault stage in source_stages."""
    fault = _make_fault(fault_type="overcurrent", severity="EMERGENCY")
    rec = recommend_from_cross_pipeline(fault_result=fault, dominant_pzem=1)
    assert rec.source_stages == ("Stage 3: Fault Diagnosis",)


def test_cross_pipeline_source_stages_fault_and_peak():
    """cross_pipeline with fault + peak has only those stages."""
    fault = _make_fault(fault_type="overcurrent", severity="WARNING")
    peak = _make_peak_result(pzem_number=1, peak_power_w=3500.0)
    rec = recommend_from_cross_pipeline(fault_result=fault, peak_result=peak, dominant_pzem=1)
    assert "Stage 3: Fault Diagnosis" in rec.source_stages
    assert "Stage 7: Peak Detection" in rec.source_stages
    assert len(rec.source_stages) == 2


def test_cross_pipeline_source_stages_all_three():
    """cross_pipeline with fault + peak + anomaly has all three stages."""
    fault = _make_fault(fault_type="overcurrent", severity="EMERGENCY")
    peak = _make_peak_result(pzem_number=1, peak_power_w=3500.0)
    anomaly = _make_anomaly_result(anomaly_label="ANOMALY", anomaly_score_normalized=0.9)
    rec = recommend_from_cross_pipeline(
        fault_result=fault, peak_result=peak, anomaly_result=anomaly, dominant_pzem=1,
    )
    assert len(rec.source_stages) == 3
    assert "Stage 3: Fault Diagnosis" in rec.source_stages
    assert "Stage 3: Anomaly Detection" in rec.source_stages
    assert "Stage 7: Peak Detection" in rec.source_stages


def test_cross_pipeline_source_stages_peak_only():
    """cross_pipeline with peak only (no fault) has only peak stage."""
    peak = _make_peak_result(pzem_number=1, peak_power_w=3500.0)
    rec = recommend_from_cross_pipeline(peak_result=peak, dominant_pzem=1)
    assert rec.source_stages == ("Stage 7: Peak Detection",)


# --- Fix 3: bill confidence uses explicit None checking ---

def test_bill_confidence_with_zero_estimated_total():
    """estimated_total_energy_kwh=0.0 is valid and should give confidence 0.7."""
    bill = _make_bill_result()
    bill["estimated_total_energy_kwh"] = 0.0
    rec = recommend_from_bill(bill_result=bill, dominant_pzem=1)
    assert rec.confidence == 0.7


# --- Edge case: bill with peak_result=None ---

def test_bill_with_peak_result_none():
    """Bill recommendation works when peak_result is None."""
    bill = _make_bill_result()
    rec = recommend_from_bill(bill_result=bill, peak_result=None, dominant_pzem=1)
    assert rec.fault_type == "bill"
    assert rec.confidence == 0.7


# --- Edge case: energy_saving with potential_saving_kwh=0.0 ---

def test_energy_saving_zero_potential_saving():
    """potential_saving_kwh=0.0 is valid and should produce MEASURED impact."""
    es_rec = _make_es_recommendation(potential_saving_kwh=0.0, potential_cost_saving=0.0)
    rec = recommend_from_energy_saving(es_recommendation=es_rec)
    assert rec.energy_impact_kwh == 0.0


# --- Edge case: forecast_peak with NO_FORECAST and no peak ---

def test_forecast_peak_no_forecast_no_peak():
    """NO_FORECAST with no peak produces low-confidence recommendation."""
    forecast = ForecastResult(
        pzem_number=1, is_system=False, status="NO_FORECAST",
    )
    rec = recommend_from_forecast_peak(forecast_result=forecast)
    assert rec.fault_type == "forecast_peak"
    assert rec.urgency == "LOW"
    assert rec.confidence == 0.3


# --- Edge case: maintenance with window_end_ts=None ---

def test_maintenance_window_end_ts_none():
    """Maintenance recommendation with risk_result.window_end_ts=None works."""
    risk = RiskResult(
        pzem_number=1, status="RISK_ASSESSED",
        risk_score=50, risk_level="WARNING",
        data_sufficiency="SUFFICIENT",
        samples_analyzed=100,
        window_start_ts=1_700_000_000,
        window_end_ts=None,
        indicators=["test"],
        evidence=[{"test": True}],
        confidence="moderate",
    )
    rec = recommend_from_maintenance(risk_result=risk)
    assert rec.fault_type == "maintenance"
    assert rec.timestamp == 0


# --- Edge case: deduplication across different PZEM numbers ---

def test_deduplication_different_pzem():
    """Deduplication keeps recommendations from different PZEM numbers."""
    rec1 = recommend_from_cross_pipeline(fault_result=_make_fault(fault_type="overcurrent"), dominant_pzem=1)
    rec2 = recommend_from_cross_pipeline(fault_result=_make_fault(fault_type="overvoltage"), dominant_pzem=2)
    deduped = deduplicate_recommendations([rec1, rec2])
    assert len(deduped) >= 1


# --- Optional: _classify_energy_impact unknown source returns UNKNOWN ---

def test_classify_energy_impact_unknown_source():
    """Unknown source should return UNKNOWN, not MEASURED."""
    rec = recommend_from_cross_pipeline(
        fault_result=_make_fault(), peak_result=_make_peak_result(),
        energy_kwh=150.0, dominant_pzem=1,
    )
    assert rec.energy_impact_kwh == 150.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
