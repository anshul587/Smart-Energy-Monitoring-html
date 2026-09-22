"""
tests/test_phase3d.py — Stage 5D Phase 3D: Final Validation / Safety Gate.

Verifies that _validate_response(), _safe_fallback(), and _apply_phase3d()
correctly validate the final user-facing response against authoritative evidence.

Categories:
1. Valid simple reading
2. Valid diagnostic response
3. Unsupported number
4. Unsupported cause
5. Unsupported action
6. Wrong confidence
7. Historical/live mismatch
8. System/PZEM attribution
9. Conflicting evidence
10. Insufficient evidence
11. Partial evidence
12. Multiple recommendations
13. None handling
14. Zero handling
15. Cost/energy impact validation
16. Maintenance timing validation
17. Duplicate uncertainty repair
18. Debug-tag rejection/repair
19. Simple-question conciseness
20. Safe fallback
21. Deterministic behavior
22. LLM response validation
23. Existing Phase 3C response accepted unchanged
24. PZEM scope validation
25. Timestamp/time-window validation
"""
from __future__ import annotations
import pytest
from ai.ask_bob import (
    _validate_response, _safe_fallback, _apply_phase3d,
    _compose_energy, _compose_combined_evidence,
    _correlate_evidence, _reason_response, ask_bob,
)
from ai import bob_tools

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _meter(n, online=True, power=None, energy=None, voltage=None, current=None,
           freq=None, pf=None):
    return {"pzem_number": n, "online": online, "voltage": voltage,
            "current": current, "power": power if online else None,
            "energy": energy, "power_factor": pf if pf else 0.9,
            "frequency": freq if online else None,
            "last_seen": 1700000000000, "age_ms": 1000}

DIAGNOSTIC_RECS = [
    {"pzem_number": 1, "timestamp": 1700000000000, "fault_type": "overvoltage",
     "severity": "WARNING", "priority": "P1 - Critical",
     "probable_cause": "Possible incoming supply overvoltage.",
     "confidence": 0.85, "what_to_check": "Verify voltage with an independent meter.",
     "what_to_do_now": "Do NOT assume equipment failure.",
     "corrective_action": "Independent voltage verification first.",
     "urgency": "HIGH", "maintenance_required": True,
     "maintenance_timing": "Schedule inspection promptly.",
     "energy_impact_kwh": None, "cost_impact": None,
     "source_stages": ("Stage 3: Fault Diagnosis",)},
    {"pzem_number": 2, "timestamp": 1700000001000, "fault_type": "high_power",
     "severity": "WARNING", "priority": "P2 - High",
     "probable_cause": "Possible overload or excessive connected load.",
     "confidence": 0.72, "what_to_check": "Check active loads and breaker rating.",
     "what_to_do_now": "Reduce or redistribute load if overloaded.",
     "corrective_action": "Reduce or redistribute load if overloaded.",
     "urgency": "MEDIUM", "maintenance_required": False,
     "maintenance_timing": "Monitor.",
     "energy_impact_kwh": None, "cost_impact": None,
     "source_stages": ("Stage 4: Diagnostic & Recommendation",)},
]
DIAGNOSTIC_RECS_EMPTY = []
HISTORICAL_DATA = {
    "status": "OK", "pzem_number": 3,
    "requested_start": 1788998400, "requested_end": 1789084799,
    "available_days": 30.0, "sample_count": 1440,
    "power": {"average": 150.0, "maximum": 450.0, "minimum": 50.0,
              "max_timestamp": 1789040000000},
    "voltage": {"average": 228.5}, "current": {"average": 0.65},
    "energy_consumption": {"consumption_kwh": 10.5},
}
SYSTEM_SUMMARY = {
    "system_status": "online", "total_power_w": 890.0,
    "total_energy_kwh": 45.2, "average_voltage_v": 229.0,
    "active_fault_count": 1,
}


def _make_results(**overrides):
    return {}

def _correlate(results, evidence=None):
    if evidence is None:
        evidence = _compose_combined_evidence("test", results)
    return _correlate_evidence(results, evidence)

def _evidence_for(results, question="test"):
    return _compose_combined_evidence(question, results)

def _render_pieces(results, question="test"):
    from ai.ask_bob import _RENDERERS, _ORDER, _has_data
    pieces = []
    for name in _ORDER:
        if name not in results:
            continue
        rendered = _RENDERERS[name](results[name], question)
        if rendered:
            pieces.append(rendered)
    return pieces


def _compose_for(results, question, correlation=None):
    """Compose a full response for testing."""
    evidence = _compose_combined_evidence(question, results)
    if correlation is None:
        correlation = _correlate(results, evidence)
    pieces = _render_pieces(results, question)
    resp = _reason_response(pieces, evidence, correlation, question)
    return resp, evidence, correlation


# ============================================================================
# 1. VALID SIMPLE READING
# ============================================================================

class TestPhase3DValidSimpleReading:
    """Simple, evidence-grounded responses should pass validation."""

    def test_valid_power_reading(self):
        results = {"get_meter": _meter(2, power=372.0)}
        resp, evidence, corr = _compose_for(results, "PZEM-2 ka power kitna hai?")
        report = _validate_response(resp, "PZEM-2 ka power kitna hai?", results, evidence, corr)
        assert report["status"] == "VALID"

    def test_valid_voltage_reading(self):
        results = {"get_meter": _meter(1, voltage=230.0)}
        resp, evidence, corr = _compose_for(results, "PZEM-1 ka voltage kitna hai?")
        report = _validate_response(resp, "PZEM-1 ka voltage kitna hai?", results, evidence, corr)
        assert report["status"] == "VALID"

    def test_valid_current_reading(self):
        results = {"get_meter": _meter(1, current=1.2)}
        resp, evidence, corr = _compose_for(results, "PZEM-1 ka current kitna hai?")
        report = _validate_response(resp, "PZEM-1 ka current kitna hai?", results, evidence, corr)
        assert report["status"] == "VALID"

    def test_valid_energy_reading(self):
        results = {"get_meter": _meter(1, energy=5.5)}
        resp, evidence, corr = _compose_for(results, "PZEM-1 ki energy kitni hai?")
        report = _validate_response(resp, "PZEM-1 ki energy kitni hai?", results, evidence, corr)
        assert report["status"] == "VALID"


# ============================================================================
# 2. VALID DIAGNOSTIC RESPONSE
# ============================================================================

class TestPhase3DValidDiagnosticResponse:
    """Diagnostic responses with Stage 4 evidence should pass."""

    def test_valid_diagnosis_why(self):
        results = {"get_meter": _meter(1, power=500.0, voltage=245.0),
                   "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        resp, evidence, corr = _compose_for(results, "PZEM-1 mein power high kyu hai?")
        report = _validate_response(resp, "PZEM-1 mein power high kyu hai?", results, evidence, corr)
        assert report["status"] == "VALID"

    def test_valid_diagnosis_action(self):
        results = {"get_meter": _meter(1, power=500.0),
                   "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        resp, evidence, corr = _compose_for(results, "PZEM-1 ka voltage high hai, kya karu?")
        report = _validate_response(resp, "PZEM-1 ka voltage high hai, kya karu?", results, evidence, corr)
        assert report["status"] == "VALID"


# ============================================================================
# 3. UNSUPPORTED NUMBER
# ============================================================================

class TestPhase3DUnsupportedNumber:
    """Response claiming numbers not in evidence must be flagged UNSAFE."""

    def test_power_500_when_evidence_372(self):
        results = {"get_meter": _meter(2, power=372.0)}
        bad_response = "PZEM 2 is online. Power: 500 W."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "PZEM-2 ka power kitna hai?", results, evidence, corr)
        assert report["status"] == "UNSAFE"
        assert any("unsupported_number" in v for v in report["violations"])

    def test_voltage_240_when_evidence_230(self):
        results = {"get_meter": _meter(1, voltage=230.0)}
        bad_response = "Voltage is 240 V."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "PZEM-1 ka voltage kitna hai?", results, evidence, corr)
        assert report["status"] == "UNSAFE"

    def test_kv_claim_without_evidence(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = "Voltage is 11 kV."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "test?", results, evidence, corr)
        assert report["status"] == "UNSAFE"


# ============================================================================
# 4. UNSUPPORTED CAUSE
# ============================================================================

class TestPhase3DUnsupportedCause:
    """Response claiming a cause not in Stage 4 evidence must be UNSAFE."""

    def test_cause_without_diagnosis(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = "Motor overload is causing high power."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "PZEM-1 mein power high kyu hai?", results, evidence, corr)
        assert report["status"] == "UNSAFE"

    def test_no_diagnosis_no_cause(self):
        results = {"get_meter": _meter(1, power=372.0)}
        good_response = "PZEM 1 is online. Power: 372.0 W."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(good_response, "PZEM-1 ka power kitna hai?", results, evidence, corr)
        assert report["status"] == "VALID"

    def test_invented_transformer_cause(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = "Transformer issue causing the fault."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "PZEM-1 mein problem kya hai?", results, evidence, corr)
        assert report["status"] == "UNSAFE"


# ============================================================================
# 5. UNSUPPORTED ACTION
# ============================================================================

class TestPhase3DUnsupportedAction:
    """Response claiming actions not in evidence must be UNSAFE."""

    def test_replace_transformer_without_evidence(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = "Replace the transformer immediately."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "PZEM-1 ka voltage high hai, kya karu?", results, evidence, corr)
        assert report["status"] == "UNSAFE"

    def test_valid_action_from_evidence(self):
        results = {"get_meter": _meter(1, power=500.0),
                   "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        good_response = "PZEM 1 is online. Power: 500.0 W.\n\nVerify voltage with an independent meter."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(good_response, "PZEM-1 ka voltage high hai, kya karu?", results, evidence, corr)
        assert report["status"] == "VALID"


# ============================================================================
# 6. WRONG CONFIDENCE
# ============================================================================

class TestPhase3DWrongConfidence:
    """Response must not upgrade confidence values."""

    def test_confidence_05_not_high(self):
        results = {"get_meter": _meter(1, power=372.0),
                   "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        bad_response = "High confidence diagnosis: overvoltage detected."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "PZEM-1 mein power high kyu hai?", results, evidence, corr)
        assert report["status"] == "REPAIRABLE"


# ============================================================================
# 7. HISTORICAL/LIVE MISMATCH
# ============================================================================

class TestPhase3DHistoricalLiveMismatch:
    """Historical questions must not contain live/current references."""

    def test_historical_contains_live_reference(self):
        results = {"get_historical_analysis": HISTORICAL_DATA}
        bad_response = "PZEM 3 historical analysis: Period: 2026-09-10. Power is online."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "10 September ko PZEM-3 ki power kya thi?", results, evidence, corr)
        assert report["status"] == "REPAIRABLE"
        assert any("historical_live_mix" in v for v in report["violations"])

    def test_historical_correct_response(self):
        results = {"get_historical_analysis": HISTORICAL_DATA}
        good_response = "PZEM 3 historical analysis: Period: 2026-09-10 00:00 UTC to 2026-09-10 23:59 UTC Maximum power: 450.0 W"
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(good_response, "10 September ko PZEM-3 ki maximum power kya thi?", results, evidence, corr)
        assert report["status"] == "VALID"


# ============================================================================
# 8. SYSTEM/PZEM ATTRIBUTION
# ============================================================================

class TestPhase3DSystemPZEMAttribution:
    """System-level evidence must not be attributed to specific PZEMs."""

    def test_system_attributed_to_pzem(self):
        results = {"get_system_summary": SYSTEM_SUMMARY}
        bad_response = "PZEM-3 caused the system peak."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "system peak kyun tha?", results, evidence, corr)
        assert report["status"] == "REPAIRABLE"
        assert any("system_pzem" in v for v in report["violations"])


# ============================================================================
# 9. CONFLICTING EVIDENCE
# ============================================================================

class TestPhase3DConflictingEvidence:
    """Conflicting evidence must not be presented as definitive."""

    def test_conflict_with_definitive_claim(self):
        results = {"get_meter": _meter(1, power=500.0, voltage=248.0),
                   "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        bad_response = "Voltage is definitely 248V and everything is confirmed normal."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "test?", results, evidence, corr)
        assert report["status"] == "REPAIRABLE"


# ============================================================================
# 10. INSUFFICIENT EVIDENCE
# ============================================================================

class TestPhase3DInsufficientEvidence:
    """INSUFFICIENT evidence must not contain definitive causes."""

    def test_insufficient_with_definitive_cause(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = "Motor overload is causing high power."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "PZEM-1 mein power high kyu hai?", results, evidence, corr)
        assert report["status"] == "UNSAFE"

    def test_insufficient_safe_response(self):
        results = {"get_meter": _meter(1, power=372.0)}
        good_response = "PZEM 1 is online. Power: 372.0 W."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(good_response, "PZEM-1 ka power kitna hai?", results, evidence, corr)
        assert report["status"] == "VALID"


# ============================================================================
# 11. PARTIAL EVIDENCE
# ============================================================================

class TestPhase3DPartialEvidence:
    """PARTIAL evidence must not invent causes for why questions."""

    def test_partial_with_cause(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = "The cause is voltage surge."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "PZEM-1 mein power high kyu hai?", results, evidence, corr)
        assert report["status"] == "UNSAFE"


# ============================================================================
# 12. MULTIPLE RECOMMENDATIONS
# ============================================================================

class TestPhase3DMultipleRecommendations:
    """Responses with multiple recommendations should be validated correctly."""

    def test_multiple_recs_valid(self):
        results = {"get_meter": _meter(1, power=500.0),
                   "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        resp, evidence, corr = _compose_for(results, "PZEM-1 mein power high kyu hai?")
        report = _validate_response(resp, "PZEM-1 mein power high kyu hai?", results, evidence, corr)
        assert report["status"] == "VALID"


# ============================================================================
# 13. NONE HANDLING
# ============================================================================

class TestPhase3DNoneHandling:
    """None values must not produce claims."""

    def test_none_diagnostic_recommendations(self):
        results = {"get_meter": _meter(1, power=372.0),
                   "get_diagnostic_recommendations": DIAGNOSTIC_RECS_EMPTY}
        good_response = "PZEM 1 is online. Power: 372.0 W."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(good_response, "PZEM-1 ka power kitna hai?", results, evidence, corr)
        assert report["status"] == "VALID"


# ============================================================================
# 14. ZERO HANDLING
# ============================================================================

class TestPhase3DZeroHandling:
    """Zero values must be validated correctly."""

    def test_zero_power_valid(self):
        results = {"get_meter": _meter(1, power=0.0)}
        good_response = "PZEM 1 is online. Power: 0.0 W."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(good_response, "PZEM-1 ka power kitna hai?", results, evidence, corr)
        assert report["status"] == "VALID"


# ============================================================================
# 15. COST/ENERGY IMPACT VALIDATION
# ============================================================================

class TestPhase3DCostEnergyImpactValidation:
    """Energy/cost impact must not be invented."""

    def test_savings_without_evidence(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = "You can save 10% on your bill."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "enerji kaisi save kare?", results, evidence, corr)
        assert report["status"] == "UNSAFE"

    def test_currency_savings_without_evidence(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = "You can save Rs 500."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "test?", results, evidence, corr)
        assert report["status"] == "UNSAFE"


# ============================================================================
# 16. MAINTENANCE TIMING VALIDATION
# ============================================================================

class TestPhase3DMaintenanceTimingValidation:
    """Maintenance timing must be traceable to evidence."""

    def test_maintenance_tomorrow_without_evidence(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = "Maintenance required tomorrow."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "maintenance kya karna chahiye?", results, evidence, corr)
        assert report["status"] == "UNSAFE"

    def test_maintenance_with_evidence(self):
        results = {"get_meter": _meter(1, power=372.0),
                   "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        good_response = "Schedule inspection promptly."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(good_response, "maintenance kya karna chahiye?", results, evidence, corr)
        # Valid if it matches evidence
        assert report["status"] in ("VALID", "REPAIRABLE")


# ============================================================================
# 17. DUPLICATE UNCERTAINTY REPAIR
# ============================================================================

class TestPhase3DDuplicateUncertaintyRepair:
    """Duplicate uncertainty phrases should be repaired."""

    def test_duplicate_uncertainty_repair(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = "Available data does not establish a specific cause. Available data does not establish a specific cause."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "PZEM-1 ka power kitna hai?", results, evidence, corr)
        # Simple question with uncertainty is REPAIRABLE
        assert report["status"] == "REPAIRABLE"


# ============================================================================
# 18. DEBUG-TAG REJECTION/REPAIR
# ============================================================================

class TestPhase3DDebugTagRepair:
    """Internal debug tags must be removed from responses."""

    def test_debug_tag_repair(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = "[Evidence status: COMPLETE] PZEM 1 is online. Power: 372.0 W."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "PZEM-1 ka power kitna hai?", results, evidence, corr)
        assert report["status"] == "REPAIRABLE"
        assert any("debug_tag" in v for v in report["violations"])

    def test_note_tag_repair(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = "[Note: This is internal] PZEM 1 is online."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "PZEM-1 ka power kitna hai?", results, evidence, corr)
        assert report["status"] == "REPAIRABLE"


# ============================================================================
# 19. SIMPLE-QUESTION CONCISENESS
# ============================================================================

class TestPhase3DSimpleQuestionConciseness:
    """Simple questions must not produce overly lengthy responses."""

    def test_simple_question_uncertainty_removed(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad_response = ("PZEM 1 is online. Power: 372.0 W. Available data does not establish a specific cause.")
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad_response, "PZEM-1 ka power kitna hai?", results, evidence, corr)
        assert report["status"] == "REPAIRABLE"
        assert any("simple_question_uncertainty" in v for v in report["violations"])


# ============================================================================
# 20. SAFE FALLBACK
# ============================================================================

class TestPhase3DSafeFallback:
    """Safe fallback generates deterministic, evidence-grounded responses."""

    def test_fallback_with_measured_evidence(self):
        results = {"get_meter": _meter(2, power=372.0)}
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        fallback = _safe_fallback(evidence, corr, "PZEM-2 ka power kitna hai?")
        assert "372.0" in fallback or "276.0" in fallback or "W" in fallback

    def test_fallback_with_no_evidence(self):
        results = {}
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        fallback = _safe_fallback(evidence, corr, "test?")
        assert len(fallback) > 0

    def test_fallback_includes_uncertainty_for_why(self):
        results = {"get_meter": _meter(1, power=372.0)}
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        corr["status"] = "INSUFFICIENT"
        fallback = _safe_fallback(evidence, corr, "PZEM-1 mein power high kyu hai?")
        assert "does not establish" in fallback.lower()

    def test_fallback_includes_uncertainty_for_action(self):
        results = {"get_meter": _meter(1, power=372.0)}
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        corr["status"] = "PARTIAL"
        fallback = _safe_fallback(evidence, corr, "PZEM-1 ka voltage high hai, kya karu?")
        assert "does not establish" in fallback.lower()


# ============================================================================
# 21. DETERMINISTIC BEHAVIOR
# ============================================================================

class TestPhase3DDeterministicBehavior:
    """Validation must be deterministic."""

    def test_same_input_same_output(self):
        results = {"get_meter": _meter(1, power=372.0)}
        resp, evidence, corr = _compose_for(results, "PZEM-1 ka power kitna hai?")
        report1 = _validate_response(resp, "PZEM-1 ka power kitna hai?", results, evidence, corr)
        report2 = _validate_response(resp, "PZEM-1 ka power kitna hai?", results, evidence, corr)
        assert report1 == report2
        assert report1["status"] == "VALID"

    def test_unsafe_always_unsafe(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad = "Motor overload is causing the issue."
        for _ in range(3):
            report = _validate_response(bad, "PZEM-1 mein power high kyu hai?", results,
                                         _compose_combined_evidence("test", results),
                                         _correlate(results, _compose_combined_evidence("test", results)))
            assert report["status"] == "UNSAFE"


# ============================================================================
# 22. LLM RESPONSE VALIDATION
# ============================================================================

class TestPhase3DLLMResponseValidation:
    """LLM-composed responses must also be validated."""

    def test_llm_unsupported_number_rejected(self):
        results = {"get_meter": _meter(2, power=372.0)}
        llm_answer = "Power is 500 W."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        validated = _apply_phase3d(llm_answer, "PZEM-2 ka power kitna hai?", results, evidence, corr)
        # Should be replaced with safe fallback
        assert "500" not in validated or "372" in validated

    def test_llm_safe_answer_preserved(self):
        results = {"get_meter": _meter(2, power=372.0)}
        llm_answer = "PZEM 2 is online. Power: 372.0 W."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        validated = _apply_phase3d(llm_answer, "PZEM-2 ka power kitna hai?", results, evidence, corr)
        assert "372.0" in validated


# ============================================================================
# 23. EXISTING PHASE 3C RESPONSE ACCEPTED UNCHANGED
# ============================================================================

class TestPhase3DExistingPhase3CResponse:
    """Valid Phase 3C responses should pass Phase 3D unchanged."""

    def test_compose_energy_passes_validation(self):
        results = {"get_meter": _meter(2, power=372.0)}
        resp = _compose_energy("PZEM-2 ka power kitna hai?", results)
        evidence = _compose_combined_evidence("PZEM-2 ka power kitna hai?", results)
        corr = _correlate(results, evidence)
        report = _validate_response(resp, "PZEM-2 ka power kitna hai?", results, evidence, corr)
        assert report["status"] == "VALID"
        assert resp == _apply_phase3d(resp, "PZEM-2 ka power kitna hai?", results, evidence, corr)

    def test_compose_energy_diagnostic_passes_validation(self):
        results = {"get_meter": _meter(1, power=500.0, voltage=245.0),
                   "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        resp = _compose_energy("PZEM-1 mein power high kyu hai?", results)
        evidence = _compose_combined_evidence("PZEM-1 mein power high kyu hai?", results)
        corr = _correlate(results, evidence)
        report = _validate_response(resp, "PZEM-1 mein power high kyu hai?", results, evidence, corr)
        assert report["status"] == "VALID"

    def test_compose_energy_historical_passes_validation(self):
        results = {"get_historical_analysis": HISTORICAL_DATA}
        resp = _compose_energy("10 September ko PZEM-3 ki maximum power kya thi?", results)
        evidence = _compose_combined_evidence("10 September ko PZEM-3 ki maximum power kya thi?", results)
        corr = _correlate(results, evidence)
        report = _validate_response(resp, "10 September ko PZEM-3 ki maximum power kya thi?", results, evidence, corr)
        assert report["status"] == "VALID"


# ============================================================================
# 24. PZEM SCOPE VALIDATION
# ============================================================================

class TestPhase3DPZEMScopeValidation:
    """PZEM questions must keep scope limited."""

    def test_pzem_scope_valid(self):
        results = {"get_meter": _meter(2, power=372.0)}
        resp, evidence, corr = _compose_for(results, "PZEM-2 ka power kitna hai?")
        report = _validate_response(resp, "PZEM-2 ka power kitna hai?", results, evidence, corr)
        assert report["status"] == "VALID"


# ============================================================================
# 25. TIMESTAMP/TIME-WINDOW VALIDATION
# ============================================================================

class TestPhase3DTimestampValidation:
    """Historical responses must include explicit time periods."""

    def test_historical_period_stated(self):
        results = {"get_historical_analysis": HISTORICAL_DATA}
        resp = _compose_energy("10 September ko PZEM-3 ki maximum power kya thi?", results)
        evidence = _compose_combined_evidence("10 September ko PZEM-3 ki maximum power kya thi?", results)
        corr = _correlate(results, evidence)
        report = _validate_response(resp, "10 September ko PZEM-3 ki maximum power kya thi?", results, evidence, corr)
        assert report["status"] == "VALID"

    def test_historical_2026_not_1970(self):
        results = {"get_historical_analysis": HISTORICAL_DATA}
        resp = _compose_energy("10 September ko PZEM-3 ki maximum power kya thi?", results)
        assert "2026" in resp
        assert "1970" not in resp

    def test_compose_combined_evidence_historical_no_1970(self):
        """Internal _compose_combined_evidence must use _fmt_ts_s for Unix-second timestamps."""
        from ai.ask_bob import _compose_combined_evidence, _fmt_ts_s
        results = {"get_historical_analysis": HISTORICAL_DATA}
        evidence = _compose_combined_evidence("test", results)
        measured = " ".join(evidence["measured"])
        # requested_start=1788998400 (seconds) must render as 2026, NOT 1970
        assert "2026" in measured
        assert "1970" not in measured
        # Verify _fmt_ts_s produces correct output
        assert _fmt_ts_s(1788998400) == "2026-09-10 00:00 UTC"

    def test_max_timestamp_uses_milliseconds(self):
        """max_timestamp (milliseconds) must still use _fmt_ts, not _fmt_ts_s."""
        from ai.ask_bob import _fmt_ts, _fmt_ts_s
        # max_timestamp is in milliseconds
        assert _fmt_ts(1789040000000) == "2026-09-10 11:33 UTC"
        # _fmt_ts_s would produce wrong output for milliseconds
        assert _fmt_ts_s(1789040000000) != "2026-09-10 11:33 UTC"

    def test_requested_end_seconds_correct(self):
        """requested_end=1789084799 (seconds) must render as 2026-09-10 23:59 UTC."""
        from ai.ask_bob import _fmt_ts_s
        assert _fmt_ts_s(1789084799) == "2026-09-10 23:59 UTC"


# ============================================================================
# 26. APPLY_PHASE3D INTEGRATION
# ============================================================================

class TestPhase3DApplyPhase3DIntegration:
    """_apply_phase3d correctly validates and falls back."""

    def test_valid_passes_through(self):
        results = {"get_meter": _meter(2, power=372.0)}
        resp, evidence, corr = _compose_for(results, "PZEM-2 ka power kitna hai?")
        validated = _apply_phase3d(resp, "PZEM-2 ka power kitna hai?", results, evidence, corr)
        assert validated == resp

    def test_unsafe_replaced_with_fallback(self):
        results = {"get_meter": _meter(1, power=372.0)}
        unsafe = "Motor overload is causing high power."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        validated = _apply_phase3d(unsafe, "PZEM-1 mein power high kyu hai?", results, evidence, corr)
        assert validated != unsafe
        assert "Motor overload" not in validated

    def test_debug_tags_removed(self):
        results = {"get_meter": _meter(1, power=372.0)}
        with_tags = "[Evidence status: COMPLETE] PZEM 1 is online. Power: 372.0 W."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        validated = _apply_phase3d(with_tags, "PZEM-1 ka power kitna hai?", results, evidence, corr)
        assert "[Evidence status:" not in validated

    def test_empty_returns_default(self):
        results = {}
        validated = _apply_phase3d("", "test?", results,
                                    _compose_combined_evidence("test", results),
                                    _correlate(results, _compose_combined_evidence("test", results)))
        assert validated == "I don't have enough current data to answer that."

    def test_none_returns_default(self):
        results = {}
        validated = _apply_phase3d(None, "test?", results,
                                    _compose_combined_evidence("test", results),
                                    _correlate(results, _compose_combined_evidence("test", results)))
        assert validated == "I don't have enough current data to answer that."

    def test_fallback_contains_evidence_values(self):
        results = {"get_meter": _meter(2, power=372.0)}
        unsafe = "Power is 500 W."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        validated = _apply_phase3d(unsafe, "PZEM-2 ka power kitna hai?", results, evidence, corr)
        # Fallback should contain 372.0 or W, not 500
        assert "500" not in validated


# ============================================================================
# CRITICAL NEGATIVE TESTS
# ============================================================================

class TestPhase3DCriticalNegativeTests:
    """These MUST fail validation."""

    def test_A_power_372_vs_500(self):
        results = {"get_meter": _meter(2, power=372.0)}
        bad = "Power = 500 W"
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad, "PZEM-2 ka power kitna hai?", results, evidence, corr)
        assert report["status"] == "UNSAFE"

    def test_B_no_diagnosis_cause_claim(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad = "Motor overload is the cause."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad, "PZEM-1 mein power high kyu hai?", results, evidence, corr)
        assert report["status"] == "UNSAFE"

    def test_C_confidence_05_not_high(self):
        results = {"get_meter": _meter(1, power=372.0),
                   "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        bad = "High confidence diagnosis."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad, "PZEM-1 mein power high kyu hai?", results, evidence, corr)
        assert report["status"] in ("REPAIRABLE", "UNSAFE")

    def test_D_historical_live_mix(self):
        results = {"get_historical_analysis": HISTORICAL_DATA}
        bad = "PZEM 3 historical analysis: Period: 2026-09-10. The meter is online."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad, "10 September ko PZEM-3 ki power kya thi?", results, evidence, corr)
        assert report["status"] == "REPAIRABLE"

    def test_E_system_attribution(self):
        results = {"get_system_summary": SYSTEM_SUMMARY}
        bad = "PZEM-3 caused it."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad, "system peak kyun tha?", results, evidence, corr)
        assert report["status"] == "REPAIRABLE"

    def test_F_savings_without_evidence(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad = "Rs 500 can be saved."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad, "test?", results, evidence, corr)
        assert report["status"] == "UNSAFE"

    def test_G_maintenance_without_timing(self):
        results = {"get_meter": _meter(1, power=372.0)}
        bad = "Maintenance required tomorrow."
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate(results, evidence)
        report = _validate_response(bad, "maintenance kya karna chahiye?", results, evidence, corr)
        assert report["status"] == "UNSAFE"


# ============================================================================
# 27. ASK_BOB INTEGRATION
# ============================================================================

class TestPhase3DAskBobIntegration:
    """ask_bob() must apply Phase 3D to final answers."""

    def test_ask_bob_simple_reading_clean(self):
        result = ask_bob("PZEM-2 ka power kitna hai?")
        assert result["status"] == "ok"
        answer = result["answer"]
        assert "[" not in answer  # No debug tags
        assert "does not establish" not in answer.lower() or "power" not in answer.lower()

    def test_ask_bob_why_question_has_uncertainty(self):
        result = ask_bob("PZEM-1 mein power high kyu hai?")
        assert result["status"] == "ok"
        answer = result["answer"]
        assert len(answer) > 0

    def test_ask_bob_no_internal_tags(self):
        result = ask_bob("PZEM-1 ka power kitna hai?")
        assert result["status"] == "ok"
        answer = result["answer"]
        assert "[Evidence status:" not in answer
        assert "[Note:" not in answer
        assert "DEBUG:" not in answer


# ============================================================================
# 28. VALIDATION REPORT STRUCTURE
# ============================================================================

class TestPhase3DValidationReportStructure:
    """Validation report must have all required fields."""

    def test_report_has_all_fields(self):
        results = {"get_meter": _meter(1, power=372.0)}
        resp, evidence, corr = _compose_for(results, "PZEM-1 ka power kitna hai?")
        report = _validate_response(resp, "PZEM-1 ka power kitna hai?", results, evidence, corr)
        assert "status" in report
        assert "violations" in report
        assert "warnings" in report
        assert "scope_ok" in report
        assert "time_ok" in report
        assert "diagnosis_ok" in report
        assert "action_ok" in report
        assert "numeric_values_ok" in report
        assert "debug_tags_ok" in report
        assert report["status"] in ("VALID", "REPAIRABLE", "UNSAFE")
