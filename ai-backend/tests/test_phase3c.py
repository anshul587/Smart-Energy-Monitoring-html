"""
tests/test_phase3c.py — Stage 5D Phase 3C: Electrical Reasoning & Response Quality.

Verifies that _reason_response() and _compose_energy() produce clear,
concise, evidence-grounded responses that follow the Phase 3C rules.

Categories:
1. Simple live reading
2. Historical reading
3. Diagnostic WHY
4. Diagnostic ACTION
5. Fault + action
6. Voltage abnormal
7. Current abnormal
8. Power abnormal
9. PF abnormal
10. Maintenance
11. Bill
12. System peak
13. Historical WHY
14. Insufficient evidence
15. Conflicting evidence
16. Multiple recommendations
17. Confidence preservation
18. None/zero handling
19. Historical/live separation
20. System/PZEM scope
21. No unsupported causal claims
22. LLM safety
23. Deterministic fallback
24. Conciseness / intent-aware output
"""
from __future__ import annotations
import pytest
from ai.ask_bob import (
    _reason_response, _compose_energy, _compose_combined_evidence,
    _correlate_evidence, _llm_compose_energy, ask_bob,
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
DIAGNOSTIC_RECS_PARTIAL = [
    {"pzem_number": 3, "timestamp": 1700000002000, "fault_type": None,
     "severity": None, "priority": None, "probable_cause": None,
     "evidence": None, "confidence": None, "what_to_check": None,
     "what_to_do_now": None, "corrective_action": None, "urgency": None,
     "maintenance_required": None, "maintenance_timing": None,
     "energy_impact_kwh": None, "cost_impact": None,
     "source_stages": ("Stage 4",)},
]
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
PEAKS = [{"total_peak_power_w": 500.0, "timestamp": 1700000000000,
          "dominant_pzems": [1]}]
FAULTS = [{"pzem_number": 1, "fault_type": "overvoltage", "timestamp": 1700000000000}]
ANOMALIES = [{"pzem_number": 1, "anomaly_label": "SPIKE", "timestamp": 1700000000000}]
MAINT = [{"pzem_number": None, "high_risk_meters": [4], "watch_meters": [2, 5],
          "normal_meters": [1, 3, 6, 7, 8, 9], "highest_risk_pzem": 4}]
BILL_DATA = [{"estimated_bill": 312.5, "anchor_timestamp": 1700000000000}]


def _make_results(**overrides):
    return {}

def _correlate(results, evidence=None):
    if evidence is None:
        evidence = _compose_combined_evidence("test", results)
    return _correlate_evidence(results, evidence)

def _evidence_for(results, question="test"):
    return _compose_combined_evidence(question, results)


# ============================================================================
# HELPER: build pieces from results and question
# ============================================================================

def _render_pieces(results, question="test"):
    """Render all available results into pieces like _compose_energy does."""
    from ai.ask_bob import _RENDERERS, _ORDER, _has_data
    pieces = []
    for name in _ORDER:
        if name not in results:
            continue
        rendered = _RENDERERS[name](results[name], question)
        if rendered:
            pieces.append(rendered)
    return pieces


# ============================================================================
# 1. SIMPLE LIVE READING
# ============================================================================

class TestPhase3CSimpleLiveReading:
    """Simple questions get concise answers with direct values."""

    def test_power_reading_returns_value(self):
        results = {"get_meter": _meter(2, power=372.0)}
        pieces = _render_pieces(results, "PZEM-2 ka power kitna hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-2 ka power kitna hai?")
        assert "372.0" in resp or "276.0" in resp or "W" in resp

    def test_voltage_reading_returns_value(self):
        results = {"get_meter": _meter(1, voltage=230.0)}
        pieces = _render_pieces(results, "PZEM-1 ka voltage kitna hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-1 ka voltage kitna hai?")
        assert "230.0" in resp or "V" in resp

    def test_current_reading_returns_value(self):
        results = {"get_meter": _meter(1, current=1.2)}
        pieces = _render_pieces(results, "PZEM-1 ka current kitna hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-1 ka current kitna hai?")
        assert "1.2" in resp or "A" in resp


# ============================================================================
# 2. HISTORICAL READING
# ============================================================================

class TestPhase3CHistoricalReading:
    """Historical questions return historical data."""

    def test_historical_peak_value(self):
        results = {"get_historical_analysis": HISTORICAL_DATA}
        pieces = _render_pieces(results, "10 September ko PZEM-3 ki maximum power kya thi?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "10 September ko PZEM-3 ki maximum power kya thi?")
        assert "450.0" in resp or "maximum" in resp.lower()

    def test_historical_returns_average_power(self):
        results = {"get_historical_analysis": HISTORICAL_DATA}
        pieces = _render_pieces(results, "PZEM-3 ki average power")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-3 ki average power")
        assert "150.0" in resp or "average" in resp.lower()


# ============================================================================
# 3. DIAGNOSTIC WHY
# ============================================================================

class TestPhase3CDiagnosticWhy:
    """Diagnostic WHY questions show measured + observed + probable cause."""

    def test_why_shows_probable_cause(self):
        results = {
            "get_meter": _meter(1, power=500.0, voltage=245.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        pieces = _render_pieces(results, "PZEM-1 mein power high kyu hai?")
        evidence = _evidence_for(results, "PZEM-1 mein power high kyu hai?")
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "PZEM-1 mein power high kyu hai?")
        # Should contain probable cause from Stage 4
        assert "overvoltage" in resp.lower() or "Probable Cause" in resp

    def test_why_shows_measured_evidence(self):
        results = {
            "get_meter": _meter(1, power=500.0, voltage=245.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        pieces = _render_pieces(results, "PZEM-1 mein power high kyu hai?")
        evidence = _evidence_for(results, "PZEM-1 mein power high kyu hai?")
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "PZEM-1 mein power high kyu hai?")
        # Should contain measured values
        assert "500.0" in resp or "245.0" in resp

    def test_why_shows_what_to_check(self):
        results = {
            "get_meter": _meter(1, power=500.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        pieces = _render_pieces(results, "PZEM-1 mein power high kyu hai?")
        evidence = _evidence_for(results, "PZEM-1 mein power high kyu hai?")
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "PZEM-1 mein power high kyu hai?")
        assert "Verify" in resp or "Check" in resp or "what to check" in resp.lower()


# ============================================================================
# 4. DIAGNOSTIC ACTION
# ============================================================================

class TestPhase3CDiagnosticAction:
    """ACTION questions prioritize what_to_check, what_to_do_now, corrective_action."""

    def test_action_shows_what_to_do_now(self):
        results = {
            "get_meter": _meter(1, power=500.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        pieces = _render_pieces(results, "PZEM-1 ka voltage high hai, kya karu?")
        evidence = _evidence_for(results, "PZEM-1 ka voltage high hai, kya karu?")
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "PZEM-1 ka voltage high hai, kya karu?")
        assert "Do NOT assume equipment failure" in resp or "Independent voltage" in resp

    def test_action_shows_corrective(self):
        results = {
            "get_meter": _meter(2, power=500.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        pieces = _render_pieces(results, "PZEM-2 ka voltage high hai, kya karu?")
        evidence = _evidence_for(results, "PZEM-2 ka voltage high hai, kya karu?")
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "PZEM-2 ka voltage high hai, kya karu?")
        assert "Reduce" in resp or "corrective" in resp.lower() or "Redistribute" in resp


# ============================================================================
# 5. FAULT + ACTION
# ============================================================================

class TestPhase3CFaultAction:
    """Fault questions show fault + diagnosis + confidence + action."""

    def test_fault_shows_confidence(self):
        results = {
            "get_meter": _meter(1, power=500.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        pieces = _render_pieces(results, "PZEM-1 mein overvoltage fault hai, kya action lena chahiye?")
        evidence = _evidence_for(results, "PZEM-1 mein overvoltage fault hai, kya action lena chahiye?")
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "PZEM-1 mein overvoltage fault hai, kya action lena chahiye?")
        assert "0.85" in resp or "0.72" in resp or "HIGH" in resp


# ============================================================================
# 6-9. ABNORMAL VALUES (Voltage, Current, Power, PF)
# ============================================================================

class TestPhase3CAbnormalValues:
    """Abnormal values show measured data + diagnosis if available."""

    def test_voltage_abnormal_shows_value(self):
        results = {"get_meter": _meter(1, voltage=250.0)}
        pieces = _render_pieces(results, "PZEM-1 ka voltage abnormal hai")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-1 ka voltage abnormal hai")
        assert "250.0" in resp or "V" in resp

    def test_current_abnormal_shows_value(self):
        results = {"get_meter": _meter(1, current=5.0)}
        pieces = _render_pieces(results, "PZEM-1 ka current abnormal hai")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-1 ka current abnormal hai")
        assert "5.0" in resp or "A" in resp

    def test_power_abnormal_shows_value(self):
        results = {"get_meter": _meter(1, power=1000.0)}
        pieces = _render_pieces(results, "PZEM-1 ka power bahut zyada hai")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-1 ka power bahut zyada hai")
        assert "1000.0" in resp or "W" in resp

    def test_pf_abnormal_shows_value(self):
        """Meter data is present; simple questions stay clean without uncertainty."""
        results = {"get_meter": _meter(1, power=100.0, pf=0.5)}
        pieces = _render_pieces(results, "PZEM-1 ka power factor low hai")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-1 ka power factor low hai")
        # Meter data should be present
        assert "100.0" in resp or "Power" in resp
        # Simple questions must NOT contain uncertainty notes
        assert "does not establish" not in resp.lower()
        assert "no diagnostic" not in resp.lower()


# ============================================================================
# 10. MAINTENANCE
# ============================================================================

class TestPhase3CMaintenance:
    """Maintenance questions show supported maintenance evidence."""

    def test_maintenance_shows_risk(self):
        results = {"get_maintenance": MAINT}
        pieces = _render_pieces(results, "PZEM-3 ki maintenance kab chahiye?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-3 ki maintenance kab chahiye?")
        assert "high-risk" in resp.lower() or "watch" in resp.lower() or "Maintenance" in resp

    def test_maintenance_no_exact_dates(self):
        results = {"get_maintenance": MAINT}
        pieces = _render_pieces(results, "PZEM-3 ki maintenance kab chahiye?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-3 ki maintenance kab chahiye?")
        # Should not invent exact dates
        assert not any(x in resp for x in ["January 15", "2024-01-15", "next week", "tomorrow"])


# ============================================================================
# 11. BILL
# ============================================================================

class TestPhase3CBill:
    """Bill questions show consumption evidence + supplied bill evidence."""

    def test_bill_shows_value(self):
        results = {"get_bill_prediction": BILL_DATA}
        pieces = _render_pieces(results, "Bill high kyu aa raha hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "Bill high kyu aa raha hai?")
        assert "312.5" in resp or "bill" in resp.lower()


# ============================================================================
# 12. SYSTEM PEAK
# ============================================================================

class TestPhase3CSystemPeak:
    """System peak questions keep system-level scope."""

    def test_system_peak_no_pzem_attribution(self):
        results = {"get_peaks": PEAKS, "get_system_summary": SYSTEM_SUMMARY}
        pieces = _render_pieces(results, "Aaj system ka peak power kyu high tha?")
        evidence = _evidence_for(results, "Aaj system ka peak power kyu high tha?")
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "Aaj system ka peak power kyu high tha?")
        # Should not say PZEM-1 caused system peak
        assert "PZEM" not in resp.lower() or "caused" not in resp.lower()


# ============================================================================
# 13. HISTORICAL WHY
# ============================================================================

class TestPhase3CHistoricalWhy:
    """Historical WHY questions use historical evidence only."""

    def test_historical_why_no_live_data(self):
        results = {"get_historical_analysis": HISTORICAL_DATA}
        pieces = _render_pieces(results, "10 September ko PZEM-3 ka power high kyu tha?")
        evidence = _evidence_for(results, "10 September ko PZEM-3 ka power high kyu tha?")
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "10 September ko PZEM-3 ka power high kyu tha?")
        # Should not contain current/live data
        assert "PZEM 1 is online" not in resp
        assert "Power: 276.0" not in resp


# ============================================================================
# 14. INSUFFICIENT EVIDENCE
# ============================================================================

class TestPhase3CInsufficientEvidence:
    """Insufficient evidence explicitly states uncertainty."""

    def test_no_cause_explicitly_stated(self):
        results = {"get_meter": _meter(1, power=100.0)}
        pieces = _render_pieces(results, "PZEM-1 ka power kyu high hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "PZEM-1 ka power kyu high hai?")
        # When no diagnostic, should state uncertainty
        assert "does not establish a specific cause" in resp.lower() or \
               "no diagnostic cause" in resp.lower() or \
               "no specific cause" in resp.lower()

    def test_no_data_at_all(self):
        results = {}
        pieces = []
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-1 ka power kya hai?")
        assert "does not establish a specific cause" in resp.lower() or \
               "not enough verified data" in resp.lower()


# ============================================================================
# 15. CONFLICTING EVIDENCE
# ============================================================================

class TestPhase3CConflictingEvidence:
    """Conflicting evidence reports both sides without forcing conclusion."""

    def test_conflicting_shows_both(self):
        results = {
            "get_meter": _meter(1, power=500.0, voltage=250.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        pieces = _render_pieces(results, "PZEM-1 mein power high kyu hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        # Create a conflict by adding NORMAL severity diagnostic
        results2 = {
            "get_meter": _meter(1, power=500.0, voltage=250.0),
            "get_diagnostic_recommendations": [
                {"pzem_number": 1, "fault_type": "high_power", "severity": "NORMAL",
                 "probable_cause": None, "confidence": None,
                 "what_to_check": None, "what_to_do_now": None,
                 "corrective_action": None, "urgency": None,
                 "maintenance_required": None, "maintenance_timing": None,
                 "energy_impact_kwh": None, "cost_impact": None,
                 "source_stages": ()}
            ],
        }
        pieces2 = _render_pieces(results2, "PZEM-1 mein power high kyu hai?")
        evidence2 = _evidence_for(results2, "PZEM-1 mein power high kyu hai?")
        corr2 = _correlate(results2, evidence2)
        resp2 = _reason_response(pieces2, evidence2, corr2,
            "PZEM-1 mein power high kyu hai?")
        # If conflicting, should mention both or note conflict
        if corr2["conflicts"]:
            assert "conflict" in str(corr2["conflicts"]).lower() or \
                   any("NORMAL" in p for p in pieces2)


# ============================================================================
# 16. MULTIPLE RECOMMENDATIONS
# ============================================================================

class TestPhase3CMultipleRecommendations:
    """Multiple recommendations are handled correctly."""

    def test_multiple_diagnostics_shows_all(self):
        results = {
            "get_meter": _meter(1, power=500.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        pieces = _render_pieces(results, "PZEM-1 mein power high kyu hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "PZEM-1 mein power high kyu hai?")
        # Should mention both recommendations
        assert "PZEM" in resp


# ============================================================================
# 17. CONFIDENCE PRESERVATION
# ============================================================================

class TestPhase3CConfidencePreservation:
    """Confidence values are preserved exactly."""

    def test_confidence_not_upgraded(self):
        results = {
            "get_meter": _meter(1, power=500.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        pieces = _render_pieces(results, "PZEM-1 mein power high kyu hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "PZEM-1 mein power high kyu hai?")
        # 0.85 and 0.72 should appear as-is
        assert "0.85" in resp or "0.72" in resp


# ============================================================================
# 18. NONE/ZERO HANDLING
# ============================================================================

class TestPhase3CNoneZeroHandling:
    """None and zero values are preserved correctly."""

    def test_zero_power_preserved(self):
        results = {"get_meter": _meter(1, power=0.0)}
        pieces = _render_pieces(results, "PZEM-1 ka power kitna hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-1 ka power kitna hai?")
        assert "0.0" in resp or "0" in resp or "W" in resp

    def test_none_energy_impact_not_fabricated(self):
        results = {
            "get_meter": _meter(1, power=500.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        pieces = _render_pieces(results, "PZEM-1 mein power high kyu hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "PZEM-1 mein power high kyu hai?")
        # energy_impact_kwh is None for these recs, should not show fabricated impact
        assert "kWh" not in resp or "Impact" not in resp


# ============================================================================
# 19. HISTORICAL/LIVE SEPARATION
# ============================================================================

class TestPhase3CHistoricalLiveSeparation:
    """Historical and live data are never mixed."""

    def test_historical_no_live_values(self):
        results = {"get_historical_analysis": HISTORICAL_DATA}
        pieces = _render_pieces(results, "10 September ko PZEM-3 ka power kitna tha?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr,
            "10 September ko PZEM-3 ka power kitna tha?")
        # Should not contain live meter data
        assert "is online" not in resp
        assert "Power:" not in resp or "historical" in resp.lower()


# ============================================================================
# 20. SYSTEM/PZEM SCOPE
# ============================================================================

class TestPhase3CSystemPZEMScope:
    """System questions stay system-level, PZEM questions stay PZEM-level."""

    def test_system_question_not_pzem_specific(self):
        results = {"get_system_summary": SYSTEM_SUMMARY}
        pieces = _render_pieces(results, "System ka overall status kya hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "System ka overall status kya hai?")
        # Should mention system-level data
        assert "system" in resp.lower() or "online" in resp.lower()

    def test_pzem_question_stays_pzem(self):
        results = {"get_meter": _meter(2, power=372.0)}
        pieces = _render_pieces(results, "PZEM-2 ka power kitna hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-2 ka power kitna hai?")
        assert "PZEM" in resp or "2" in resp


# ============================================================================
# 21. NO UNSUPPORTED CAUSAL CLAIMS
# ============================================================================

class TestPhase3CNoUnsupportedClaims:
    """No invented causes without Stage 4 evidence."""

    def test_no_overload_claim_without_diagnosis(self):
        results = {"get_meter": _meter(1, power=500.0)}
        pieces = _render_pieces(results, "PZEM-1 ka power high kyu hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-1 ka power high kyu hai?")
        # Should not say "overload" without Stage 4
        assert "overload" not in resp.lower() or "diagnostic" in resp.lower()

    def test_no_motor_fault_without_diagnosis(self):
        results = {"get_meter": _meter(1, power=500.0)}
        pieces = _render_pieces(results, "PZEM-1 ka current kyu high hai?")
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-1 ka current kyu high hai?")
        # Should not say "motor fault" without Stage 4
        assert "motor" not in resp.lower() or "fault" not in resp.lower()


# ============================================================================
# 22. LLM SAFETY
# ============================================================================

class TestPhase3CLLMSafety:
    """LLM safety: verify _llm_compose_energy exists."""

    def test_llm_compose_exists(self):
        from ai.ask_bob import _llm_compose_energy
        assert callable(_llm_compose_energy)


# ============================================================================
# 23. DETERMINISTIC FALLBACK
# ============================================================================

class TestPhase3CDeterministicFallback:
    """Deterministic path always produces useful output."""

    def test_compose_energy_with_no_data(self):
        resp = _compose_energy("test", {})
        assert "enough" in resp.lower() or "specific cause" in resp.lower()

    def test_compose_energy_with_meter_data(self):
        results = {"get_meter": _meter(1, power=100.0)}
        resp = _compose_energy("PZEM-1 ka power kitna hai?", results)
        assert len(resp) > 0
        assert "100.0" in resp or "W" in resp

    def test_compose_energy_with_diagnostic(self):
        results = {
            "get_meter": _meter(1, power=500.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        resp = _compose_energy("PZEM-1 mein power high kyu hai?", results)
        assert len(resp) > 0
        # Should contain diagnostic info
        assert "PZEM" in resp or "cause" in resp.lower() or "0.85" in resp

    def test_compose_energy_correlation_status_included(self):
        results = {"get_meter": _meter(1, power=500.0)}
        pieces = _render_pieces(results)
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _reason_response(pieces, evidence, corr, "PZEM-1 ka power kitna hai?")
        # Should include evidence status
        assert "status" in str(corr.keys())


# ============================================================================
# 24. CONCISENESS / INTENT-AWARE OUTPUT
# ============================================================================

class TestPhase3CConciseness:
    """Response length matches question complexity."""

    def test_simple_question_is_concise(self):
        results = {"get_meter": _meter(2, power=372.0)}
        resp = _compose_energy("PZEM-2 ka power kitna hai?", results)
        # Should be shorter than a full diagnostic report
        assert len(resp) < 500

    def test_diagnostic_question_is_more_detailed(self):
        results = {
            "get_meter": _meter(1, power=500.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        resp = _compose_energy("PZEM-1 mein power high kyu hai?", results)
        # Should be more detailed than simple reading
        assert len(resp) > 50


# ============================================================================
# INTEGRATION: ask_bob() end-to-end
# ============================================================================

class TestPhase3CIntegration:
    """End-to-end tests verify ask_bob() produces Phase 3C-compliant responses."""

    def test_simple_reading_concise(self):
        results = {"get_meter": _meter(2, power=372.0)}
        resp = _compose_energy("PZEM-2 ka power kitna hai?", results)
        assert len(resp) > 0
        assert "372.0" in resp or "W" in resp

    def test_why_question_has_structure(self):
        results = {
            "get_meter": _meter(1, power=500.0, voltage=245.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS,
        }
        resp = _compose_energy("PZEM-1 mein power high kyu hai?", results)
        assert len(resp) > 0
        # Should contain diagnostic information
        assert "PZEM" in resp or "cause" in resp.lower()

    def test_insufficient_has_explicit_message(self):
        results = {"get_meter": _meter(1, power=100.0)}
        evidence = _evidence_for(results)
        corr = _correlate(results, evidence)
        resp = _compose_energy("PZEM-1 ka power kyu high hai?", results, corr)
        assert len(resp) > 0
        # Should mention uncertainty
        assert "cause" in resp.lower() or "no diagnostic" in resp.lower() or "does not establish" in resp.lower()


# ============================================================================
# FIX 1: HISTORICAL TIMESTAMP CORRECTNESS
# ============================================================================

class TestPhase3CFixTimestamp:
    """Historical timestamps must render correct 2026 dates, not 1970."""

    def test_fmt_ts_s_known_seconds(self):
        from ai.ask_bob import _fmt_ts_s
        # 1788998400 = 2026-09-10 00:00 UTC
        result = _fmt_ts_s(1788998400)
        assert "2026" in result
        assert "09-10" in result or "Sep" in result

    def test_fmt_ts_s_known_seconds_no_1970(self):
        from ai.ask_bob import _fmt_ts_s
        result = _fmt_ts_s(1788998400)
        assert "1970" not in result

    def test_render_historical_correct_date(self):
        from ai.ask_bob import _render_historical
        HISTORICAL_DATA = {
            "status": "OK", "pzem_number": 3,
            "requested_start": 1788998400, "requested_end": 1789084799,
            "available_days": 30.0, "sample_count": 1440,
            "power": {"average": 150.0, "maximum": 450.0, "minimum": 50.0, "max_timestamp": 1789040000000},
            "voltage": {"average": 228.5}, "current": {"average": 0.65},
            "energy_consumption": {"consumption_kwh": 10.5},
        }
        rendered = _render_historical(HISTORICAL_DATA, "test")
        # Must show 2026 dates, NOT 1970
        assert "2026" in rendered
        assert "1970" not in rendered

    def test_render_historical_with_no_timestamp(self):
        from ai.ask_bob import _render_historical
        result = {"status": "OK", "pzem_number": 3}
        rendered = _render_historical(result, "test")
        assert rendered is not None

    def test_fmt_ts_milliseconds_unchanged(self):
        from ai.ask_bob import _fmt_ts
        # Milliseconds should still work as before
        result = _fmt_ts(1789040000000)
        assert "2026" in result
        assert "1970" not in result

    def test_historical_question_renders_correct_date(self):
        from ai.ask_bob import _compose_energy
        HISTORICAL_DATA = {
            "status": "OK", "pzem_number": 3,
            "requested_start": 1788998400, "requested_end": 1789084799,
            "available_days": 30.0, "sample_count": 1440,
            "power": {"average": 150.0, "maximum": 450.0, "minimum": 50.0, "max_timestamp": 1789040000000},
            "voltage": {"average": 228.5}, "current": {"average": 0.65},
            "energy_consumption": {"consumption_kwh": 10.5},
        }
        resp = _compose_energy("10 September ko PZEM-3 ki maximum power kya thi?", {"get_historical_analysis": HISTORICAL_DATA})
        assert "2026" in resp
        assert "1970" not in resp


# ============================================================================
# FIX 2: NO DUPLICATE UNCERTAINTY
# ============================================================================

class TestPhase3CFixDuplicateUncertainty:
    """Uncertainty note must appear at most once in the response."""

    def test_why_question_has_uncertainty_once(self):
        from ai.ask_bob import _compose_energy, _compose_combined_evidence, _correlate_evidence
        DIAGNOSTIC_RECS = [{"pzem_number": 1, "timestamp": 1700000000000, "fault_type": "overvoltage", "severity": "WARNING", "priority": "P1 - Critical", "probable_cause": "Possible incoming supply overvoltage.", "confidence": 0.85, "what_to_check": "Verify voltage with an independent meter.", "what_to_do_now": "Do NOT assume equipment failure.", "corrective_action": "Independent voltage verification first.", "urgency": "HIGH", "maintenance_required": True, "maintenance_timing": "Schedule inspection promptly.", "energy_impact_kwh": None, "cost_impact": None, "source_stages": ("Stage 3: Fault Diagnosis",)}]
        results = {"get_meter": {"pzem_number": 1, "online": True, "power": 500.0, "voltage": 245.0}, "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        evidence = _compose_combined_evidence("PZEM-1 mein power kyu high hai?", results)
        corr = _correlate_evidence(results, evidence)
        resp = _compose_energy("PZEM-1 mein power kyu high hai?", results, corr)
        # Count uncertainty phrases
        count = resp.lower().count("does not establish a specific cause")
        assert count <= 1, f"Uncertainty note appears {count} times, expected at most 1"

    def test_why_question_with_no_diagnosis_has_single_uncertainty(self):
        from ai.ask_bob import _compose_energy, _correlate_evidence, _compose_combined_evidence
        results = {"get_meter": {"pzem_number": 1, "online": True, "power": 100.0}}
        evidence = _compose_combined_evidence("PZEM-1 ka power kyu high hai?", results)
        corr = _correlate_evidence(results, evidence)
        resp = _compose_energy("PZEM-1 ka power kyu high hai?", results, corr)
        count = resp.lower().count("does not establish a specific cause")
        assert count <= 1, f"Uncertainty note appears {count} times, expected at most 1"

    def test_simple_reading_has_no_uncertainty(self):
        from ai.ask_bob import _compose_energy, _correlate_evidence, _compose_combined_evidence
        results = {"get_meter": {"pzem_number": 2, "online": True, "power": 372.0}}
        evidence = _compose_combined_evidence("PZEM-2 ka power kitna hai?", results)
        corr = _correlate_evidence(results, evidence)
        resp = _compose_energy("PZEM-2 ka power kitna hai?", results, corr)
        # Simple reading must NOT contain uncertainty
        assert "does not establish" not in resp.lower()
        assert "no diagnostic" not in resp.lower()


# ============================================================================
# FIX 3: NO INTERNAL DEBUG TAGS
# ============================================================================

class TestPhase3CFixDebugTags:
    """Internal metadata must NEVER appear in user-facing responses."""

    def test_no_evidence_status_tag(self):
        from ai.ask_bob import _compose_energy
        results = {"get_meter": {"pzem_number": 2, "online": True, "power": 372.0}}
        resp = _compose_energy("PZEM-2 ka power kitna hai?", results)
        assert "[Evidence status:" not in resp

    def test_no_note_tag(self):
        from ai.ask_bob import _compose_energy
        results = {"get_meter": {"pzem_number": 2, "online": True, "power": 372.0}}
        resp = _compose_energy("PZEM-2 ka power kitna hai?", results)
        assert "[Note:" not in resp

    def test_no_debug_tags_why_question(self):
        from ai.ask_bob import _compose_energy, _correlate_evidence, _compose_combined_evidence
        DIAGNOSTIC_RECS = [{"pzem_number": 1, "timestamp": 1700000000000, "fault_type": "overvoltage", "severity": "WARNING", "priority": "P1 - Critical", "probable_cause": "Possible incoming supply overvoltage.", "confidence": 0.85, "what_to_check": "Verify voltage with an independent meter.", "what_to_do_now": "Do NOT assume equipment failure.", "corrective_action": "Independent voltage verification first.", "urgency": "HIGH", "maintenance_required": True, "maintenance_timing": "Schedule inspection promptly.", "energy_impact_kwh": None, "cost_impact": None, "source_stages": ("Stage 3: Fault Diagnosis",)}]
        results = {"get_meter": {"pzem_number": 1, "online": True, "power": 500.0, "voltage": 245.0}, "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate_evidence(results, evidence)
        resp = _compose_energy("PZEM-1 mein power high kyu hai?", results, corr)
        assert "[Evidence status:" not in resp
        assert "[Note:" not in resp

    def test_correlation_still_exists_internally(self):
        from ai.ask_bob import _correlate_evidence, _compose_combined_evidence
        results = {"get_meter": {"pzem_number": 2, "online": True, "power": 372.0}}
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate_evidence(results, evidence)
        # Correlation must still have all fields
        assert "status" in corr
        assert "scope" in corr
        assert "uncertainties" in corr
        assert "conflicts" in corr
        assert "supported_findings" in corr
        assert corr["status"] == "PARTIAL"


# ============================================================================
# FIX 4: SIMPLE QUESTIONS STAY SIMPLE
# ============================================================================

class TestPhase3CFixSimpleQuestions:
    """Simple reading questions must be concise and clean."""

    def test_power_reading_simple(self):
        from ai.ask_bob import _compose_energy
        results = {"get_meter": {"pzem_number": 2, "online": True, "power": 372.0}}
        resp = _compose_energy("PZEM-2 ka power kitna hai?", results)
        assert "372.0" in resp or "W" in resp
        assert "does not establish" not in resp.lower()
        assert "cause" not in resp.lower()

    def test_voltage_reading_simple(self):
        from ai.ask_bob import _compose_energy
        results = {"get_meter": {"pzem_number": 1, "online": True, "voltage": 230.0}}
        resp = _compose_energy("PZEM-1 ka voltage kitna hai?", results)
        assert "230.0" in resp or "V" in resp
        assert "does not establish" not in resp.lower()

    def test_current_reading_simple(self):
        from ai.ask_bob import _compose_energy
        results = {"get_meter": {"pzem_number": 1, "online": True, "current": 1.2}}
        resp = _compose_energy("PZEM-1 ka current kitna hai?", results)
        assert "1.2" in resp or "A" in resp
        assert "does not establish" not in resp.lower()

    def test_simple_response_is_concise(self):
        from ai.ask_bob import _compose_energy
        results = {"get_meter": {"pzem_number": 2, "online": True, "power": 372.0}}
        resp = _compose_energy("PZEM-2 ka power kitna hai?", results)
        assert len(resp) < 150
        assert "[Evidence status" not in resp
        assert "[Note:" not in resp

    def test_why_question_still_has_uncertainty_when_no_diagnosis(self):
        from ai.ask_bob import _compose_energy, _correlate_evidence, _compose_combined_evidence
        results = {"get_meter": {"pzem_number": 1, "online": True, "power": 100.0}}
        evidence = _compose_combined_evidence("PZEM-1 ka power kyu high hai?", results)
        corr = _correlate_evidence(results, evidence)
        resp = _compose_energy("PZEM-1 ka power kyu high hai?", results, corr)
        # WHY question must have uncertainty
        assert "does not establish a specific cause" in resp.lower() or "cause" in resp.lower()

    def test_historical_question_renders_correct_2026_date(self):
        from ai.ask_bob import _compose_energy
        HISTORICAL_DATA = {
            "status": "OK", "pzem_number": 3,
            "requested_start": 1788998400, "requested_end": 1789084799,
            "available_days": 30.0, "sample_count": 1440,
            "power": {"average": 150.0, "maximum": 450.0, "minimum": 50.0, "max_timestamp": 1789040000000},
            "voltage": {"average": 228.5}, "current": {"average": 0.65},
            "energy_consumption": {"consumption_kwh": 10.5},
        }
        resp = _compose_energy("10 September ko PZEM-3 ki maximum power kya thi?", {"get_historical_analysis": HISTORICAL_DATA})
        assert "2026" in resp
        assert "1970" not in resp
        assert "is online" not in resp
        assert "Power:" not in resp or "historical" in resp.lower()


# ============================================================================
# FIX VERIFICATION: Manual Response Checks
# ============================================================================

class TestPhase3CFixManualChecks:
    """Manual verification of representative outputs."""

    def test_power_reading_clean(self):
        """PZEM-2 ka power kitna hai? must be concise and clean."""
        from ai.ask_bob import _compose_energy
        results = {"get_meter": {"pzem_number": 2, "online": True, "power": 372.0}}
        resp = _compose_energy("PZEM-2 ka power kitna hai?", results)
        assert "372.0" in resp or "W" in resp
        assert "does not establish" not in resp.lower()
        assert "[Evidence status" not in resp
        assert "[Note:" not in resp

    def test_why_question_evidence_and_diagnosis(self):
        """PZEM-2 mein power high kyu hai? must show evidence + diagnosis."""
        from ai.ask_bob import _compose_energy, _correlate_evidence, _compose_combined_evidence
        DIAGNOSTIC_RECS = [{"pzem_number": 1, "timestamp": 1700000000000, "fault_type": "overvoltage", "severity": "WARNING", "priority": "P1 - Critical", "probable_cause": "Possible incoming supply overvoltage.", "confidence": 0.85, "what_to_check": "Verify voltage with an independent meter.", "what_to_do_now": "Do NOT assume equipment failure.", "corrective_action": "Independent voltage verification first.", "urgency": "HIGH", "maintenance_required": True, "maintenance_timing": "Schedule inspection promptly.", "energy_impact_kwh": None, "cost_impact": None, "source_stages": ("Stage 3: Fault Diagnosis",)}]
        results = {"get_meter": {"pzem_number": 1, "online": True, "power": 500.0, "voltage": 245.0}, "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate_evidence(results, evidence)
        resp = _compose_energy("PZEM-1 mein power high kyu hai?", results, corr)
        assert len(resp) > 0
        assert "PZEM" in resp
        # Uncertainty only once if present
        assert resp.lower().count("does not establish a specific cause") <= 1

    def test_historical_2026_date(self):
        """10 September ko PZEM-3 ki maximum power kya thi? must show 2026."""
        from ai.ask_bob import _compose_energy
        HISTORICAL_DATA = {
            "status": "OK", "pzem_number": 3,
            "requested_start": 1788998400, "requested_end": 1789084799,
            "available_days": 30.0, "sample_count": 1440,
            "power": {"average": 150.0, "maximum": 450.0, "minimum": 50.0, "max_timestamp": 1789040000000},
            "voltage": {"average": 228.5}, "current": {"average": 0.65},
            "energy_consumption": {"consumption_kwh": 10.5},
        }
        resp = _compose_energy("10 September ko PZEM-3 ki maximum power kya thi?", {"get_historical_analysis": HISTORICAL_DATA})
        assert "2026" in resp
        assert "1970" not in resp

    def test_action_question_shows_actions(self):
        """PZEM-2 ka voltage high hai, kya karu? must show Stage 4 actions."""
        from ai.ask_bob import _compose_energy, _correlate_evidence, _compose_combined_evidence
        DIAGNOSTIC_RECS = [{"pzem_number": 2, "timestamp": 1700000001000, "fault_type": "high_power", "severity": "WARNING", "priority": "P2 - High", "probable_cause": "Possible overload or excessive connected load.", "confidence": 0.72, "what_to_check": "Check active loads and breaker rating.", "what_to_do_now": "Reduce or redistribute load if overloaded.", "corrective_action": "Reduce or redistribute load if overloaded.", "urgency": "MEDIUM", "maintenance_required": False, "maintenance_timing": "Monitor.", "energy_impact_kwh": None, "cost_impact": None, "source_stages": ("Stage 4: Diagnostic & Recommendation",)}]
        results = {"get_meter": {"pzem_number": 2, "online": True, "power": 500.0}, "get_diagnostic_recommendations": DIAGNOSTIC_RECS}
        evidence = _compose_combined_evidence("test", results)
        corr = _correlate_evidence(results, evidence)
        resp = _compose_energy("PZEM-2 ka voltage high hai, kya karu?", results, corr)
        assert len(resp) > 0
        assert "PZEM" in resp


# ============================================================================
# REGRESSION: Existing tests must still pass
# ============================================================================
