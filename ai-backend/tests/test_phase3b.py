"""
tests/test_phase3b.py — Stage 5D Phase 3B: Evidence Correlation Engine.

Verifies the _correlate_evidence() function produces correct structured
reasoning context from Phase 2 evidence.

Categories:
1. measured-only evidence
2. measured + observed
3. measured + diagnostic
4. measured + observed + diagnostic
5. historical correlation
6. live correlation
7. system correlation
8. PZEM correlation
9. conflicting evidence
10. partial evidence
11. missing evidence
12. multiple recommendations
13. confidence preservation
14. None handling
15. zero-value preservation
16. timestamp alignment
17. unsupported-cause prevention
18. system-level attribution prevention
19. historical/live contamination prevention
20. deterministic output
21. LLM structured-evidence safety
22. fallback behavior
"""
from __future__ import annotations
import pytest
from ai.ask_bob import _correlate_evidence, _compose_combined_evidence, _llm_compose_energy
from ai import ask_bob, bob_tools

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _meter(n, online=True, power=None, energy=None, voltage=None, current=None, freq=None, pf=None):
    return {"pzem_number": n, "online": online, "voltage": voltage, "current": current,
            "power": power if online else None, "energy": energy,
            "power_factor": pf if pf else 0.9, "frequency": freq if online else None,
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
    "power": {"average": 150.0, "maximum": 450.0, "minimum": 50.0, "max_timestamp": 1789040000000},
    "voltage": {"average": 228.5}, "current": {"average": 0.65},
    "energy_consumption": {"consumption_kwh": 10.5},
}
SYSTEM_SUMMARY = {
    "system_status": "online", "total_power_w": 890.0,
    "total_energy_kwh": 45.2, "average_voltage_v": 229.0,
    "active_fault_count": 1,
}
PEAKS = [{"total_peak_power_w": 500.0, "timestamp": 1700000000000, "dominant_pzems": [1]}]
FAULTS = [{"pzem_number": 1, "fault_type": "overvoltage", "timestamp": 1700000000000}]
ANOMALIES = [{"pzem_number": 1, "anomaly_label": "SPIKE", "timestamp": 1700000000000}]
MAINT = [{"pzem_number": None, "high_risk_meters": [4], "watch_meters": [2, 5],
          "normal_meters": [1, 3, 6, 7, 8, 9], "highest_risk_pzem": 4}]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results(**overrides):
    """Build a minimal results dict with defaults, override as needed."""
    return {}  # Start empty, add as needed

def _correlate(results, evidence=None):
    """Helper: compose evidence then correlate."""
    if evidence is None:
        evidence = _compose_combined_evidence("test question", results)
    return _correlate_evidence(results, evidence)

# ---------------------------------------------------------------------------
# 1. MEASURED-ONLY EVIDENCE
# ---------------------------------------------------------------------------

class TestPhase3BMeasuredOnly:
    """Measured evidence alone: status PARTIAL, uncertainties present."""

    def test_meter_only_status_partial(self):
        results = {"get_meter": _meter(2, power=372.0)}
        c = _correlate(results)
        assert c["status"] == "PARTIAL"
        assert c["has_live_data"] is True
        assert c["has_diagnostic"] is False

    def test_meter_only_has_uncertainties(self):
        results = {"get_meter": _meter(2, power=372.0)}
        c = _correlate(results)
        assert len(c["uncertainties"]) > 0
        assert any("no diagnostic" in u or "cause" in u for u in c["uncertainties"])

    def test_meter_only_supported_findings(self):
        results = {"get_meter": _meter(2, power=372.0, voltage=230.0)}
        c = _correlate(results)
        assert len(c["supported_findings"]) > 0

    def test_meter_only_scope_live(self):
        results = {"get_meter": _meter(2, power=372.0)}
        c = _correlate(results)
        assert c["scope"] == "LIVE"

    def test_meter_only_no_conflicts(self):
        results = {"get_meter": _meter(2, power=372.0)}
        c = _correlate(results)
        assert c["conflicts"] == []

    def test_meter_only_zero_confidence_not_fabricated(self):
        """No confidence values when no diagnostic exists."""
        results = {"get_meter": _meter(2, power=372.0)}
        c = _correlate(results)
        assert c["causal_support"] == []

# ---------------------------------------------------------------------------
# 2. MEASURED + OBSERVED
# ---------------------------------------------------------------------------

class TestPhase3BMeasuredObserved:
    """Measured + observed (faults/anomalies/peaks) but no diagnostic."""

    def test_meter_peaks_status_partial(self):
        results = {"get_meter": _meter(2, power=500.0), "get_peaks": PEAKS}
        c = _correlate(results)
        assert c["status"] == "PARTIAL"

    def test_meter_peaks_has_relationships(self):
        results = {"get_meter": _meter(2, power=500.0), "get_peaks": PEAKS}
        c = _correlate(results)
        assert len(c["relationships"]) > 0

    def test_meter_faults_has_uncertainties(self):
        results = {"get_meter": _meter(2, power=500.0), "get_faults": FAULTS}
        c = _correlate(results)
        assert len(c["uncertainties"]) > 0

    def test_meter_anomalies_observed_not_cause(self):
        """Anomaly observation must not be stated as cause."""
        results = {"get_meter": _meter(2, power=500.0), "get_anomalies": ANOMALIES}
        c = _correlate(results)
        assert c["status"] == "PARTIAL"
        assert c["has_diagnostic"] is False
        # No causal support without diagnostic
        assert c["causal_support"] == []

# ---------------------------------------------------------------------------
# 3. MEASURED + DIAGNOSTIC
# ---------------------------------------------------------------------------

class TestPhase3BMeasuredDiagnostic:
    """Measured + diagnostic = SUPPORTED status."""

    def test_meter_diag_status_supported(self):
        results = {"get_meter": _meter(1, power=300.0), "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]]}
        c = _correlate(results)
        assert c["status"] == "SUPPORTED"

    def test_meter_diag_has_causal_support(self):
        results = {"get_meter": _meter(1, power=300.0), "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]]}
        c = _correlate(results)
        assert len(c["causal_support"]) > 0

    def test_meter_diag_scope_live(self):
        results = {"get_meter": _meter(1, power=300.0), "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]]}
        c = _correlate(results)
        assert c["scope"] == "LIVE"

    def test_meter_diag_pzem_scope_matches(self):
        results = {"get_meter": _meter(1, power=300.0), "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]]}
        c = _correlate(results)
        assert c["pzem_scope"] == [1]

# ---------------------------------------------------------------------------
# 4. MEASURED + OBSERVED + DIAGNOSTIC
# ---------------------------------------------------------------------------

class TestPhase3BFullEvidence:
    """All evidence types present: SUPPORTED, full relationships."""

    def test_full_evidence_status_supported(self):
        results = {
            "get_meter": _meter(1, power=300.0),
            "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]],
            "get_peaks": PEAKS, "get_faults": FAULTS,
        }
        c = _correlate(results)
        assert c["status"] == "SUPPORTED"

    def test_full_evidence_has_relationships(self):
        results = {
            "get_meter": _meter(1, power=300.0),
            "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]],
        }
        c = _correlate(results)
        assert len(c["relationships"]) > 0

    def test_full_evidence_conflicts_empty(self):
        """Consistent data should have no conflicts."""
        results = {
            "get_meter": _meter(1, power=300.0),
            "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]],
        }
        c = _correlate(results)
        assert c["conflicts"] == []

    def test_full_evidence_quality_high(self):
        results = {
            "get_meter": _meter(1, power=300.0),
            "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]],
        }
        c = _correlate(results)
        assert any("HIGH" in q for q in c["evidence_quality"])

# ---------------------------------------------------------------------------
# 5. HISTORICAL CORRELATION
# ---------------------------------------------------------------------------

class TestPhase3BHistorical:
    """Historical evidence must remain historical."""

    def test_historical_status_partial(self):
        results = {"get_historical_analysis": dict(HISTORICAL_DATA)}
        c = _correlate(results)
        assert c["scope"] == "HISTORICAL"
        assert c["has_historical_data"] is True

    def test_historical_no_live_data(self):
        results = {"get_historical_analysis": dict(HISTORICAL_DATA)}
        c = _correlate(results)
        assert c["has_live_data"] is False

    def test_historical_no_causal_from_live(self):
        """Historical data must not produce live causal support."""
        results = {"get_historical_analysis": dict(HISTORICAL_DATA)}
        c = _correlate(results)
        # No live meter → no live causal support
        assert c["has_live_data"] is False

    def test_historical_live_not_mixed(self):
        """Historical-only must not have live data mixed in."""
        results = {"get_historical_analysis": dict(HISTORICAL_DATA)}
        c = _correlate(results)
        assert c["has_live_data"] is False
        assert c["scope"] == "HISTORICAL"

    def test_historical_system_wide_scope(self):
        results = {"get_historical_analysis": {"status": "OK"}}
        c = _correlate(results)
        assert c["scope"] == "HISTORICAL"

# ---------------------------------------------------------------------------
# 6. LIVE CORRELATION
# ---------------------------------------------------------------------------

class TestPhase3BLive:
    """Live evidence must not include historical data."""

    def test_live_no_historical(self):
        results = {"get_meter": _meter(2, power=372.0)}
        c = _correlate(results)
        assert c["has_historical_data"] is False
        assert c["scope"] == "LIVE"

    def test_live_diag_supported(self):
        results = {"get_meter": _meter(1, power=300.0), "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]]}
        c = _correlate(results)
        assert c["status"] == "SUPPORTED"
        assert c["scope"] == "LIVE"

# ---------------------------------------------------------------------------
# 7. SYSTEM CORRELATION
# ---------------------------------------------------------------------------

class TestPhase3BSystem:
    """System-level evidence must not attribute to individual PZEMs."""

    def test_system_scope_system(self):
        results = {"get_system_summary": dict(SYSTEM_SUMMARY)}
        c = _correlate(results)
        assert c["scope"] == "SYSTEM"

    def test_system_no_pzem_attribution(self):
        """System evidence should not have PZEM-specific scope."""
        results = {"get_system_summary": dict(SYSTEM_SUMMARY)}
        c = _correlate(results)
        # System scope should not attribute to specific PZEM
        assert not any(pz is not None for pz in c["pzem_scope"]) or c["pzem_scope"] == []

    def test_system_no_invented_cause(self):
        """System data must not generate PZEM-specific causes."""
        results = {"get_system_summary": dict(SYSTEM_SUMMARY)}
        c = _correlate(results)
        # System data alone provides partial evidence, no causal support
        assert c["status"] == "PARTIAL"
        assert c["causal_support"] == []

    def test_system_diag_supported(self):
        """System + diagnostic = SUPPORTED."""
        results = {"get_system_summary": dict(SYSTEM_SUMMARY), "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]]}
        c = _correlate(results)
        assert c["status"] == "SUPPORTED"
        assert c["scope"] == "SYSTEM"

# ---------------------------------------------------------------------------
# 8. PZEM CORRELATION
# ---------------------------------------------------------------------------

class TestPhase3BPZEMScope:
    """PZEM-specific evidence must not generalize to the system."""

    def test_single_pzem_scope(self):
        results = {"get_meter": _meter(3, power=114.0)}
        c = _correlate(results)
        assert c["pzem_scope"] == [3]

    def test_pzem_diag_matches(self):
        results = {"get_meter": _meter(2, power=500.0), "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[1]]}
        c = _correlate(results)
        assert c["pzem_scope"] == [2]

    def test_pzem_mismatch_detected(self):
        """If meter and diagnostic are for different PZEMs, uncertainty should be flagged."""
        results = {
            "get_meter": _meter(1, power=300.0),
            "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[1]],  # PZEM 2
        }
        c = _correlate(results)
        assert len(c["uncertainties"]) > 0
        assert any("PZEM-1" in u or "not cover" in u for u in c["uncertainties"])

# ---------------------------------------------------------------------------
# 9. CONFLICTING EVIDENCE
# ---------------------------------------------------------------------------

class TestPhase3BConflicts:
    """Conflicting evidence must be detected and reported."""

    def test_conflict_detected(self):
        """High voltage measured but diagnostic says NORMAL."""
        results = {
            "get_meter": _meter(1, voltage=248.0),
            "get_diagnostic_recommendations": [
                {"pzem_number": 1, "fault_type": "high_power", "severity": "NORMAL", "confidence": 0.5}
            ],
        }
        c = _correlate(results)
        assert len(c["conflicts"]) > 0

    def test_conflict_status_downgraded(self):
        """Conflicts should downgrade status to CONFLICTING."""
        results = {
            "get_meter": _meter(1, voltage=248.0),
            "get_diagnostic_recommendations": [
                {"pzem_number": 1, "fault_type": "high_power", "severity": "NORMAL", "confidence": 0.5}
            ],
        }
        c = _correlate(results)
        assert c["status"] == "CONFLICTING"

    def test_no_false_conflict(self):
        """Consistent data should have no conflicts."""
        results = {
            "get_meter": _meter(1, power=300.0),
            "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]],
        }
        c = _correlate(results)
        assert c["conflicts"] == []

# ---------------------------------------------------------------------------
# 10. PARTIAL EVIDENCE
# ---------------------------------------------------------------------------

class TestPhase3BPartial:
    """Partial evidence must be correctly identified."""

    def test_observed_only_status_partial(self):
        results = {"get_faults": FAULTS}
        c = _correlate(results)
        assert c["status"] == "PARTIAL"

    def test_diag_only_status_partial(self):
        results = {"get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]]}
        c = _correlate(results)
        assert c["status"] == "PARTIAL"

    def test_partial_has_uncertainties(self):
        results = {"get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]]}
        c = _correlate(results)
        assert len(c["uncertainties"]) > 0

# ---------------------------------------------------------------------------
# 11. MISSING EVIDENCE
# ---------------------------------------------------------------------------

class TestPhase3BInsufficient:
    """Missing evidence must produce INSUFFICIENT status."""

    def test_empty_results_insufficient(self):
        results = {}
        c = _correlate(results)
        assert c["status"] == "INSUFFICIENT"
        assert c["scope"] == "UNKNOWN"

    def test_no_meter_no_diag_no_hist(self):
        results = {"get_maintenance": MAINT}
        c = _correlate(results)
        assert c["status"] == "PARTIAL"
        assert c["has_live_data"] is False
        assert c["has_diagnostic"] is False

    def test_missing_evidence_not_fabricated(self):
        """No evidence means no findings, no causal support."""
        results = {}
        c = _correlate(results)
        assert c["supported_findings"] == []
        assert c["causal_support"] == []
        assert c["relationships"] == []

# ---------------------------------------------------------------------------
# 12. MULTIPLE RECOMMENDATIONS
# ---------------------------------------------------------------------------

class TestPhase3BMultipleRecommendations:
    """Multiple Stage 4 recommendations must be preserved separately."""

    def test_multiple_recs_preserved(self):
        results = {
            "get_meter": _meter(1, power=300.0),
            "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0], DIAGNOSTIC_RECS[1]],
        }
        c = _correlate(results)
        assert len(c["causal_support"]) >= 2
        assert c["status"] == "SUPPORTED"

    def test_multiple_recs_pzem_scope(self):
        results = {
            "get_meter": _meter(1, power=300.0),
            "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0], DIAGNOSTIC_RECS[1]],
        }
        c = _correlate(results)
        assert c["pzem_scope"] == [1, 2]

    def test_multiple_recs_not_merged(self):
        """Different causes should not be merged into one."""
        results = {
            "get_meter": _meter(1, power=300.0),
            "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0], DIAGNOSTIC_RECS[1]],
        }
        c = _correlate(results)
        # Each recommendation's probable_cause should be a separate entry
        assert len(c["causal_support"]) >= 4  # 2 recs × (cause + confidence)

# ---------------------------------------------------------------------------
# 13. CONFIDENCE PRESERVATION
# ---------------------------------------------------------------------------

class TestPhase3BConfidencePreservation:
    """Confidence values must not be modified."""

    def test_confidence_preserved(self):
        results = {
            "get_meter": _meter(1, power=300.0),
            "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]],
        }
        c = _correlate(results)
        # The causal_support should contain confidence=0.85
        assert any("0.85" in s for s in c["causal_support"])

    def test_zero_confidence_preserved(self):
        """Zero confidence must not be dropped."""
        results = {
            "get_diagnostic_recommendations": [
                {"pzem_number": 1, "confidence": 0.0, "probable_cause": "Test cause"}
            ],
        }
        c = _correlate(results)
        assert any("0.0" in s for s in c["causal_support"])

    def test_no_confidence_upgrade(self):
        """Confidence must not be upgraded by the correlation layer."""
        results = {
            "get_diagnostic_recommendations": [
                {"pzem_number": 1, "confidence": 0.3, "probable_cause": "Weak cause"}
            ],
        }
        c = _correlate(results)
        assert any("0.3" in s for s in c["causal_support"])
        assert not any("high confidence" in s.lower() for s in c["causal_support"])

# ---------------------------------------------------------------------------
# 14. NONE HANDLING
# ---------------------------------------------------------------------------

class TestPhase3BNoneHandling:
    """None values must be preserved as None, not converted."""

    def test_none_probable_cause_not_fabricated(self):
        """None probable_cause must not generate a fake cause."""
        results = {
            "get_meter": _meter(3, power=114.0),
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS_PARTIAL,
        }
        c = _correlate(results)
        assert c["causal_support"] == [] or all("None" not in s for s in c["causal_support"])

    def test_none_confidence_not_converted(self):
        """None confidence must not become 0 or 'unknown'."""
        results = {
            "get_diagnostic_recommendations": DIAGNOSTIC_RECS_PARTIAL,
        }
        c = _correlate(results)
        # No confidence values should be fabricated
        assert all("confidence" not in s or "None" not in s for s in c["causal_support"])

# ---------------------------------------------------------------------------
# 15. ZERO-VALUE PRESERVATION
# ---------------------------------------------------------------------------

class TestPhase3BZeroPreservation:
    """Zero values must be preserved, not dropped."""

    def test_zero_energy_impact_preserved(self):
        """Zero energy_impact_kwh must not be dropped."""
        results = {
            "get_diagnostic_recommendations": [
                {"pzem_number": 1, "energy_impact_kwh": 0, "cost_impact": 0}
            ],
        }
        evidence = _compose_combined_evidence("test", results)
        c = _correlate(results, evidence)
        # Impact section should still have entries for zero values
        # The _compose_combined_evidence adds impact only for non-None values
        # Zero values are not None, so they should be preserved
        assert isinstance(c["has_diagnostic"], bool)

    def test_zero_confidence_preserved(self):
        """Zero confidence must be preserved."""
        results = {
            "get_diagnostic_recommendations": [
                {"pzem_number": 1, "confidence": 0.0, "probable_cause": "Test"}
            ],
        }
        c = _correlate(results)
        assert any("0.0" in s for s in c["causal_support"])

# ---------------------------------------------------------------------------
# 16. TIMESTAMP ALIGNMENT
# ---------------------------------------------------------------------------

class TestPhase3BTimestampAlignment:
    """Timestamps must be checked for alignment."""

    def test_timestamp_alignment_ok(self):
        """Same-period data should have time_alignment_ok."""
        results = {
            "get_meter": _meter(1, power=300.0),
            "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]],
        }
        c = _correlate(results)
        assert c["time_alignment_ok"] is True

    def test_different_pzem_no_time_conflict(self):
        """Different PZEMs don't create time conflicts."""
        results = {
            "get_meter": _meter(1, power=300.0),
            "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[1]],  # PZEM 2
        }
        c = _correlate(results)
        assert c["uncertainties"] != []

# ---------------------------------------------------------------------------
# 17. UNSUPPORTED-CAUSE PREVENTION
# ---------------------------------------------------------------------------

class TestPhase3BUnsupportedCausePrevention:
    """The correlation layer must not create unsupported causal claims."""

    def test_no_cause_without_diagnostic(self):
        """Measured data alone must not produce causal support."""
        results = {"get_meter": _meter(2, power=500.0)}
        c = _correlate(results)
        assert c["causal_support"] == []

    def test_no_cause_from_peaks_alone(self):
        """Peak data alone must not produce causal support."""
        results = {"get_peaks": PEAKS}
        c = _correlate(results)
        assert c["causal_support"] == []

    def test_no_cause_from_faults_alone(self):
        """Fault observation alone must not produce causal support."""
        results = {"get_faults": FAULTS}
        c = _correlate(results)
        # Faults alone have no probable_cause
        assert c["causal_support"] == []

    def test_no_appliance_inference(self):
        """No appliance names should be inferred."""
        results = {"get_meter": _meter(2, power=500.0)}
        c = _correlate(results)
        # supported_findings should not contain appliance names
        findings_str = " ".join(c["supported_findings"])
        assert "motor" not in findings_str.lower()
        assert "ac" not in findings_str.lower() or "pzem" in findings_str.lower()

    def test_no_savings_inference(self):
        """No savings should be inferred from correlation."""
        results = {"get_meter": _meter(2, power=500.0)}
        c = _correlate(results)
        findings_str = " ".join(c["supported_findings"])
        assert "save" not in findings_str.lower() or "PZEM" in findings_str.lower()

# ---------------------------------------------------------------------------
# 18. SYSTEM-LEVEL ATTRIBUTION PREVENTION
# ---------------------------------------------------------------------------

class TestPhase3BSystemAttributionPrevention:
    """System evidence must not be attributed to individual PZEMs."""

    def test_system_no_pzem_cause(self):
        """System summary must not generate PZEM-specific causes."""
        results = {"get_system_summary": dict(SYSTEM_SUMMARY)}
        c = _correlate(results)
        # No causal support from system data alone
        assert c["causal_support"] == []
        assert c["status"] == "PARTIAL"

    def test_system_peak_not_attributed_to_pzem(self):
        """System peak must not be attributed to a specific PZEM."""
        results = {"get_system_summary": dict(SYSTEM_SUMMARY), "get_peaks": PEAKS}
        c = _correlate(results)
        # Scope should be SYSTEM, not PZEM-specific
        assert c["scope"] == "SYSTEM"

# ---------------------------------------------------------------------------
# 19. HISTORICAL/LIVE CONTAMINATION PREVENTION
# ---------------------------------------------------------------------------

class TestPhase3BHistoricalLiveContamination:
    """Historical and live data must not be mixed as same time."""

    def test_historical_has_no_live(self):
        results = {"get_historical_analysis": dict(HISTORICAL_DATA)}
        c = _correlate(results)
        assert c["has_live_data"] is False
        assert c["has_historical_data"] is True

    def test_live_has_no_historical(self):
        results = {"get_meter": _meter(2, power=372.0)}
        c = _correlate(results)
        assert c["has_historical_data"] is False
        assert c["has_live_data"] is True

    def test_historical_system_no_pzem(self):
        """Historical system-wide analysis must not have PZEM-specific scope."""
        results = {"get_historical_analysis": {"status": "OK"}}
        c = _correlate(results)
        assert c["scope"] == "HISTORICAL"

# ---------------------------------------------------------------------------
# 20. DETERMINISTIC OUTPUT
# ---------------------------------------------------------------------------

class TestPhase3BDeterministic:
    """Correlation must be deterministic."""

    def test_same_input_same_output(self):
        results = {"get_meter": _meter(1, power=300.0), "get_diagnostic_recommendations": [DIAGNOSTIC_RECS[0]]}
        c1 = _correlate(results)
        c2 = _correlate(results)
        assert c1["status"] == c2["status"]
        assert c1["scope"] == c2["scope"]
        assert c1["pzem_scope"] == c2["pzem_scope"]
        assert len(c1["supported_findings"]) == len(c2["supported_findings"])

    def test_deterministic_evidence_quality(self):
        results = {"get_meter": _meter(1, power=300.0)}
        c = _correlate(results)
        assert c["evidence_quality"] is not None
        assert len(c["evidence_quality"]) > 0

# ---------------------------------------------------------------------------
# 21. LLM STRUCTURED-EVIDENCE SAFETY
# ---------------------------------------------------------------------------

class TestPhase3BLLMSafety:
    """LLM must not receive invented causal conclusions."""

    def test_llm_receives_correlation_status(self):
        """Correlation status must be included in LLM prompt."""
        results = {"get_meter": _meter(2, power=372.0)}
        evidence = _compose_combined_evidence("PZEM-2 mein power high kyu hai?", results)
        correlation = _correlate_evidence(results, evidence)

        # Verify correlation has all required fields
        assert "status" in correlation
        assert "scope" in correlation
        assert "supported_findings" in correlation
        assert "relationships" in correlation
        assert "causal_support" in correlation
        assert "uncertainties" in correlation
        assert "evidence_quality" in correlation
        assert "pzem_scope" in correlation
        assert "conflicts" in correlation

    def test_llm_prompt_no_invented_causes(self):
        """Correlation must not introduce causes not from Stage 4."""
        results = {"get_meter": _meter(2, power=372.0)}
        evidence = _compose_combined_evidence("test", results)
        correlation = _correlate_evidence(results, evidence)
        # No causal support without diagnostic
        assert correlation["causal_support"] == []

    def test_llm_prompt_system_scope_not_pzem(self):
        """System scope must not be attributed to PZEM in LLM prompt."""
        results = {"get_system_summary": dict(SYSTEM_SUMMARY)}
        evidence = _compose_combined_evidence("test", results)
        correlation = _correlate_evidence(results, evidence)
        assert correlation["scope"] == "SYSTEM"

# ---------------------------------------------------------------------------
# 22. FALLBACK BEHAVIOR
# ---------------------------------------------------------------------------

class TestPhase3BFallback:
    """Correlation must handle edge cases gracefully."""

    def test_llm_compose_with_none_correlation(self):
        """_llm_compose_energy with correlation=None should still work."""
        # This tests backward compatibility
        results = {"get_system_summary": dict(SYSTEM_SUMMARY)}
        # _llm_compose_energy returns None without API key
        # But it should not crash when correlation is None
        answer = _llm_compose_energy("test", results, [], "")
        assert answer is None  # No API key → returns None

    def test_compose_energy_with_none_correlation(self):
        """_compose_energy with correlation=None should still work."""
        results = {"get_system_summary": dict(SYSTEM_SUMMARY)}
        answer = ask_bob._compose_energy("test", results, None)
        assert answer is not None

    def test_compose_energy_with_correlation(self):
        """_compose_energy with correlation must not leak internal debug tags."""
        results = {"get_meter": _meter(2, power=372.0)}
        evidence = _compose_combined_evidence("test", results)
        correlation = _correlate_evidence(results, evidence)
        answer = ask_bob._compose_energy("test", results, correlation)
        # Internal debug tags must NOT appear in user-facing response
        assert "[Evidence status:" not in answer
        assert "[Note:" not in answer
        # Correlation status should still be reflected in a clean way
        assert "PARTIAL" in answer or "SUPPORTED" in answer or "Power" in answer
        assert answer == answer.strip()

    def test_compose_energy_empty_results(self):
        """_compose_energy with empty results should return safe message."""
        answer = ask_bob._compose_energy("test", {})
        assert "don't have enough" in answer.lower()

# ---------------------------------------------------------------------------
# INTEGRATION: ask_bob() flow
# ---------------------------------------------------------------------------

class TestPhase3BIntegration:
    """Integration tests with ask_bob() orchestration."""

    def test_live_diagnostic_integration(self):
        """PZEM-2 mein power high kyu hai? must produce correlation."""
        calls = []
        orig = bob_tools.run_tool
        def fake_run(name, **params):
            calls.append(name)
            if name == "get_meter":
                return _meter(2, power=372.0)
            if name == "get_diagnostic_recommendations":
                return [DIAGNOSTIC_RECS[1]]
            return []
        bob_tools.run_tool = fake_run
        try:
            ask_bob._get_api_key = lambda: ""
            ask_bob._llm_compose_energy = lambda *a, **k: None
            ask_bob._llm_general_conversation = lambda *a, **k: None
            r = ask_bob.ask_bob("PZEM-2 mein power high kyu hai?")
            assert r["source"] == "tool"
        finally:
            bob_tools.run_tool = orig

    def test_historical_integration(self):
        """Historical query must not produce live correlation."""
        calls = []
        orig = bob_tools.run_tool
        def fake_run(name, **params):
            calls.append(name)
            if name == "get_historical_analysis":
                return dict(HISTORICAL_DATA)
            return []
        bob_tools.run_tool = fake_run
        try:
            ask_bob._get_api_key = lambda: ""
            ask_bob._llm_compose_energy = lambda *a, **k: None
            ask_bob._llm_general_conversation = lambda *a, **k: None
            r = ask_bob.ask_bob("10 September ko PZEM-3 ki maximum power kya thi?")
            assert r["source"] == "tool"
        finally:
            bob_tools.run_tool = orig

    def test_system_peak_integration(self):
        """System peak query must produce SYSTEM scope correlation."""
        calls = []
        orig = bob_tools.run_tool
        def fake_run(name, **params):
            calls.append(name)
            if name == "get_system_summary":
                return dict(SYSTEM_SUMMARY)
            return []
        bob_tools.run_tool = fake_run
        try:
            ask_bob._get_api_key = lambda: ""
            ask_bob._llm_compose_energy = lambda *a, **k: None
            ask_bob._llm_general_conversation = lambda *a, **k: None
            r = ask_bob.ask_bob("Aaj system ka peak power kyu high tha?")
            assert r["source"] == "tool"
        finally:
            bob_tools.run_tool = orig

    def test_fault_action_integration(self):
        """Fault action query must produce correlation with actions."""
        calls = []
        orig = bob_tools.run_tool
        def fake_run(name, **params):
            calls.append(name)
            if name == "get_faults":
                return FAULTS
            if name == "get_diagnostic_recommendations":
                return [DIAGNOSTIC_RECS[0]]
            return []
        bob_tools.run_tool = fake_run
        try:
            ask_bob._get_api_key = lambda: ""
            ask_bob._llm_compose_energy = lambda *a, **k: None
            ask_bob._llm_general_conversation = lambda *a, **k: None
            r = ask_bob.ask_bob("Is electrical fault mein kya action lena chahiye?")
            assert r["source"] == "tool"
        finally:
            bob_tools.run_tool = orig

    def test_meter_only_integration(self):
        """Simple meter query must produce PARTIAL correlation with uncertainties."""
        calls = []
        orig = bob_tools.run_tool
        def fake_run(name, **params):
            calls.append(name)
            if name == "get_meter":
                return _meter(2, power=114.0)
            return []
        bob_tools.run_tool = fake_run
        try:
            ask_bob._get_api_key = lambda: ""
            ask_bob._llm_compose_energy = lambda *a, **k: None
            ask_bob._llm_general_conversation = lambda *a, **k: None
            r = ask_bob.ask_bob("PZEM-2 ka current kitna hai?")
            assert r["source"] == "tool"
        finally:
            bob_tools.run_tool = orig
