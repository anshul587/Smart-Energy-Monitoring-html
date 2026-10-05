"""Regression tests for the BOB recent-fault path.

Covers the four defects found in the master audit:
  1. Unix-SECONDS timestamps were rendered through the millisecond formatter,
     so 2026 fault data displayed as 1970-01-21.
  2. Recent-fault questions never requested maintenance context.
  3. BOB could not resolve a mapped load name ("Fan 1") to its PZEM.
  4. Missing diagnostic/maintenance context was left to improvisation instead of
     being stated explicitly.

Run: python -m pytest ai-backend/tests/test_bob_recent_faults.py -q
"""
from __future__ import annotations

import pytest

from ai import ask_bob, bob_tools
from ai import mapping

# 2026-09-10 11:33:20 UTC, Unix seconds - the unit every BOB source uses.
TS_2026 = 1789040000
TS_2026_MS = TS_2026 * 1000


@pytest.fixture(autouse=True)
def _mapped():
    """Central mapping fixture: pzem_1 -> Fan 1 / Classroom, nothing else."""
    mapping._REMOTE = {"pzem_1": {"load_name": "Fan 1", "location": "Classroom"}}
    mapping._REMOTE_LOADED = True
    yield
    mapping._REMOTE = {}
    mapping._REMOTE_LOADED = True


def _faults():
    return [
        {"pzem_number": 1, "fault_type": "power_factor_drop", "severity": "WARNING",
         "measured_value": 0.62, "timestamp": TS_2026},
        {"pzem_number": 2, "fault_type": "overvoltage", "severity": "EMERGENCY",
         "measured_value": 265.0, "timestamp": TS_2026},
    ]


def _diagnostics():
    return [{
        "pzem_number": 1, "fault_type": "power_factor_drop", "severity": "WARNING",
        "timestamp": TS_2026,
        "probable_cause": "Load drawing lagging power factor",
        "confidence": 0.72,
        "what_to_check": "Check capacitor bank terminals",
        "what_to_do_now": "Log PF at the meter for 30 minutes",
        "urgency": "MEDIUM", "maintenance_required": False,
    }]


def _maintenance():
    return [{"pzem_number": 1, "risk_level": "WATCH", "timestamp": TS_2026}]


def _results(faults=None, diag=None, maint=None):
    """Plain tool data, as _data_only() hands it to the composers."""
    out = {
        "get_faults": _faults() if faults is None else faults,
        "get_diagnostic_recommendations": _diagnostics() if diag is None else diag,
        "get_maintenance": _maintenance() if maint is None else maint,
    }
    return {k: v for k, v in out.items() if v}


def _answer(results, question="Any recent faults?"):
    results = {k: v for k, v in results.items() if ask_bob._has_data(v)}
    evidence = ask_bob._compose_combined_evidence(question, results)
    correlation = ask_bob._correlate_evidence(results, evidence)
    return ask_bob._compose_energy(question, results, correlation), evidence, correlation


# ---------------------------------------------------------------------------
# 1. TIMESTAMP UNIT
# ---------------------------------------------------------------------------

class TestTimestampUnit:
    def test_seconds_render_2026_not_1970(self):
        assert ask_bob._fmt_ts_s(TS_2026) == "2026-09-10 11:33 UTC"

    def test_fault_timestamp_seconds(self):
        out = ask_bob._render_faults(_faults(), "Any recent faults?")
        assert "2026-09-10 11:33 UTC" in out
        assert "1970" not in out

    def test_evidence_fault_timestamp_seconds(self):
        evidence = ask_bob._compose_combined_evidence("Any recent faults?", _results())
        observed = " ".join(evidence["observed"])
        assert "2026-09-10 11:33 UTC" in observed
        assert "1970" not in observed

    def test_no_double_conversion(self):
        """A seconds value must never be divided by 1000 anywhere in the path."""
        out, evidence, _ = _answer(_results())
        assert "1970" not in out
        assert "1970" not in " ".join(evidence["observed"])

    def test_millisecond_helper_still_correct_for_ms(self):
        """_fmt_ts keeps its documented millisecond contract."""
        assert ask_bob._fmt_ts(TS_2026_MS) == "2026-09-10 11:33 UTC"

    def test_no_render_path_uses_the_ms_formatter(self):
        """Guard: no BOB data source supplies milliseconds."""
        import inspect
        src = inspect.getsource(ask_bob)
        body = src.split("def _fmt_ts_s", 1)[1]
        assert "_fmt_ts(" not in body, "a BOB renderer is formatting seconds with the ms helper"

    def test_anomaly_and_peak_timestamps_seconds(self):
        anomalies = [{"pzem_number": 1, "anomaly_label": "spike", "timestamp": TS_2026}]
        peaks = [{"pzem_number": None, "total_peak_power_w": 900.0, "timestamp": TS_2026}]
        res = {"get_anomalies": anomalies, "get_peaks": peaks}
        evidence = ask_bob._compose_combined_evidence("any anomalies?", res)
        observed = " ".join(evidence["observed"])
        assert "1970" not in observed
        assert "2026-09-10 11:33 UTC" in observed


# ---------------------------------------------------------------------------
# 2. RECENT FAULT -> DIAGNOSTIC / MAINTENANCE ROUTING
# ---------------------------------------------------------------------------

class TestRecentFaultRouting:
    @pytest.mark.parametrize("q", [
        "Any recent faults?",
        "Are there any recent faults?",
        "What are the recent faults?",
        "Recent faults kya hain?",
        "Abhi koi fault hai kya?",
        "PZEM-1 ka recent fault kya hai?",
    ])
    def test_recent_fault_selects_fault_tool(self, q):
        tools = [t for t, _ in ask_bob._select_tools(q, [])]
        assert "get_faults" in tools

    @pytest.mark.parametrize("q", [
        "Any recent faults?",
        "Abhi koi fault hai kya?",
        "PZEM-1 ka recent fault kya hai?",
    ])
    def test_recent_fault_selects_diagnostic_and_maintenance(self, q):
        tools = [t for t, _ in ask_bob._select_tools(q, [])]
        assert "get_diagnostic_recommendations" in tools
        assert "get_maintenance" in tools

    def test_recent_fault_does_not_pull_historical_tool(self):
        tools = [t for t, _ in ask_bob._select_tools("Any recent faults?", [])]
        assert "get_historical_analysis" not in tools

    def test_historical_fault_does_not_pull_maintenance(self):
        """A historical fault question must not mix in current maintenance state."""
        q = "PZEM-1 par 10 September ko kaunsa fault tha?"
        tools = [t for t, _ in ask_bob._select_tools(q, [])]
        assert "get_faults" in tools
        assert "get_maintenance" not in tools

    def test_casual_fault_question_not_over_selected(self):
        tools = [t for t, _ in ask_bob._select_tools("Any faults?", [])]
        assert set(tools) <= {"get_faults", "get_diagnostic_recommendations", "get_maintenance"}


# ---------------------------------------------------------------------------
# 3. DIAGNOSTIC SAFETY (no fabrication)
# ---------------------------------------------------------------------------

class TestDiagnosticSafety:
    def test_only_supplied_fields_are_used(self):
        _, evidence, _ = _answer(_results(diag=_diagnostics()))
        probable = " ".join(evidence["probable"])
        action = " ".join(evidence["action"])
        assert "Load drawing lagging power factor" in probable
        assert "confidence = 0.72" in probable
        assert "Check capacitor bank terminals" in action
        assert "Log PF at the meter for 30 minutes" in action
        # energy_impact_kwh / cost_impact absent from the record -> absent from evidence
        assert not any("kWh" in i for i in evidence["impact"])
        assert not evidence["impact"]

    def test_missing_fields_are_not_invented(self):
        sparse = [{"pzem_number": 1, "fault_type": "power_factor_drop",
                   "severity": "WARNING", "timestamp": TS_2026}]
        _, evidence, _ = _answer(_results(diag=sparse))
        assert evidence["probable"] == []
        assert evidence["action"] == []
        assert evidence["maintenance"] == []
        out, _, _ = _answer(_results(diag=sparse))
        for banned in ("capacitor", "Capacitor", "motor is faulty", "replace"):
            assert banned not in out

    def test_no_diagnostic_states_it_explicitly(self):
        out, _, _ = _answer(_results(diag=[], maint=[]))
        assert ("No matching diagnostic recommendation or maintenance information is "
                "currently available for these faults.") in out

    def test_diagnostic_without_maintenance_states_maintenance_missing(self):
        out, _, _ = _answer(_results(diag=_diagnostics(), maint=[]))
        assert "Maintenance information is currently unavailable for these faults." in out
        assert "Load drawing lagging power factor" in out

    def test_maintenance_without_diagnostic_states_diagnostic_missing(self):
        out, _, _ = _answer(_results(diag=[], maint=_maintenance()))
        assert ("No matching diagnostic recommendation is currently available "
                "for these faults.") in out
        assert "WATCH" in out

    def test_both_available_uses_both(self):
        out, _, _ = _answer(_results(diag=_diagnostics(), maint=_maintenance()))
        assert "Load drawing lagging power factor" in out
        assert "WATCH" in out
        assert "currently unavailable" not in out

    def test_no_knowledge_cutoff_language(self):
        out, _, _ = _answer(_results(diag=[]))
        low = out.lower()
        assert "knowledge cutoff" not in low
        assert "real-time access" not in low
        assert "i don't have real-time access" not in low

    def test_fault_type_and_severity_unchanged(self):
        out, _, _ = _answer(_results())
        assert "power_factor_drop" in out
        assert "overvoltage" in out
        assert "WARNING" in out
        assert "EMERGENCY" in out


# ---------------------------------------------------------------------------
# 4. PZEM LOAD MAPPING
# ---------------------------------------------------------------------------

class TestBobMapping:
    def test_load_name_resolves_to_pzem(self):
        assert ask_bob._pzem_from_text("Fan 1 ka current kitna hai?") == 1

    def test_pzem_query_still_works(self):
        assert ask_bob._pzem_from_text("PZEM-1 ka current?") == 1
        assert ask_bob._pzem_from_text("PZEM 1 ka power?") == 1
        assert ask_bob._pzem_from_text("PZEM_1 ka fault?") == 1
        assert ask_bob._pzem_from_text("meter 2 kitna use kar raha hai?") == 2

    def test_pzem_query_wins_over_load_name(self):
        assert ask_bob._pzem_from_text("PZEM-2 ka current, Fan 1 comparison?") == 2

    def test_unknown_load_name_does_not_guess(self):
        assert ask_bob._pzem_from_text("Chiller 7 ka current kitna hai?") is None

    def test_placeholder_names_do_not_match(self):
        """Unmapped meters keep their PZEM id as the label and never match by name."""
        assert ask_bob._pzem_from_load_name("PZEM-3 status") is None

    def test_load_name_query_selects_pzem_scoped_tools(self):
        plan = ask_bob._select_tools("Fan 1 ka current kitna hai?", [])
        assert ("get_meter", 1) in [(t, p.get("pzem_number")) for t, p in plan]

    def test_fault_output_shows_mapped_identity_and_keeps_pzem_id(self):
        out, _, _ = _answer(_results())
        assert "PZEM-1 (Fan 1 - Classroom)" in out
        assert "PZEM-2" in out

    def test_maintenance_output_shows_mapped_identity(self):
        _, evidence, _ = _answer(_results(maint=_maintenance()))
        assert "PZEM-1 (Fan 1 - Classroom)" in " ".join(evidence["observed"])

    def test_unmapped_meter_still_shows_pzem_id(self):
        assert "PZEM-2" in ask_bob._render_faults(_faults(), "Any recent faults?")
        assert "Fan" not in ask_bob._pz_label(2)

    def test_mapping_is_presentation_only(self):
        """Mapping helpers must not touch data, thresholds or Firebase writes."""
        assert ask_bob._pz_label(1).startswith("PZEM-1")
        faults = _faults()
        before = [dict(f) for f in faults]
        ask_bob._pz_label(1)
        assert faults == before


# ---------------------------------------------------------------------------
# 5. ALERT / FAULT / RECOMMENDATION SEPARATION
# ---------------------------------------------------------------------------

class TestSemanticSeparation:
    def test_fault_render_does_not_invent_recommendation(self):
        out = ask_bob._render_faults(_faults(), "Any recent faults?")
        assert "probable" not in out.lower()
        assert "recommend" not in out.lower()

    def test_recommendation_tool_not_selected_for_plain_fault_question(self):
        tools = [t for t, _ in ask_bob._select_tools("Any recent faults?", [])]
        assert "get_energy_saving" not in tools

    def test_no_alert_records_are_synthesised(self):
        """BOB has no alert tool: a fault never becomes an alert."""
        assert not any("alert" in t for t in bob_tools.available_tools())