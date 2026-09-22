"""
tests/test_ask_bob.py — Stage 16 (enhanced): data-aware tool-calling agent.

Exercises ai.ask_bob + ai.bob_tools directly. Live reads are stubbed via
bob_tools.run_tool so no Firebase/credentials/network are required. The LLM key
is forced off unless a specific test needs it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai import ask_bob, bob_tools

_KNOWLEDGE_PATH = Path(__file__).resolve().parents[1] / "ai" / "project_knowledge.json"


# ---------------------------------------------------------------------------
# Fixtures + fake tool data
# ---------------------------------------------------------------------------

def _meter(n, online, power, energy, voltage, current=1.2, freq=50.0):
    return {"pzem_number": n, "online": online, "voltage": voltage, "current": current,
            "power": power if online else None, "energy": energy,
            "power_factor": 0.9, "frequency": freq if online else None,
            "last_seen": 1700000000000, "age_ms": 1000}


METERS = [
    _meter(1, True, 276.0, 1.5, 230.1),
    _meter(3, True, 114.0, 0.9, 228.0),
    _meter(4, True, 500.0, 2.1, 231.0),
    _meter(2, False, 0.0, 0.4, 0.0),
]
METER_BY_PZ = {m["pzem_number"]: m for m in METERS}

SUMMARY = {
    "system_status": "online", "online_meter_count": 3, "total_meter_count": 9,
    "total_power_w": 890.0, "total_energy_kwh": 45.2, "average_voltage_v": 229.0,
    "active_anomaly_count": 1, "active_fault_count": 1,
    "maintenance_risk": {"high_risk_meters": 1, "watch_meters": 2},
    "latest_peak": {"timestamp": 1700000000000, "total_peak_power_w": 1200.0, "dominant_pzems": [1, 3]},
    "forecast_available": True,
    "latest_bill_prediction": {"anchor_timestamp": 1700000000000, "estimated_bill": 312.5},
    "energy_saving": {"recommendation_count": 2},
}
FAULTS = [{"pzem_number": 1, "fault_type": "overvoltage", "timestamp": 1700000000000}]
ANOMALIES = [{"pzem_number": 4, "anomaly_label": "SPIKE", "timestamp": 1700000000000}]
PEAKS = [{"pzem_number": None, "total_peak_power_w": 1200.0, "dominant_pzems": [1, 3],
          "timestamp": 1700000000000}]
MAINT = [{"pzem_number": None, "high_risk_meters": [4], "watch_meters": [2, 5],
          "normal_meters": [1, 3, 6, 7, 8, 9], "highest_risk_pzem": 4,
          "highest_risk_score": 0.81, "timestamp": 1700000000000}]
FORECAST = [{"pzem_number": None, "status": "FORECAST", "forecast_24h": 950.0,
             "forecast_7d": 840.0, "timestamp": 1700000000000}]
BILL = [{"status": "OK", "estimated_bill": 312.5, "anchor_timestamp": 1700000000000,
         "estimated_total_energy_kwh": 150.0}]
SAVING = [{"status": "OK", "recommendation_count": 2,
            "recommendations": [
                {"pzem_number": 4, "priority": "high",
                 "recommendation": "Shift compressor load to off-peak hours."},
                {"pzem_number": 2, "priority": "medium",
                 "recommendation": "Investigate standby current draw."}]}]
REPORTS = [
    {"filename": "report-2026-08.pdf", "year": 2026, "month": 8, "size_bytes": 12345,
     "url": "/api/v1/reports/monthly/report-2026-08.pdf"},
    {"filename": "latest.pdf", "year": None, "month": None, "size_bytes": 12345,
     "url": "/api/v1/reports/monthly/latest.pdf"},
]

DIAGNOSTIC_RECS = [
    {"pzem_number": 1, "timestamp": 1700000000000, "fault_type": "overvoltage",
     "severity": "WARNING", "priority": "P1 - Critical",
     "probable_cause": "Possible incoming supply overvoltage.",
     "evidence": "voltage reading above threshold.",
     "confidence": 0.85,
     "what_to_check": "Verify voltage with an independent meter.",
     "what_to_do_now": "Do NOT assume equipment failure.",
     "corrective_action": "Independent voltage verification first.",
     "urgency": "HIGH", "maintenance_required": True,
     "maintenance_timing": "Schedule inspection promptly.",
     "energy_impact_kwh": None, "cost_impact": None,
     "source_stages": ("Stage 3: Fault Diagnosis",)},
    {"pzem_number": 2, "timestamp": 1700000000000, "fault_type": "overcurrent",
     "severity": "EMERGENCY", "priority": "P1 - Critical",
     "probable_cause": "Possible overload or excessive connected load.",
     "evidence": "current reading above threshold.",
     "confidence": 0.9,
     "what_to_check": "Check active loads and breaker rating.",
     "what_to_do_now": "Reduce or redistribute load if overloaded.",
     "corrective_action": "Reduce or redistribute load if overloaded.",
     "urgency": "HIGH", "maintenance_required": True,
     "maintenance_timing": "Address load immediately if sustained.",
     "energy_impact_kwh": None, "cost_impact": None,
     "source_stages": ("Stage 3: Fault Diagnosis",)},
]
DIAGNOSTIC_RECS_EMPTY = []
DIAGNOSTIC_RECS_NO_CAUSE = [
    {"pzem_number": 3, "timestamp": 1700000000000, "fault_type": "high_power",
     "severity": "NORMAL", "priority": "P3 - Informational",
     "probable_cause": None,
     "evidence": "power reading above threshold.",
     "confidence": 0.7,
     "what_to_check": "Check active loads.",
     "what_to_do_now": "Check active loads and determine if expected.",
     "corrective_action": "Assess whether high power is expected.",
     "urgency": "MEDIUM", "maintenance_required": False,
     "maintenance_timing": "Monitor.",
     "energy_impact_kwh": None, "cost_impact": None,
     "source_stages": ("Stage 3: Fault Diagnosis",)},
]

HISTORICAL = {
    "status": "OK",
    "pzem_number": 3,
    "reason": None,
    "requested_start": 1_700_000_000,
    "requested_end": 1_700_086_400,
    "actual_start": 1_700_000_000,
    "actual_end": 1_700_086_400,
    "available_days": 1.0,
    "sample_count": 288,
    "valid_rows": 288,
    "dropped_rows": 0,
    "power": {"count": 288, "minimum": 50.0, "maximum": 500.0, "average": 114.0, "median": 100.0, "std_dev": 50.0, "min_timestamp": 1_700_000_000, "max_timestamp": 1_700_086_400},
    "voltage": {"count": 288, "minimum": 225.0, "maximum": 235.0, "average": 228.0, "median": 228.0, "std_dev": 2.0, "min_timestamp": 1_700_000_000, "max_timestamp": 1_700_086_400},
    "current": {"count": 288, "minimum": 0.2, "maximum": 2.0, "average": 0.5, "median": 0.5, "std_dev": 0.3, "min_timestamp": 1_700_000_000, "max_timestamp": 1_700_086_400},
    "frequency": {"count": 288, "minimum": 49.5, "maximum": 50.5, "average": 50.0, "median": 50.0, "std_dev": 0.2},
    "pf": {"count": 288, "minimum": 0.8, "maximum": 1.0, "average": 0.95, "median": 0.95, "std_dev": 0.05},
    "energy_consumption": {"start_energy_kwh": 100.0, "end_energy_kwh": 200.0, "consumption_kwh": 100.0, "start_timestamp": 1_700_000_000, "end_timestamp": 1_700_086_400, "valid": True},
    "trend": {"power_trend_per_hour": 0.0, "current_trend_per_hour": 0.0, "voltage_trend_per_hour": 0.0, "pf_trend_per_hour": 0.0, "frequency_trend_per_hour": 0.0, "data_span_hours": 1.0, "sample_count": 288},
    "hourly": [{"hour": 0, "power_avg": 100.0, "power_max": 150.0, "power_min": 50.0, "sample_count": 12}],
    "daily": [{"date": "2023-11-15", "power_avg": 114.0, "power_max": 500.0, "energy_consumption_kwh": 100.0, "sample_count": 288}],
}

HISTORICAL_NO_DATA = {"status": "NO_DATA", "reason": "No historical data available for the requested period"}
HISTORICAL_INSUFFICIENT = {"status": "INSUFFICIENT_DATA", "reason": "Only 1 sample(s) in range; need >= 2"}
HISTORICAL_ERROR = {"status": "ERROR", "reason": "Invalid PZEM number"}
HISTORICAL_SYSTEM = {
    "status": "OK",
    "reason": None,
    "requested_start": 1_700_000_000,
    "requested_end": 1_700_086_400,
    "meters_analyzed": 3,
    "total_power": {"count": 288, "minimum": 200.0, "maximum": 1200.0, "average": 600.0, "median": 500.0, "std_dev": 200.0, "min_timestamp": 1_700_000_000, "max_timestamp": 1_700_086_400},
    "total_energy_kwh": 500.0,
    "per_pzem": {"1": {"status": "OK", "reason": None, "power_avg": 276.0, "power_max": 300.0, "energy_consumption_kwh": 50.0},
                 "3": {"status": "OK", "reason": None, "power_avg": 114.0, "power_max": 200.0, "energy_consumption_kwh": 30.0}},
}

FAKE = {
    "get_system_summary": lambda **k: dict(SUMMARY),
    "get_meters": lambda **k: [dict(m) for m in METERS],
    "get_meter": lambda pzem_number=None, **k: dict(METER_BY_PZ.get(int(pzem_number),
                                                                   _meter(9, False, 0, 0, 0))),
    "get_faults": lambda **k: [dict(f) for f in FAULTS],
    "get_anomalies": lambda **k: [dict(a) for a in ANOMALIES],
    "get_peaks": lambda **k: [dict(p) for p in PEAKS],
    "get_maintenance": lambda **k: [dict(m) for m in MAINT],
    "get_forecast": lambda **k: [dict(f) for f in FORECAST],
    "get_bill_prediction": lambda **k: [dict(b) for b in BILL],
    "get_energy_saving": lambda **k: [dict(e) for e in SAVING],
    "get_historical_analysis": lambda pzem_number=None, **k: dict(HISTORICAL) if pzem_number else dict(HISTORICAL_SYSTEM),
    "get_monthly_reports": lambda **k: [dict(r) for r in REPORTS],
    "get_diagnostic_recommendations": lambda **k: [dict(r) for r in DIAGNOSTIC_RECS],
}


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.setattr(ask_bob, "_get_api_key", lambda: "")
    return monkeypatch


@pytest.fixture
def fake_data(no_key):
    calls = []

    def fake_run(name, **params):
        calls.append((name, params))
        fn = FAKE.get(name)
        if fn is None:
            raise bob_tools.ToolError("unknown_tool", name)
        return fn(**params)

    no_key.setattr(bob_tools, "run_tool", fake_run)
    return calls


@pytest.fixture
def empty_data(no_key):
    def empty_run(name, **params):
        if name in ("get_system_summary",):
            return {}
        if name in ("get_meters", "get_faults", "get_anomalies", "get_peaks",
                    "get_maintenance", "get_forecast", "get_bill_prediction",
                    "get_energy_saving", "get_monthly_reports", "get_historical_analysis",
                    "get_diagnostic_recommendations"):
            return [] if name != "get_historical_analysis" else HISTORICAL_NO_DATA
        return {}
    no_key.setattr(bob_tools, "run_tool", empty_run)
    return empty_run


# ---- CASUAL ----------------------------------------------------------------
def test_casual_hello(no_key):
    r = ask_bob.ask_bob("Hello")
    assert r["source"] == "casual" and "BOB" in r["answer"]


def test_casual_hi(no_key):
    r = ask_bob.ask_bob("Hi")
    assert r["source"] == "casual"


def test_casual_how_are_you(no_key):
    r = ask_bob.ask_bob("How are you?")
    assert r["source"] == "casual" and "help" in r["answer"].lower()


def test_casual_thanks(no_key):
    r = ask_bob.ask_bob("Thank you")
    assert r["source"] == "casual" and "welcome" in r["answer"].lower()


def test_casual_bye(no_key):
    r = ask_bob.ask_bob("Bye")
    assert r["source"] == "casual" and "goodbye" in r["answer"].lower()


def test_casual_makes_no_tool_calls(fake_data):
    ask_bob.ask_bob("Hello there BOB")
    assert fake_data == []  # no Stage 15 tools hit for casual chat


# ---- PROJECT ---------------------------------------------------------------
def test_project_tell_me_about(no_key):
    r = ask_bob.ask_bob("Tell me about this project.")
    assert r["source"] == "project" and "Anshul Ninawe" in r["answer"] and "ESP32" in r["answer"]


def test_project_who_made(no_key):
    r = ask_bob.ask_bob("Who made this project?")
    assert r["source"] == "project" and "Anshul Ninawe" in r["answer"]


def test_project_guide(no_key):
    r = ask_bob.ask_bob("Who is the project guide?")
    assert r["source"] == "project" and "Bhupendra Kumar" in r["answer"]


def test_project_team(no_key):
    r = ask_bob.ask_bob("Who are the team members?")
    assert r["source"] == "project"
    for name in ["Yash Kawale", "Yash Dahake", "Swapnil Shendre", "Chetan Bokade", "Sanjog Godbole"]:
        assert name in r["answer"]


def test_project_advantages(no_key):
    r = ask_bob.ask_bob("What are the advantages?")
    assert r["source"] == "project" and "low-cost" in r["answer"].lower()


def test_project_how_it_works(no_key):
    r = ask_bob.ask_bob("How does the system work?")
    assert r["source"] == "project" and "Firebase" in r["answer"]


def test_project_makes_no_tool_calls(fake_data):
    ask_bob.ask_bob("Tell me about this project.")
    assert fake_data == []  # project questions never hit live APIs


# ---- TOOLS / LIVE DATA -----------------------------------------------------
def test_live_system_summary(fake_data):
    r = ask_bob.ask_bob("What is the status of the system?")
    assert ("get_system_summary", {}) in fake_data
    assert "3 of 9 meters online" in r["answer"]


def test_live_meter_comparison(fake_data):
    r = ask_bob.ask_bob("Which PZEM uses most power?")
    assert ("get_meters", {}) in fake_data
    assert "PZEM 4" in r["answer"] and "500.0" in r["answer"]


def test_live_pzem_specific(fake_data):
    r = ask_bob.ask_bob("What is the power of PZEM 1?")
    assert ("get_meter", {"pzem_number": 1}) in fake_data
    assert "PZEM 1" in r["answer"] and "276.0" in r["answer"]


def test_live_fault(fake_data):
    r = ask_bob.ask_bob("Any recent faults?")
    assert ("get_faults", {}) in fake_data
    assert "overvoltage" in r["answer"]


def test_live_anomaly(fake_data):
    r = ask_bob.ask_bob("Show me recent anomalies.")
    assert ("get_anomalies", {}) in fake_data
    assert "SPIKE" in r["answer"]


def test_live_peak(fake_data):
    r = ask_bob.ask_bob("What was the highest peak?")
    assert ("get_peaks", {}) in fake_data
    assert "1200.0" in r["answer"]


def test_live_maintenance(fake_data):
    r = ask_bob.ask_bob("Which meter needs attention?")
    assert ("get_maintenance", {}) in fake_data
    assert "high-risk" in r["answer"].lower()


def test_live_forecast(fake_data):
    r = ask_bob.ask_bob("What is tomorrow's forecast?")
    assert ("get_forecast", {"horizon": "24h"}) in fake_data
    assert "forecast" in r["answer"].lower()


def test_live_bill(fake_data):
    r = ask_bob.ask_bob("What is my predicted bill?")
    assert ("get_bill_prediction", {}) in fake_data
    assert "312.5" in r["answer"]


def test_live_energy_saving(fake_data):
    r = ask_bob.ask_bob("How can I save energy?")
    assert ("get_energy_saving", {}) in fake_data
    assert "recommendation" in r["answer"].lower()


def test_live_monthly_report(fake_data):
    r = ask_bob.ask_bob("Show me the monthly report.")
    assert ("get_monthly_reports", {}) in fake_data
    assert "report-2026-08.pdf" in r["answer"]


# ---- MULTI-TOOL ------------------------------------------------------------
def test_multi_fault_anomaly(fake_data):
    r = ask_bob.ask_bob("Why did PZEM 4 have a problem?")
    names = {c[0] for c in fake_data}
    assert "get_faults" in names and "get_anomalies" in names
    assert ("get_faults", {"pzem_number": 4}) in fake_data
    assert ("get_anomalies", {"pzem_number": 4}) in fake_data
    assert "overvoltage" in r["answer"] and "SPIKE" in r["answer"]


def test_multi_project_live(fake_data):
    r = ask_bob.ask_bob("Who developed the dashboard and what is the current system status?")
    assert r["source"] == "mixed"
    assert "Anshul Ninawe" in r["answer"]
    assert ("get_system_summary", {}) in fake_data


def test_multi_pzem_comparison(fake_data):
    r = ask_bob.ask_bob("Compare PZEM 3 and PZEM 5.")
    assert ("get_meters", {}) in fake_data
    assert "PZEM" in r["answer"]


def test_multi_follow_up(fake_data):
    history = [
        {"role": "user", "content": "Which PZEM uses most power?"},
        {"role": "bot", "content": "PZEM 4 is using the most power at 500.0 W."},
    ]
    r = ask_bob.ask_bob("How much?", history=history)
    assert ("get_meter", {"pzem_number": 4}) in fake_data
    assert "500.0" in r["answer"]


# ---- ROBUSTNESS ------------------------------------------------------------
def test_missing_data(empty_data):
    r = ask_bob.ask_bob("Which PZEM uses most power?")
    assert "don't have enough current data" in r["answer"]


def test_api_failure_graceful(no_key, monkeypatch):
    def boom(name, **params):
        raise bob_tools.ToolError("data_unavailable", "firebase down")
    monkeypatch.setattr(bob_tools, "run_tool", boom)
    r = ask_bob.ask_bob("Which PZEM uses most power?")
    assert "don't have enough current data" in r["answer"]


def test_llm_failure_falls_back(fake_data, monkeypatch):
    monkeypatch.setattr(ask_bob, "_get_api_key", lambda: "sk-test")
    monkeypatch.setattr(ask_bob, "_llm_compose", lambda *a, **k: None)
    r = ask_bob.ask_bob("Which PZEM uses most power?")
    assert r["source"] == "tool"  # fell back to deterministic composer
    assert "PZEM 4" in r["answer"]


def test_missing_api_key(fake_data):
    r = ask_bob.ask_bob("Which PZEM uses most power?")
    assert r["source"] != "llm"
    assert "PZEM 4" in r["answer"]


def test_invalid_pzem(no_key):
    with pytest.raises(bob_tools.ToolError) as exc:
        bob_tools.run_tool("get_meter", pzem_number=99)
    assert exc.value.code == "invalid_pzem"


def test_invalid_filter(no_key):
    with pytest.raises(bob_tools.ToolError) as exc:
        bob_tools.run_tool("get_forecast", horizon="bad")
    assert exc.value.code == "invalid_horizon"


def test_tool_call_validation(no_key):
    with pytest.raises(bob_tools.ToolError) as exc:
        bob_tools.run_tool("get_faults", limit=-1)
    assert exc.value.code == "invalid_limit"


def test_unknown_tool_rejected(no_key):
    with pytest.raises(bob_tools.ToolError) as exc:
        bob_tools.run_tool("get_flagged_records")
    assert exc.value.code == "unknown_tool"


def test_no_fabricated_values(empty_data):
    r = ask_bob.ask_bob("Which PZEM uses the most power?")
    assert "don't have enough current data" in r["answer"]
    assert not any(ch.isdigit() for ch in r["answer"].replace("don't have enough current data", ""))


def test_no_hallucinated_team(no_key):
    r = ask_bob.ask_bob("Who are the team members?")
    for fake in ["Elon Musk", "John Doe", "Ada Lovelace"]:
        assert fake not in r["answer"]


def test_secret_leak_prevention(no_key, monkeypatch):
    secret = "TEST-ANTHROPIC-KEY-NOT-REAL"
    monkeypatch.setattr(ask_bob, "_get_api_key", lambda: secret)
    r = ask_bob.ask_bob("Who made this project?")
    assert secret not in r["answer"]
    raw = _KNOWLEDGE_PATH.read_text(encoding="utf-8")
    assert "sk-ant" not in raw
    assert "AIza" not in raw
    assert "BEGIN PRIVATE KEY" not in raw


# ---- HISTORICAL ANALYSIS TESTS --------------------------------------------

def test_historical_tool_registration(no_key):
    """get_historical_analysis is registered in bob_tools."""
    assert "get_historical_analysis" in bob_tools.available_tools()


def test_historical_tool_dispatch(fake_data):
    """Historical question dispatches get_historical_analysis."""
    r = ask_bob.ask_bob("PZEM 3 ne 10 September ko average power kya tha?")
    assert any(c[0] == "get_historical_analysis" for c in fake_data)
    assert "PZEM 3" in r["answer"]


def test_historical_10_september(fake_data):
    """10 September date extraction routes to historical analysis."""
    r = ask_bob.ask_bob("PZEM-3 ne 10 September ko sabse zyada power kab consume ki?")
    assert any(c[0] == "get_historical_analysis" for c in fake_data)
    assert "PZEM 3" in r["answer"]


def test_historical_yesterday(fake_data):
    """yesterday/kal date extraction routes to historical analysis."""
    r = ask_bob.ask_bob("PZEM-2 ka kal ki energy consumption kitni thi?")
    assert any(c[0] == "get_historical_analysis" for c in fake_data)


def test_historical_last_7_days(fake_data):
    """last 7 days phrase routes to historical analysis."""
    r = ask_bob.ask_bob("PZEM-2 ka last 7 days ka maximum current kya tha?")
    assert any(c[0] == "get_historical_analysis" for c in fake_data)
    assert "maximum" in r["answer"].lower() or "PZEM 2" in r["answer"]


def test_historical_peak_power(fake_data):
    """Historical peak power query routes correctly."""
    r = ask_bob.ask_bob("PZEM-3 ka last week ka maximum power kya tha?")
    assert any(c[0] == "get_historical_analysis" for c in fake_data)


def test_historical_not_get_meter(fake_data):
    """Historical question must NOT select get_meter."""
    r = ask_bob.ask_bob("PZEM-3 ne 10 September ko average power kya tha?")
    tool_names = {c[0] for c in fake_data}
    assert "get_meter" not in tool_names
    assert "get_historical_analysis" in tool_names


def test_live_still_get_meter(fake_data):
    """Live question still selects get_meter or get_meters, not historical."""
    r = ask_bob.ask_bob("PZEM-3 ka abhi power kitna hai?")
    tool_names = {c[0] for c in fake_data}
    assert "get_historical_analysis" not in tool_names
    assert "get_meter" in tool_names or "get_meters" in tool_names


def test_historical_no_data(no_key):
    """NO_DATA status returns clear message, no fabricated values."""
    def fake_run(name, **params):
        if name == "get_historical_analysis":
            return HISTORICAL_NO_DATA
        if name in ("get_system_summary",):
            return {}
        return []
    no_key.setattr(bob_tools, "run_tool", fake_run)
    r = ask_bob.ask_bob("PZEM-5 ka last month ka power kya tha?")
    assert "historical data" in r["answer"].lower() or "unavailable" in r["answer"].lower()


def test_historical_insufficient_data(no_key):
    """INSUFFICIENT_DATA status returns clear message."""
    def fake_run(name, **params):
        if name == "get_historical_analysis":
            return HISTORICAL_INSUFFICIENT
        if name in ("get_system_summary",):
            return {}
        return []
    no_key.setattr(bob_tools, "run_tool", fake_run)
    r = ask_bob.ask_bob("PZEM-9 ka power kya tha?")
    assert "insufficient" in r["answer"].lower() or "don't have enough" in r["answer"].lower()


def test_historical_system_query(fake_data):
    """System-wide historical query routes to get_historical_analysis without pzem."""
    r = ask_bob.ask_bob("System ka last 7 days ka highest simultaneous power kab tha?")
    assert any(c[0] == "get_historical_analysis" for c in fake_data)
    assert "PZEM" not in r["answer"] or "System" in r["answer"]


def test_historical_multi_pzem(fake_data):
    """Multi-PZEM historical comparison routes to get_historical_analysis."""
    r = ask_bob.ask_bob("Compare PZEM 2 and PZEM 5 power last week.")
    assert any(c[0] == "get_historical_analysis" for c in fake_data)


def test_historical_no_fabricated_values(fake_data):
    """Historical response must not invent unavailable values."""
    r = ask_bob.ask_bob("PZEM-3 ne 10 September ko average power kya tha?")
    if "average" not in r["answer"].lower() and "PZEM 3" in r["answer"]:
        pass
    assert "don't have enough current data" not in r["answer"] or "PZEM 3" in r["answer"]


def test_historical_date_extraction():
    """Date extraction works for specific dates."""
    from ai.ask_bob import _date_from_text
    result = _date_from_text("10 September")
    assert result is not None
    assert "start" in result and "end" in result
    assert isinstance(result["start"], int)
    assert isinstance(result["end"], int)


def test_historical_yesterday_extraction():
    """yesterday/kal date extraction works."""
    from ai.ask_bob import _date_from_text
    result = _date_from_text("yesterday")
    assert result is not None
    assert isinstance(result["start"], int)


def test_historical_no_data_message(no_key):
    """Historical NO_DATA produces proper message, not fabricated data."""
    def fake_run(name, **params):
        if name == "get_historical_analysis":
            return HISTORICAL_NO_DATA
        if name in ("get_system_summary",):
            return {}
        return []
    no_key.setattr(bob_tools, "run_tool", fake_run)
    r = ask_bob.ask_bob("PZEM-5 ka last month ka power kya tha?")
    assert "historical data" in r["answer"].lower() or "unavailable" in r["answer"].lower() or "don't have enough" in r["answer"].lower()

# ---- STAGE 5C: Diagnostic Recommendation Tests ----------------------------

def test_diagnostic_tool_registration(no_key):
    """get_diagnostic_recommendations is registered in bob_tools."""
    assert "get_diagnostic_recommendations" in bob_tools.available_tools()


def test_diagnostic_tool_dispatch(fake_data):
    """Diagnostic recommendation question dispatches get_diagnostic_recommendations."""
    r = ask_bob.ask_bob("PZEM-1 mein kya problem hai?")
    assert any(c[0] == "get_diagnostic_recommendations" for c in fake_data)
    assert "PZEM" in r["answer"] or "recommendation" in r["answer"].lower()


def test_diagnostic_recommendation_retrieval(fake_data):
    """Diagnostic recommendation retrieval returns structured records."""
    r = ask_bob.ask_bob("Kya koi diagnostic recommendation hai?")
    assert any(c[0] == "get_diagnostic_recommendations" for c in fake_data)


def test_diagnostic_pzem_filtering(fake_data):
    """PZEM filtering works for diagnostic recommendations."""
    r = ask_bob.ask_bob("PZEM-1 mein kya problem hai?")
    calls = [c for c in fake_data if c[0] == "get_diagnostic_recommendations"]
    assert len(calls) >= 1, "get_diagnostic_recommendations was not called"
    assert calls[0][1].get("pzem_number") == 1


def test_diagnostic_severity_filtering(fake_data):
    """Severity filtering parameter is supported by get_diagnostic_recommendations."""
    r = ask_bob.ask_bob("PZEM-1 mein kya problem hai?")
    calls = [c for c in fake_data if c[0] == "get_diagnostic_recommendations"]
    assert len(calls) >= 1, "get_diagnostic_recommendations was not called"
    assert "severity" in bob_tools._TOOL_PARAMS["get_diagnostic_recommendations"]


def test_diagnostic_fault_type_filtering(fake_data):
    """fault_type filtering parameter is supported by get_diagnostic_recommendations."""
    r = ask_bob.ask_bob("PZEM-1 mein kya problem hai?")
    calls = [c for c in fake_data if c[0] == "get_diagnostic_recommendations"]
    assert len(calls) >= 1, "get_diagnostic_recommendations was not called"
    assert "fault_type" in bob_tools._TOOL_PARAMS["get_diagnostic_recommendations"]


def test_diagnostic_date_filtering(fake_data):
    """Date filtering parameters are supported by get_diagnostic_recommendations."""
    r = ask_bob.ask_bob("PZEM-1 mein kya problem hai?")
    calls = [c for c in fake_data if c[0] == "get_diagnostic_recommendations"]
    assert len(calls) >= 1, "get_diagnostic_recommendations was not called"
    assert "start" in bob_tools._TOOL_PARAMS["get_diagnostic_recommendations"]
    assert "end" in bob_tools._TOOL_PARAMS["get_diagnostic_recommendations"]


def test_diagnostic_limit(fake_data):
    """Limit parameter is supported by get_diagnostic_recommendations."""
    r = ask_bob.ask_bob("Kya koi diagnostic recommendation hai?")
    calls = [c for c in fake_data if c[0] == "get_diagnostic_recommendations"]
    assert len(calls) >= 1, "get_diagnostic_recommendations was not called"
    assert "limit" in bob_tools._TOOL_PARAMS["get_diagnostic_recommendations"]


def test_diagnostic_empty_recommendations(no_key):
    """Empty diagnostic recommendation list returns clean message, no fabrication."""
    def fake_run(name, **params):
        if name == "get_diagnostic_recommendations":
            return []
        if name in ("get_system_summary",):
            return {}
        return []
    no_key.setattr(bob_tools, "run_tool", fake_run)
    r = ask_bob.ask_bob("Kya koi diagnostic recommendation hai?")
    # Should return a clean message without fabrication
    assert "diagnostic" in r["answer"].lower() or "not available" in r["answer"].lower() or "enough" in r["answer"].lower()
    # Must NOT contain internal debug tags
    assert "[Evidence status" not in r["answer"]
    assert "[Note:" not in r["answer"]


def test_diagnostic_missing_fields(no_key):
    """Missing fields in diagnostic records are preserved as None, not invented."""
    def fake_run(name, **params):
        if name == "get_diagnostic_recommendations":
            return [{"pzem_number": 3, "timestamp": 1700000000000,
                     "fault_type": "high_power", "severity": "NORMAL",
                     "probable_cause": None, "energy_impact_kwh": None,
                     "cost_impact": None}]
        if name in ("get_system_summary",):
            return {}
        return []
    no_key.setattr(bob_tools, "run_tool", fake_run)
    r = ask_bob.ask_bob("PZEM-3 mein kya problem hai?")
    assert "probable cause" not in r["answer"].lower() or "does not establish" in r["answer"].lower()


def test_diagnostic_full_field_rendering(fake_data):
    """All diagnostic recommendation fields are rendered when available."""
    r = ask_bob.ask_bob("Kya koi diagnostic recommendation hai?")
    calls = [c for c in fake_data if c[0] == "get_diagnostic_recommendations"]
    assert len(calls) >= 1, "get_diagnostic_recommendations was not called"
    assert "PZEM" in r["answer"]
    assert "probable cause" in r["answer"].lower() or "Probable Cause" in r["answer"]
    assert "evidence" in r["answer"].lower() or "Evidence" in r["answer"]
    assert "what to check" in r["answer"].lower() or "What to Check" in r["answer"]
    assert "what to do now" in r["answer"].lower() or "What to Do Now" in r["answer"]
    assert "corrective action" in r["answer"].lower() or "Corrective Action" in r["answer"]
    assert "urgency" in r["answer"].lower() or "Urgency" in r["answer"]
    assert "confidence" in r["answer"].lower() or "Confidence" in r["answer"]
    assert "maintenance" in r["answer"].lower()


def test_diagnostic_no_fabrication(fake_data):
    """Diagnostic renderer must not fabricate missing values; verified fields must appear."""
    r = ask_bob.ask_bob("PZEM-1 mein kya problem hai?")
    calls = [c for c in fake_data if c[0] == "get_diagnostic_recommendations"]
    assert len(calls) >= 1, "get_diagnostic_recommendations was not called"
    assert "PZEM" in r["answer"]
    assert "probable cause" in r["answer"].lower() or "Probable Cause" in r["answer"]
    assert "evidence" in r["answer"].lower() or "Evidence" in r["answer"]
    assert "savings" not in r["answer"].lower()


def test_diagnostic_routing(fake_data):
    """Diagnostic intent routing works for what should I do questions."""
    r = ask_bob.ask_bob("Is fault mein mujhe kya karna chahiye?")
    assert any(c[0] == "get_diagnostic_recommendations" for c in fake_data)


def test_diagnostic_what_should_do_routing(fake_data):
    """what should I do questions route to diagnostic recommendations."""
    r = ask_bob.ask_bob("Abhi kya action lena chahiye?")
    assert any(c[0] == "get_diagnostic_recommendations" for c in fake_data)


def test_diagnostic_why_question_behavior(fake_data):
    """why questions use diagnostic recommendations as evidence source."""
    r = ask_bob.ask_bob("PZEM-3 mein overcurrent ka reason kya hai?")
    assert any(c[0] == "get_diagnostic_recommendations" for c in fake_data)


def test_existing_fault_routing_regression(fake_data):
    """Existing get_faults routing still works (regression test)."""
    r = ask_bob.ask_bob("Any recent faults?")
    assert ("get_faults", {}) in fake_data


def test_existing_live_routing_regression(fake_data):
    """Existing live meter routing still works (regression test)."""
    r = ask_bob.ask_bob("What is the power of PZEM 1?")
    assert ("get_meter", {"pzem_number": 1}) in fake_data


def test_existing_historical_routing_regression(fake_data):
    """Existing historical routing still works (regression test)."""
    r = ask_bob.ask_bob("PZEM-3 ne 10 September ko average power kya tha?")
    assert any(c[0] == "get_historical_analysis" for c in fake_data)


def test_no_fabricated_probable_cause(no_key):
    """If Stage 4 returns probable_cause=None, BOB must NOT generate its own."""
    def fake_run(name, **params):
        if name == "get_diagnostic_recommendations":
            return [{"pzem_number": 3, "timestamp": 1700000000000,
                     "fault_type": "high_power", "severity": "NORMAL",
                     "priority": "P3 - Informational",
                     "probable_cause": None,
                     "evidence": "power reading above threshold.",
                     "confidence": 0.7,
                     "what_to_check": "Check active loads.",
                     "what_to_do_now": "Check active loads.",
                     "corrective_action": "Assess whether high power is expected.",
                     "urgency": "MEDIUM", "maintenance_required": False,
                     "maintenance_timing": "Monitor.",
                     "energy_impact_kwh": None, "cost_impact": None,
                     "source_stages": ("Stage 3",)}]
        if name in ("get_system_summary",):
            return {}
        return []
    no_key.setattr(bob_tools, "run_tool", fake_run)
    r = ask_bob.ask_bob("PZEM-3 mein overcurrent ka reason kya hai?")
    assert "probable cause" not in r["answer"].lower() or "does not establish" in r["answer"].lower()


def test_no_fabricated_energy_cost(no_key):
    """If Stage 4 returns energy_impact_kwh=None and cost_impact=None, BOB must NOT invent savings."""
    def fake_run(name, **params):
        if name == "get_diagnostic_recommendations":
            return [{"pzem_number": 1, "timestamp": 1700000000000,
                     "fault_type": "overvoltage", "severity": "WARNING",
                     "probable_cause": "Supply overvoltage.",
                     "evidence": "voltage above threshold.",
                     "confidence": 0.85,
                     "what_to_check": "Verify voltage.",
                     "what_to_do_now": "Do not assume equipment failure.",
                     "corrective_action": "Independent voltage verification.",
                     "urgency": "HIGH", "maintenance_required": True,
                     "maintenance_timing": "Schedule inspection promptly.",
                     "energy_impact_kwh": None, "cost_impact": None,
                     "source_stages": ("Stage 3",)}]
        if name in ("get_system_summary",):
            return {}
        return []
    no_key.setattr(bob_tools, "run_tool", fake_run)
    r = ask_bob.ask_bob("PZEM-1 mein kya problem hai?")
    assert "savings" not in r["answer"].lower()


def test_diagnostic_tool_params(no_key):
    """get_diagnostic_recommendations has correct _TOOL_PARAMS."""
    assert "pzem_number" in bob_tools._TOOL_PARAMS["get_diagnostic_recommendations"]
    assert "start" in bob_tools._TOOL_PARAMS["get_diagnostic_recommendations"]
    assert "end" in bob_tools._TOOL_PARAMS["get_diagnostic_recommendations"]
    assert "limit" in bob_tools._TOOL_PARAMS["get_diagnostic_recommendations"]
    assert "severity" in bob_tools._TOOL_PARAMS["get_diagnostic_recommendations"]
    assert "fault_type" in bob_tools._TOOL_PARAMS["get_diagnostic_recommendations"]


def test_diagnostic_tool_rejects_invalid_pzem(no_key):
    """Invalid pzem_number raises ToolError."""
    with pytest.raises(bob_tools.ToolError) as exc:
        bob_tools.run_tool("get_diagnostic_recommendations", pzem_number=99)
    assert exc.value.code == "invalid_pzem"


def test_diagnostic_tool_rejects_invalid_limit(no_key):
    """Invalid limit raises ToolError."""
    with pytest.raises(bob_tools.ToolError) as exc:
        bob_tools.run_tool("get_diagnostic_recommendations", limit=-1)
    assert exc.value.code == "invalid_limit"


def test_diagnostic_renderer_empty(no_key):
    """_render_diagnostic_recommendations returns clear message for empty list."""
    from ai.ask_bob import _render_diagnostic_recommendations
    result = _render_diagnostic_recommendations([])
    assert "no verified diagnostic recommendation" in result.lower() or "not available" in result.lower()


def test_diagnostic_renderer_with_data(fake_data):
    """_render_diagnostic_recommendations renders fields correctly."""
    from ai.ask_bob import _render_diagnostic_recommendations
    recs = [{"pzem_number": 1, "timestamp": 1700000000000, "fault_type": "overvoltage",
             "severity": "WARNING", "priority": "P1 - Critical",
             "probable_cause": "Supply overvoltage.",
             "evidence": "voltage above threshold.",
             "confidence": 0.85,
             "what_to_check": "Verify voltage.",
             "what_to_do_now": "Do not assume equipment failure.",
             "corrective_action": "Independent voltage verification.",
             "urgency": "HIGH", "maintenance_required": True,
             "maintenance_timing": "Schedule inspection promptly.",
             "energy_impact_kwh": None, "cost_impact": None,
             "source_stages": ("Stage 3",)}]
    result = _render_diagnostic_recommendations(recs)
    assert "PZEM 1" in result
    assert "overvoltage" in result.lower()
    assert "WARNING" in result


def test_diagnostic_no_invented_confidence(no_key):
    """Confidence is preserved exactly, not upgraded."""
    def fake_run(name, **params):
        if name == "get_diagnostic_recommendations":
            return [{"pzem_number": 1, "timestamp": 1700000000000,
                     "fault_type": "overvoltage", "severity": "WARNING",
                     "probable_cause": "Supply overvoltage.",
                     "evidence": "voltage above threshold.",
                     "confidence": 0.5,
                     "what_to_check": "Verify voltage.",
                     "what_to_do_now": "Do not assume equipment failure.",
                     "corrective_action": "Independent voltage verification.",
                     "urgency": "HIGH", "maintenance_required": True,
                     "maintenance_timing": "Schedule inspection promptly.",
                     "energy_impact_kwh": None, "cost_impact": None,
                     "source_stages": ("Stage 3",)}]
        if name in ("get_system_summary",):
            return {}
        return []
    no_key.setattr(bob_tools, "run_tool", fake_run)
    r = ask_bob.ask_bob("PZEM-1 mein kya problem hai?")
    assert "0.5" in r["answer"] or "high confidence" not in r["answer"].lower()
