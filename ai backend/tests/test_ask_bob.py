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
    "get_monthly_reports": lambda **k: [dict(r) for r in REPORTS],
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
                    "get_energy_saving", "get_monthly_reports"):
            return []
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
