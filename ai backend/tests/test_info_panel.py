"""
tests/test_info_panel.py — Stage 16 Info button / About panel.

Verifies:
  * project_knowledge.json is the single authoritative source for credits
    (guide, dashboard developer, hardware team) used by both Ask BOB and the
    frontend Info panel.
  * Ask BOB answers the required project / credits / guide / team questions
    from that knowledge (no fabrication).
  * The dashboard Info panel (index.html) renders the same authoritative
    credits and major sections, and contains no secrets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ai import ask_bob

_AI_ROOT = Path(__file__).resolve().parents[1]  # ai backend/
_KNOWLEDGE_PATH = _AI_ROOT / "ai" / "project_knowledge.json"
_DASHBOARD_HTML = (
    Path(__file__).resolve().parents[2]
    / "Dashboard Smart-Monitoring-System"
    / "index.html"
)

with open(_KNOWLEDGE_PATH, "r", encoding="utf-8") as _fh:
    KNOWLEDGE = json.load(_fh)

_FAKE_NAMES = ["Elon Musk", "John Doe", "Ada Lovelace", "Tony Stark"]


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.setattr(ask_bob, "_get_api_key", lambda: "")
    return monkeypatch


# ---- Shared knowledge source ------------------------------------------------
def test_knowledge_has_authoritative_credits():
    assert KNOWLEDGE.get("project_name")
    assert KNOWLEDGE.get("project_guide") == "Dr. Bhupendra Kumar"
    assert KNOWLEDGE.get("dashboard_developer") == "Anshul Ninawe"
    hw = KNOWLEDGE.get("hardware_team", [])
    assert hw == [
        "Yash Kawale", "Yash Dahake", "Swapnil Shendre",
        "Chetan Bokade", "Sanjog Godbole",
    ]


def test_info_panel_renders_major_sections(no_key):
    html = _DASHBOARD_HTML.read_text(encoding="utf-8")
    for section in [
        "About the Project", "Project Credits", "Purpose", "How It Works",
        "What the System Monitors", "Main Features", "AI Intelligence",
        "Energy Conservation", "Fault", "Advantages", "Ask BOB",
        "Current Limitations", "Future Scope",
    ]:
        assert section.lower() in html.lower(), f"missing section: {section}"


def test_info_panel_has_correct_credits(no_key):
    html = _DASHBOARD_HTML.read_text(encoding="utf-8")
    assert "Dr. Bhupendra Kumar" in html
    assert "Anshul Ninawe" in html
    for name in KNOWLEDGE["hardware_team"]:
        assert name in html, f"hardware member missing from Info panel: {name}"


def test_info_panel_contains_no_secrets(no_key):
    raw = _DASHBOARD_HTML.read_text(encoding="utf-8")
    assert "sk-ant" not in raw
    assert "AIza" not in raw
    assert "private_key" not in raw
    assert "BEGIN PRIVATE KEY" not in raw
    # The Info panel is project-focused; no programming-language / stack section.
    assert "Programming Language" not in raw
    assert "Technology Stack" not in raw


# ---- Ask BOB project / credits questions ------------------------------------
def test_ask_project_name(no_key):
    r = ask_bob.ask_bob("What is the project name?")
    assert r["source"] == "project"
    assert KNOWLEDGE["project_name"] in r["answer"]


def test_ask_tell_me_about(no_key):
    r = ask_bob.ask_bob("Tell me about this project.")
    assert r["source"] == "project"
    assert "Anshul Ninawe" in r["answer"]
    assert "ESP32" in r["answer"]


def test_ask_who_made_dashboard(no_key):
    for q in ["Who made the dashboard?", "Who programmed the dashboard?"]:
        r = ask_bob.ask_bob(q)
        assert r["source"] == "project"
        assert "Anshul Ninawe" in r["answer"]


def test_ask_who_is_project_guide(no_key):
    r = ask_bob.ask_bob("Who is the project guide?")
    assert r["source"] == "project"
    assert "Dr. Bhupendra Kumar" in r["answer"]


def test_ask_who_worked_on_hardware(no_key):
    r = ask_bob.ask_bob("Who worked on the hardware?")
    assert r["source"] == "project"
    for name in KNOWLEDGE["hardware_team"]:
        assert name in r["answer"]


def test_ask_team_members(no_key):
    r = ask_bob.ask_bob("Who are the team members?")
    assert r["source"] == "project"
    for name in KNOWLEDGE["hardware_team"]:
        assert name in r["answer"]


def test_ask_purpose(no_key):
    r = ask_bob.ask_bob("What is the purpose of the project?")
    assert r["source"] == "project"
    assert "energy" in r["answer"].lower()


def test_ask_how_system_works(no_key):
    r = ask_bob.ask_bob("How does the system work?")
    assert r["source"] == "project"
    assert "ESP32" in r["answer"]


def test_ask_features(no_key):
    r = ask_bob.ask_bob("What are the features?")
    assert r["source"] == "project"
    assert "real-time" in r["answer"].lower()


def test_ask_advantages(no_key):
    r = ask_bob.ask_bob("What are the advantages?")
    assert r["source"] == "project"
    assert "real-time" in r["answer"].lower() or "low-cost" in r["answer"].lower()


def test_ask_what_can_bob_do(no_key):
    r = ask_bob.ask_bob("What can BOB do?")
    assert r["source"] == "casual"
    assert "energy" in r["answer"].lower()


# ---- No fabrication / shared consistency ------------------------------------
def test_ask_no_fabricated_team(no_key):
    r = ask_bob.ask_bob("Who are the team members?")
    for fake in _FAKE_NAMES:
        assert fake not in r["answer"]


def test_shared_knowledge_consistency(no_key):
    """Ask BOB and the Info panel must agree on the authoritative credits."""
    guide = ask_bob.ask_bob("Who is the project guide?")["answer"]
    assert KNOWLEDGE["project_guide"] in guide

    dev = ask_bob.ask_bob("Who made the dashboard?")["answer"]
    assert KNOWLEDGE["dashboard_developer"] in dev

    html = _DASHBOARD_HTML.read_text(encoding="utf-8")
    assert KNOWLEDGE["project_guide"] in html
    assert KNOWLEDGE["dashboard_developer"] in html
    for name in KNOWLEDGE["hardware_team"]:
        assert name in html
