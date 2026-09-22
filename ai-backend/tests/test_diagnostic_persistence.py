"""
tests/test_diagnostic_persistence.py
-----------------------------------------
Stage 4: DiagnosticRecommendation persistence tests.
All tests use mocked Firebase — no production database required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai import persist_ai_results as par
from ai.diagnostic_recommendation import DiagnosticRecommendation
from ai.config import Settings


def _make_rec(
    pzem_number: int = 1,
    fault_type: str = "overvoltage",
    timestamp: int = 1_700_000_000,
    **kwargs,
) -> DiagnosticRecommendation:
    """Build a minimal DiagnosticRecommendation for testing."""
    default_kwargs = dict(
        recommendation_id=f"REC-TEST-{pzem_number}-{timestamp}",
        timestamp=timestamp,
        pzem_system=f"PZEM-{pzem_number}" if pzem_number else "SYSTEM",
        condition="Test condition.",
        fault_type=fault_type,
        severity="WARNING" if fault_type == "overvoltage" else "NORMAL",
        priority="P1 - Critical",
        probable_cause="Test probable cause.",
        why_it_happened="Test why.",
        evidence="Test evidence.",
        what_to_check="Check this.",
        what_to_do_now="Do that.",
        corrective_action="Fix this.",
        urgency="HIGH",
        maintenance_required=True,
        maintenance_timing="ASAP",
        energy_impact_kwh=5.0,
        cost_impact=1.23,
        confidence=0.85,
        source_stages=("Stage 2A", "Stage 2B"),
    )
    default_kwargs.update(kwargs)
    return DiagnosticRecommendation(**default_kwargs)


def _make_settings(pzem_count: int = 9, tmp_path: Path | None = None) -> Settings:
    if tmp_path is None:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            return Settings(
                firebase_service_account_path="unused.json",
                firebase_database_url="https://unused.example/",
                pzem_count=pzem_count,
                history_retention_days=60,
                cache_dir=td,
                anthropic_api_key="",
            )
    return Settings(
        firebase_service_account_path="unused.json",
        firebase_database_url="https://unused.example/",
        pzem_count=pzem_count,
        history_retention_days=60,
        cache_dir=tmp_path,
        anthropic_api_key="",
    )


# ============================================================================
# 1. Payload construction
# ============================================================================

def test_diagnostic_payload_has_all_fields():
    """Payload contains every DiagnosticRecommendation field."""
    rec = _make_rec()
    payload = par._diagnostic_recommendation_payload(rec)

    expected_keys = [
        "recommendation_id", "timestamp", "pzem_number", "pzem_system",
        "condition", "fault_type", "severity", "priority",
        "probable_cause", "why_it_happened", "evidence",
        "what_to_check", "what_to_do_now", "corrective_action",
        "urgency", "maintenance_required", "maintenance_timing",
        "energy_impact_kwh", "cost_impact", "confidence",
        "source_stages",
    ]
    for key in expected_keys:
        assert key in payload, f"Missing key: {key}"


def test_diagnostic_payload_pzem_extraction():
    """pzem_number is extracted correctly from pzem_system."""
    rec = _make_rec(pzem_number=3)
    payload = par._diagnostic_recommendation_payload(rec)
    assert payload["pzem_number"] == 3


def test_diagnostic_payload_system_has_none_pzem():
    """SYSTEM pzem_system results in pzem_number=None."""
    rec = _make_rec(pzem_number=0)
    payload = par._diagnostic_recommendation_payload(rec)
    assert payload["pzem_number"] is None
    assert payload["pzem_system"] == "SYSTEM"


def test_diagnostic_payload_source_stages_is_list():
    """source_stages (tuple) is serialized as a list for JSON safety."""
    rec = _make_rec()
    payload = par._diagnostic_recommendation_payload(rec)
    assert isinstance(payload["source_stages"], list)
    assert payload["source_stages"] == ["Stage 2A", "Stage 2B"]


def test_diagnostic_payload_none_values():
    """None energy_impact_kwh and cost_impact are preserved."""
    rec = _make_rec(energy_impact_kwh=None, cost_impact=None)
    payload = par._diagnostic_recommendation_payload(rec)
    assert payload["energy_impact_kwh"] is None
    assert payload["cost_impact"] is None


def test_diagnostic_payload_numeric_types():
    """Numeric fields are the correct types."""
    rec = _make_rec(timestamp=1_700_000_000)
    payload = par._diagnostic_recommendation_payload(rec)
    assert isinstance(payload["timestamp"], int)
    assert isinstance(payload["confidence"], float)
    assert isinstance(payload["energy_impact_kwh"], float)


def test_diagnostic_payload_round_trip():
    """Payload dict can be round-tripped back into equivalent values.
    pzem_number is a storage helper, not a dataclass field, so it must be removed.
    pzem_system must be preserved."""
    rec = _make_rec()
    payload = par._diagnostic_recommendation_payload(rec)
    payload.pop("pzem_number", None)
    reconstructed = DiagnosticRecommendation(**payload)
    assert reconstructed.recommendation_id == rec.recommendation_id
    assert reconstructed.timestamp == rec.timestamp
    assert reconstructed.fault_type == rec.fault_type
    assert reconstructed.priority == rec.priority
    assert reconstructed.evidence == rec.evidence
    assert reconstructed.condition == rec.condition
    assert reconstructed.pzem_system == rec.pzem_system


# ============================================================================
# 2. Write function
# ============================================================================

def test_write_diagnostic_recommendation_valid(tmp_path: Path, monkeypatch):
    """write_diagnostic_recommendation returns True for a valid rec."""
    settings = _make_settings(tmp_path=tmp_path)
    monkeypatch.setattr(par, "get_settings", lambda: settings)
    mock_db = MagicMock()
    mock_db.child.return_value.get.return_value = None
    monkeypatch.setattr(par, "_db_ref", lambda path: mock_db)

    rec = _make_rec(pzem_number=1)
    result = par.write_diagnostic_recommendation(rec)
    assert result is True


def test_write_diagnostic_recommendation_out_of_range_pzem(tmp_path: Path, monkeypatch):
    """write_diagnostic_recommendation returns False for out-of-range PZEM."""
    settings = _make_settings(pzem_count=3, tmp_path=tmp_path)
    monkeypatch.setattr(par, "get_settings", lambda: settings)

    rec = _make_rec(pzem_number=99)
    result = par.write_diagnostic_recommendation(rec)
    assert result is False


def test_write_diagnostic_recommendation_negative_timestamp(tmp_path: Path, monkeypatch):
    """write_diagnostic_recommendation returns False for negative timestamp."""
    settings = _make_settings(tmp_path=tmp_path)
    monkeypatch.setattr(par, "get_settings", lambda: settings)

    rec = _make_rec(timestamp=-1)
    result = par.write_diagnostic_recommendation(rec)
    assert result is False


def test_write_diagnostic_recommendation_system_level(tmp_path: Path, monkeypatch):
    """SYSTEM-level recommendations (pzem=None) can be persisted."""
    settings = _make_settings(tmp_path=tmp_path)
    monkeypatch.setattr(par, "get_settings", lambda: settings)
    mock_db = MagicMock()
    mock_db.child.return_value.get.return_value = None
    monkeypatch.setattr(par, "_db_ref", lambda path: mock_db)

    rec = _make_rec(pzem_number=0)
    result = par.write_diagnostic_recommendation(rec)
    assert result is True


def test_write_diagnostic_recommendation_duplicate(monkeypatch, tmp_path):
    """write_diagnostic_recommendation returns True for duplicates (idempotent)."""
    settings = _make_settings(tmp_path=tmp_path)
    monkeypatch.setattr(par, "get_settings", lambda: settings)
    mock_db = MagicMock()
    mock_db.child.return_value.get.return_value = {"exists": True}
    monkeypatch.setattr(par, "_db_ref", lambda path: mock_db)

    rec = _make_rec(pzem_number=1)
    result = par.write_diagnostic_recommendation(rec)
    assert result is True


# ============================================================================
# 3. Persist all recommendations
# ============================================================================

def test_persist_diagnostic_recommendations_returns_counts(tmp_path: Path, monkeypatch):
    """persist_diagnostic_recommendations returns a PZEM->count dict."""
    settings = _make_settings(pzem_count=3, tmp_path=tmp_path)
    monkeypatch.setattr(par, "get_settings", lambda: settings)
    mock_db = MagicMock()
    mock_db.child.return_value.get.return_value = None
    monkeypatch.setattr(par, "_db_ref", lambda path: mock_db)

    recs = {1: [_make_rec(pzem_number=1)]}
    counts = par.persist_diagnostic_recommendations(recs)
    assert counts[1] == 1
    assert counts[2] == 0
    assert counts[3] == 0


def test_persist_diagnostic_recommendations_empty(tmp_path: Path, monkeypatch):
    """Empty recommendations_map returns zero counts for all PZEMs."""
    settings = _make_settings(pzem_count=3, tmp_path=tmp_path)
    monkeypatch.setattr(par, "get_settings", lambda: settings)
    mock_db = MagicMock()
    monkeypatch.setattr(par, "_db_ref", lambda path: mock_db)

    counts = par.persist_diagnostic_recommendations({})
    assert counts == {1: 0, 2: 0, 3: 0}


def test_persist_diagnostic_recommendations_all_zero(monkeypatch, tmp_path):
    """When all writes fail, all counts are 0."""
    settings = _make_settings(pzem_count=2, tmp_path=tmp_path)
    monkeypatch.setattr(par, "get_settings", lambda: settings)
    mock_db = MagicMock()
    mock_db.child.return_value.get.return_value = None
    mock_db.child.return_value.set.side_effect = Exception("firebase error")
    monkeypatch.setattr(par, "_db_ref", lambda path: mock_db)

    recs = {1: [_make_rec(pzem_number=1)], 2: [_make_rec(pzem_number=2)]}
    counts = par.persist_diagnostic_recommendations(recs)
    assert counts[1] == 0
    assert counts[2] == 0


# ============================================================================
# 4. Payload structure fidelity
# ============================================================================

def test_diagnostic_payload_preserves_evidence():
    """Evidence string is preserved exactly."""
    rec = _make_rec()
    payload = par._diagnostic_recommendation_payload(rec)
    assert payload["evidence"] == "Test evidence."


def test_diagnostic_payload_preserves_corrective_action():
    """Corrective action is preserved exactly."""
    rec = _make_rec()
    payload = par._diagnostic_recommendation_payload(rec)
    assert payload["corrective_action"] == "Fix this."


def test_diagnostic_payload_maintenance_fields():
    """Maintenance fields are correctly serialized."""
    rec = _make_rec(maintenance_required=False, maintenance_timing="N/A")
    payload = par._diagnostic_recommendation_payload(rec)
    assert payload["maintenance_required"] is False
    assert payload["maintenance_timing"] == "N/A"


def test_diagnostic_payload_priority_values():
    """P1/P2/P3 priority values are preserved."""
    for priority in ("P1 - Critical", "P2 - Important", "P3 - Informational"):
        rec = _make_rec(priority=priority)
        payload = par._diagnostic_recommendation_payload(rec)
        assert payload["priority"] == priority


def test_diagnostic_payload_source_stage_order():
    """source_stages preserves order."""
    rec = _make_rec()
    payload = par._diagnostic_recommendation_payload(rec)
    assert payload["source_stages"] == ["Stage 2A", "Stage 2B"]
