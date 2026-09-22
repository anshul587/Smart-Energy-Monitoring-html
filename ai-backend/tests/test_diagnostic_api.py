"""
tests/test_diagnostic_api.py
-----------------------------------------
Stage 4: DiagnosticRecommendation REST API tests.
All tests use mocked Firebase via ai.api_store.set_db_get — no credentials needed.
"""
from __future__ import annotations

import time
import types

import pytest

from ai import api_store
from ai.api_server import create_app


def _now():
    return int(time.time())


def make_store():
    now = _now()
    return {
        "ai/diagnostic_recommendations/pzem_1": {
            str(now): {
                "recommendation_id": "REC-1-1",
                "timestamp": now,
                "pzem_number": 1,
                "pzem_system": "PZEM-1",
                "condition": "Overvoltage detected.",
                "fault_type": "overvoltage",
                "severity": "WARNING",
                "priority": "P1 - Critical",
                "probable_cause": "Supply overvoltage.",
                "why_it_happened": "Voltage above threshold.",
                "evidence": "voltage > 250V",
                "what_to_check": "Verify with independent meter.",
                "what_to_do_now": "Do not assume equipment failure.",
                "corrective_action": "Independent voltage verification.",
                "urgency": "HIGH",
                "maintenance_required": True,
                "maintenance_timing": "ASAP",
                "energy_impact_kwh": 5.0,
                "cost_impact": 1.23,
                "confidence": 0.85,
                "source_stages": ["Stage 2A"],
            },
            str(now - 100): {
                "recommendation_id": "REC-1-2",
                "timestamp": now - 100,
                "pzem_number": 1,
                "pzem_system": "PZEM-1",
                "condition": "High power detected.",
                "fault_type": "high_power",
                "severity": "NORMAL",
                "priority": "P2 - Important",
                "probable_cause": "High load.",
                "why_it_happened": "Power above threshold.",
                "evidence": "power > rated",
                "what_to_check": "Check loads.",
                "what_to_do_now": "Reduce load.",
                "corrective_action": "Reduce load.",
                "urgency": "MEDIUM",
                "maintenance_required": False,
                "maintenance_timing": "N/A",
                "energy_impact_kwh": None,
                "cost_impact": None,
                "confidence": 0.7,
                "source_stages": ["Stage 2A", "Stage 2B"],
            },
        },
        "ai/diagnostic_recommendations/pzem_2": {
            str(now - 200): {
                "recommendation_id": "REC-2-1",
                "timestamp": now - 200,
                "pzem_number": 2,
                "pzem_system": "PZEM-2",
                "condition": "Undervoltage.",
                "fault_type": "undervoltage",
                "severity": "NORMAL",
                "priority": "P3 - Informational",
                "probable_cause": "Supply sag.",
                "why_it_happened": "Voltage below threshold.",
                "evidence": "voltage < 220V",
                "what_to_check": "Verify with independent meter.",
                "what_to_do_now": "Verify voltage.",
                "corrective_action": "Verify voltage.",
                "urgency": "LOW",
                "maintenance_required": False,
                "maintenance_timing": "N/A",
                "energy_impact_kwh": None,
                "cost_impact": None,
                "confidence": 0.5,
                "source_stages": ["Stage 2A"],
            },
        },
        "ai/diagnostic_recommendations/system": {
            str(now): {
                "recommendation_id": "REC-SYS-1",
                "timestamp": now,
                "pzem_number": None,
                "pzem_system": "SYSTEM",
                "condition": "System-wide issue.",
                "fault_type": "communication_degraded",
                "severity": "WARNING",
                "priority": "P1 - Critical",
                "probable_cause": "Comm degraded.",
                "why_it_happened": "Comm issues.",
                "evidence": "comm",
                "what_to_check": "Check comms.",
                "what_to_do_now": "Fix comms.",
                "corrective_action": "Fix comms.",
                "urgency": "HIGH",
                "maintenance_required": True,
                "maintenance_timing": "ASAP",
                "energy_impact_kwh": None,
                "cost_impact": None,
                "confidence": 0.9,
                "source_stages": ["Stage 2C"],
            },
        },
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    fake_settings = types.SimpleNamespace(pzem_count=9)
    monkeypatch.setattr("ai.api_store.get_settings", lambda: fake_settings)
    monkeypatch.setattr("ai.api_server.get_settings", lambda: fake_settings)
    store = make_store()
    monkeypatch.setattr("ai.api_store._db_get", lambda path: store.get(path))
    api_store.clear_cache()
    app = create_app()
    app.testing = True
    with app.test_client() as c:
        yield c
    api_store.clear_cache()


# ---- 1. All recommendations -------------------------------------------------

def test_all_diagnostic_recommendations(client):
    r = client.get("/api/v1/diagnostic-recommendations")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    data = body["data"]
    assert len(data) == 3  # 2 for PZEM-1 + 1 for SYSTEM (pzem_2 is separate)
    # Actually: pzem_1 has 2, pzem_2 has 1, system has 1 = 4 total
    # Wait, _read_pzem_collection iterates range(1, pzem_count+1), not "system"
    # So: pzem_1 (2) + pzem_2 (1) = 3 records
    assert len(data) == 3


# ---- 2. PZEM filtering ------------------------------------------------------

def test_diagnostic_recommendations_pzem_filter(client):
    r = client.get("/api/v1/diagnostic-recommendations?pzem_number=1")
    assert r.status_code == 200
    data = r.get_json()["data"]
    assert all(rec["pzem_number"] == 1 for rec in data)


def test_diagnostic_recommendations_invalid_pzem(client):
    r = client.get("/api/v1/diagnostic-recommendations?pzem_number=99")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_pzem"


def test_diagnostic_recommendations_pzem_none_system(client):
    """SYSTEM-level recommendations (pzem_number=None) are not returned
    when filtering by a specific PZEM."""
    r = client.get("/api/v1/diagnostic-recommendations?pzem_number=2")
    data = r.get_json()["data"]
    assert all(rec["pzem_number"] == 2 for rec in data)


# ---- 3. Timestamp filtering -------------------------------------------------

def test_diagnostic_recommendations_start_filter(client):
    now = _now()
    r = client.get(f"/api/v1/diagnostic-recommendations?start={now - 50}")
    assert r.status_code == 200
    data = r.get_json()["data"]
    for rec in data:
        ts = rec.get("timestamp")
        assert ts is None or ts >= now - 50


def test_diagnostic_recommendations_end_filter(client):
    now = _now()
    r = client.get(f"/api/v1/diagnostic-recommendations?end={now - 50}")
    assert r.status_code == 200
    data = r.get_json()["data"]
    for rec in data:
        ts = rec.get("timestamp")
        assert ts is None or ts <= now - 50


def test_diagnostic_recommendations_start_after_end(client):
    r = client.get("/api/v1/diagnostic-recommendations?start=100&end=50")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_timestamp"


def test_diagnostic_recommendations_invalid_timestamp(client):
    r = client.get("/api/v1/diagnostic-recommendations?start=abc")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_timestamp"


# ---- 4. Fault type filtering ------------------------------------------------

def test_diagnostic_recommendations_fault_type_filter(client):
    r = client.get("/api/v1/diagnostic-recommendations?fault_type=overvoltage")
    assert r.status_code == 200
    data = r.get_json()["data"]
    assert all(rec["fault_type"] == "overvoltage" for rec in data)


def test_diagnostic_recommendations_fault_type_empty(client):
    r = client.get("/api/v1/diagnostic-recommendations?fault_type=nonexistent")
    assert r.status_code == 200
    data = r.get_json()["data"]
    assert len(data) == 0


# ---- 5. Severity filtering --------------------------------------------------

def test_diagnostic_recommendations_severity_filter(client):
    r = client.get("/api/v1/diagnostic-recommendations?severity=WARNING")
    assert r.status_code == 200
    data = r.get_json()["data"]
    assert all(rec["severity"] == "WARNING" for rec in data)


# ---- 6. Priority filtering --------------------------------------------------

def test_diagnostic_recommendations_priority_filter(client):
    r = client.get("/api/v1/diagnostic-recommendations?priority=P1 - Critical")
    assert r.status_code == 200
    data = r.get_json()["data"]
    assert all(rec["priority"] == "P1 - Critical" for rec in data)


# ---- 7. Limit ---------------------------------------------------------------

def test_diagnostic_recommendations_limit(client):
    r = client.get("/api/v1/diagnostic-recommendations?limit=1")
    assert r.status_code == 200
    data = r.get_json()["data"]
    assert len(data) == 1
    assert r.get_json()["meta"]["limit"] == 1


def test_diagnostic_recommendations_invalid_limit(client):
    r = client.get("/api/v1/diagnostic-recommendations?limit=abc")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_limit"


def test_diagnostic_recommendations_limit_too_large(client):
    r = client.get("/api/v1/diagnostic-recommendations?limit=999")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "limit_too_large"


# ---- 8. Empty results -------------------------------------------------------

def test_diagnostic_recommendations_empty(monkeypatch):
    """No records when store has no diagnostic_recommendations."""
    fake_settings = types.SimpleNamespace(pzem_count=9)
    monkeypatch.setattr("ai.api_store.get_settings", lambda: fake_settings)
    monkeypatch.setattr("ai.api_server.get_settings", lambda: fake_settings)
    empty_store = {}
    monkeypatch.setattr("ai.api_store._db_get", lambda path: empty_store.get(path))
    api_store.clear_cache()
    app = create_app()
    app.testing = True
    with app.test_client() as c:
        r = c.get("/api/v1/diagnostic-recommendations")
    assert r.status_code == 200
    data = r.get_json()["data"]
    assert len(data) == 0
    assert r.get_json()["meta"]["count"] == 0


# ---- 9. Error handling ------------------------------------------------------

def test_diagnostic_recommendations_data_unavailable(monkeypatch):
    """API returns 503 when data source raises."""
    fake_settings = types.SimpleNamespace(pzem_count=9)
    monkeypatch.setattr("ai.api_store.get_settings", lambda: fake_settings)
    monkeypatch.setattr("ai.api_server.get_settings", lambda: fake_settings)
    monkeypatch.setattr("ai.api_store._db_get", lambda path: (_ for _ in ()).throw(RuntimeError("firebase down")))
    api_store.clear_cache()
    app = create_app()
    app.testing = True
    with app.test_client() as c:
        r = c.get("/api/v1/diagnostic-recommendations")
    assert r.status_code == 503
    assert r.get_json()["error"]["code"] == "data_unavailable"


# ---- 10. Round-trip data integrity ------------------------------------------

def test_diagnostic_recommendations_round_trip(client):
    """All fields preserved through API retrieval."""
    r = client.get("/api/v1/diagnostic-recommendations?pzem_number=1")
    data = r.get_json()["data"]
    for rec in data:
        assert "recommendation_id" in rec
        assert "timestamp" in rec
        assert "pzem_number" in rec
        assert "pzem_system" in rec
        assert "condition" in rec
        assert "fault_type" in rec
        assert "severity" in rec
        assert "priority" in rec
        assert "probable_cause" in rec
        assert "why_it_happened" in rec
        assert "evidence" in rec
        assert "what_to_check" in rec
        assert "what_to_do_now" in rec
        assert "corrective_action" in rec
        assert "urgency" in rec
        assert "maintenance_required" in rec
        assert "maintenance_timing" in rec
        assert "energy_impact_kwh" in rec
        assert "cost_impact" in rec
        assert "confidence" in rec
        assert "source_stages" in rec


def test_diagnostic_recommendations_meta_fields(client):
    """Meta contains count, limit, total and pzem_number when filtered."""
    r = client.get("/api/v1/diagnostic-recommendations?pzem_number=1&limit=50")
    meta = r.get_json()["meta"]
    assert "count" in meta
    assert "limit" in meta
    assert "total" in meta
    assert meta["pzem_number"] == 1


# ---- 11. Sort order ---------------------------------------------------------

def test_diagnostic_recommendations_sorted_by_timestamp(client):
    """Results sorted by timestamp descending."""
    now = _now()
    r = client.get("/api/v1/diagnostic-recommendations?pzem_number=1")
    data = r.get_json()["data"]
    timestamps = [rec["timestamp"] for rec in data]
    assert timestamps == sorted(timestamps, reverse=True)


# ---- 12. Audit Fix Regression Tests ---------------------------------------

def test_api_limit_zero(client):
    """API limit=0 returns empty list with meta count=0."""
    r = client.get("/api/v1/diagnostic-recommendations?limit=0")
    assert r.status_code == 200
    data = r.get_json()["data"]
    assert len(data) == 0
    assert r.get_json()["meta"]["count"] == 0
    assert r.get_json()["meta"]["limit"] == 0


def test_api_unsupported_fault_type(client):
    """API returns empty results for unsupported fault_type."""
    r = client.get("/api/v1/diagnostic-recommendations?fault_type=nonexistent")
    assert r.status_code == 200
    data = r.get_json()["data"]
    assert len(data) == 0


def test_api_future_start_timestamp(client):
    """API with future start timestamp returns empty results."""
    import time
    future_ts = int(time.time()) + 999999
    r = client.get(f"/api/v1/diagnostic-recommendations?start={future_ts}")
    assert r.status_code == 200
    data = r.get_json()["data"]
    assert len(data) == 0


def test_api_pzem_none_system_excluded(client):
    """SYSTEM-level records (pzem_number=None) are excluded when filtering by specific PZEM."""
    r = client.get("/api/v1/diagnostic-recommendations?pzem_number=2")
    data = r.get_json()["data"]
    assert all(rec["pzem_number"] == 2 for rec in data)
    assert all(rec["pzem_number"] is not None for rec in data)


def test_api_priority_top_level_filtering(client):
    """Priority filtering checks top-level record priority directly."""
    r = client.get("/api/v1/diagnostic-recommendations?priority=P1 - Critical")
    data = r.get_json()["data"]
    assert all(rec["priority"] == "P1 - Critical" for rec in data)


# ---- Regression: Firebase init failure returns empty data, not 500 ----

def test_default_db_get_returns_empty_on_firebase_failure(monkeypatch):
    """_default_db_get returns {} instead of raising when Firebase is
    unavailable (e.g., service account JSON not yet written during deploy).
    This prevents 500 errors on production endpoints."""
    from ai import api_store
    fake_settings = types.SimpleNamespace(pzem_count=9)
    monkeypatch.setattr("ai.api_store.get_settings", lambda: fake_settings)
    monkeypatch.setattr("ai.api_server.get_settings", lambda: fake_settings)
    # Simulate Firebase init failure in _db_ref
    def failing_db_ref(path):
        raise RuntimeError("Service account file not found")
    monkeypatch.setattr("ai.data_loader._db_ref", failing_db_ref)
    api_store.clear_cache()
    # _default_db_get should return {} instead of raising
    result = api_store._default_db_get("ai/diagnostic_recommendations/pzem_1")
    assert result == {}
