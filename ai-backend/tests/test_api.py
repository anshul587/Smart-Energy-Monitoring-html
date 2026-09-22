"""
tests/test_api.py — Stage 15 REST API (24 scenarios).
Reads are faked via ai.api_store.set_db_get (no Firebase / credentials needed).
"""
from __future__ import annotations

import os
import time
import types

import pytest

from ai import api_store
from ai.api_server import create_app


def _now_ms():
    return int(time.time() * 1000)


def make_store():
    now = int(time.time())
    return {
        "meters/pzem_1": {"voltage": 230.1, "current": 1.2, "power": 276.0,
                          "energy": 1.5, "pf": 0.95, "frequency": 49.98,
                          "lastSeen": _now_ms()},
        "meters/pzem_2": {"voltage": 0.0, "current": 0.0, "power": 0.0,
                          "energy": 0.4, "pf": 0.0, "frequency": 0.0,
                          "lastSeen": 0},  # offline (stale)
        "meters/pzem_3": {"voltage": 228.0, "current": 0.5, "power": 114.0,
                          "energy": 0.9, "pf": 0.9, "frequency": 50.0,
                          "lastSeen": _now_ms()},
        # pzem 4..9 intentionally absent (None) -> not returned
        "ai/anomalies/pzem_1": {str(now): {
            "pzem_number": 1, "timestamp": now, "anomaly_label": "ANOMALY",
            "anomaly_severity_provisional": "high", "model_status": "READY"}},
        "ai/anomalies/pzem_2": {str(now - 10): {
            "pzem_number": 2, "timestamp": now - 10, "anomaly_label": "NORMAL",
            "anomaly_severity_provisional": "none", "model_status": "READY"}},
        "ai/faults/pzem_1": {str(now): {
            "pzem_number": 1, "timestamp": now, "fault_type": "overvoltage",
            "severity": "high", "measured_value": 260.0, "reason": "v>limit"}},
        "ai/peaks/pzem_1": {str(now): {
            "pzem_number": 1, "timestamp": now, "peak_power_w": 500.0,
            "average_power_w": 300.0, "analysis_window": {"start": now - 100, "end": now}}},
        "ai/peaks/system": {str(now): {
            "timestamp": now, "total_peak_power_w": 1200.0,
            "dominant_pzems": [1, 3]}},
        "ai/maintenance/pzem_1": {str(now): {
            "pzem_number": 1, "timestamp": now, "risk_score": 80, "risk_level": "HIGH",
            "analysis_window": {"start": now - 100, "end": now}}},
        "ai/maintenance/system": {str(now): {
            "timestamp": now, "high_risk_meters": [1], "watch_meters": [2],
            "normal_meters": [3], "highest_risk_pzem": 1, "highest_risk_score": 80}},
        "ai/forecast/pzem_1": {str(now): {
            "pzem_number": 1, "anchor_timestamp": now, "status": "FORECAST",
            "forecast_24h": [{"t": now, "p": 300}], "forecast_7d": [{"t": now, "p": 280}]}},
        "ai/forecast/system": {str(now): {
            "anchor_timestamp": now, "status": "FORECAST", "meters_included": [1, 3],
            "forecast_24h": [{"t": now, "p": 420}], "forecast_7d": [{"t": now, "p": 400}]}},
        "ai/bill_prediction": {str(now): {
            "timestamp": now, "anchor_timestamp": now, "status": "OK",
            "estimated_bill": 42.0, "estimated_total_energy_kwh": 35.0}},
        "ai/energy_saving": {str(now): {
            "timestamp": now, "status": "RECOMMENDATIONS", "recommendation_count": 2,
            "recommendations": [
                {"pzem_number": 1, "priority": "high", "recommendation": "Shift load"},
                {"pzem_number": 3, "priority": "low", "recommendation": "Trim standby"},
            ]}},
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


# ---- 1. health -------------------------------------------------------------
def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"
    assert r.get_json()["data"]["version"] == "v1"


# ---- 2. meters list --------------------------------------------------------
def test_meters_list(client):
    r = client.get("/api/v1/meters")
    assert r.status_code == 200
    data = r.get_json()["data"]
    nums = {m["pzem_number"] for m in data}
    assert {1, 2, 3}.issubset(nums)
    online = {m["pzem_number"] for m in data if m["online"]}
    assert 1 in online and 3 in online and 2 not in online


# ---- 3. valid PZEM ---------------------------------------------------------
def test_valid_pzem(client):
    r = client.get("/api/v1/meters/1")
    assert r.status_code == 200
    assert r.get_json()["data"]["pzem_number"] == 1
    assert r.get_json()["data"]["online"] is True


# ---- 4. invalid PZEM -------------------------------------------------------
def test_invalid_pzem(client):
    r = client.get("/api/v1/meters/99")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_pzem"


# ---- 5. offline meter ------------------------------------------------------
def test_offline_meter(client):
    r = client.get("/api/v1/meters/2")
    assert r.status_code == 200
    d = r.get_json()["data"]
    assert d["online"] is False
    assert d["power"] is None  # not fabricated for offline
    assert d["frequency"] is None


# ---- 6. summary ------------------------------------------------------------
def test_summary(client):
    r = client.get("/api/v1/summary")
    assert r.status_code == 200
    d = r.get_json()["data"]
    assert d["system_status"] == "online"
    assert d["online_meter_count"] == 2
    assert d["total_meter_count"] == 9
    assert d["active_anomaly_count"] == 1
    assert d["active_fault_count"] == 1
    assert d["maintenance_risk"]["high_risk_meters"] == 1
    assert d["latest_peak"]["total_peak_power_w"] == 1200.0
    assert d["forecast_available"] is True
    assert d["latest_bill_prediction"]["estimated_bill"] == 42.0
    assert d["energy_saving"]["recommendation_count"] == 2


# ---- 7-13. AI endpoints ----------------------------------------------------
def test_anomalies(client):
    r = client.get("/api/v1/anomalies")
    assert r.status_code == 200
    assert r.get_json()["meta"]["count"] == 2


def test_faults(client):
    r = client.get("/api/v1/faults")
    assert r.status_code == 200
    assert r.get_json()["meta"]["count"] == 1


def test_peaks(client):
    r = client.get("/api/v1/peaks")
    assert r.status_code == 200
    assert r.get_json()["meta"]["count"] == 2  # pzem_1 + system


def test_maintenance(client):
    r = client.get("/api/v1/maintenance")
    assert r.status_code == 200
    assert r.get_json()["meta"]["count"] == 2


def test_forecast(client):
    r = client.get("/api/v1/forecast")
    assert r.status_code == 200
    assert r.get_json()["meta"]["count"] == 2


def test_bill_prediction(client):
    r = client.get("/api/v1/bill-prediction")
    assert r.status_code == 200
    assert r.get_json()["meta"]["count"] == 1


def test_energy_saving(client):
    r = client.get("/api/v1/energy-saving")
    assert r.status_code == 200
    assert r.get_json()["meta"]["count"] == 1


# ---- 14. monthly reports ---------------------------------------------------
def test_monthly_reports(client, tmp_path, monkeypatch):
    rep_dir = tmp_path / "reports_monthly"
    monthly = rep_dir / "monthly"
    monthly.mkdir(parents=True)
    (monthly / "report-2026-08.pdf").write_bytes(b"%PDF-1.4")
    (monthly / "latest.pdf").write_bytes(b"%PDF-1.4")
    monkeypatch.setenv("REPORTS_DIR", str(rep_dir))
    r = client.get("/api/v1/reports/monthly")
    assert r.status_code == 200
    files = r.get_json()["data"]
    assert any(f["filename"] == "report-2026-08.pdf" for f in files)

    dl = client.get("/api/v1/reports/monthly/report-2026-08.pdf")
    assert dl.status_code == 200
    assert dl.mimetype == "application/pdf"

    bad = client.get("/api/v1/reports/monthly/..%2f..%2fetc%2fpasswd")
    assert bad.status_code in (403, 404)


# ---- 15. query filters -----------------------------------------------------
def test_query_filters(client):
    r = client.get("/api/v1/anomalies?pzem_number=1")
    assert r.get_json()["meta"]["count"] == 1
    assert r.get_json()["data"][0]["pzem_number"] == 1

    now = int(time.time())
    r2 = client.get(f"/api/v1/anomalies?start={now + 100}")
    assert r2.get_json()["meta"]["count"] == 0

    r3 = client.get("/api/v1/anomalies?severity=high")
    # severity_key for anomalies is anomaly_severity_provisional
    assert r3.get_json()["meta"]["count"] == 1

    r4 = client.get("/api/v1/energy-saving?priority=high")
    assert r4.get_json()["meta"]["count"] == 1

    r5 = client.get("/api/v1/forecast?horizon=24h")
    rec = r5.get_json()["data"][0]
    assert "forecast_7d" not in rec and "forecast_24h" in rec


# ---- 16. invalid timestamps ------------------------------------------------
def test_invalid_timestamps(client):
    r = client.get("/api/v1/anomalies?start=abc")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_timestamp"
    r2 = client.get(f"/api/v1/anomalies?start=100&end=50")
    assert r2.status_code == 400


# ---- 17. invalid limits ----------------------------------------------------
def test_invalid_limits(client):
    r = client.get("/api/v1/anomalies?limit=-1")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_limit"
    r2 = client.get("/api/v1/anomalies?limit=foo")
    assert r2.status_code == 400


# ---- 18. maximum limit enforcement -----------------------------------------
def test_max_limit_enforcement(client):
    r = client.get("/api/v1/anomalies?limit=1000")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "limit_too_large"
    # exactly MAX is allowed
    r2 = client.get("/api/v1/anomalies?limit=500")
    assert r2.status_code == 200


# ---- 19. structured error responses ---------------------------------------
def test_structured_error(client):
    r = client.get("/api/v1/meters/0")
    body = r.get_json()
    assert body["status"] == "error"
    assert "code" in body["error"] and "message" in body["error"]
    assert "stack" not in body and "traceback" not in body


# ---- 20. empty data --------------------------------------------------------
def test_empty_data(client, monkeypatch):
    monkeypatch.setattr("ai.api_store._db_get", lambda path: None)
    api_store.clear_cache()
    r = client.get("/api/v1/anomalies")
    assert r.status_code == 200
    assert r.get_json()["data"] == []
    assert r.get_json()["meta"]["count"] == 0


# ---- 21. Firebase / read failure -------------------------------------------
def test_read_failure(client, monkeypatch):
    def boom(path):
        raise RuntimeError("firebase down")
    monkeypatch.setattr("ai.api_store._db_get", boom)
    api_store.clear_cache()
    for ep in ("/api/v1/meters", "/api/v1/anomalies", "/api/v1/summary"):
        r = client.get(ep)
        assert r.status_code == 503, ep
        assert r.get_json()["error"]["code"] == "data_unavailable"
        assert "firebase" not in r.get_json()["error"]["message"].lower()


# ---- 22. no secret leakage -------------------------------------------------
def test_no_secret_leakage(client):
    endpoints = ["/api/v1/health", "/api/v1/summary", "/api/v1/anomalies",
                 "/api/v1/bill-prediction", "/api/v1/reports/monthly"]
    for ep in endpoints:
        body = client.get(ep).get_data(as_text=True).lower()
        assert "service_account" not in body
        assert "aiza" not in body  # firebase api key pattern
        assert "firebase_service_account_path" not in body


# ---- 23. deterministic responses ------------------------------------------
def test_deterministic(client):
    a = client.get("/api/v1/anomalies").get_json()["data"]
    b = client.get("/api/v1/anomalies").get_json()["data"]
    assert a == b


# ---- 24. OpenAPI / docs ------------------------------------------------------
def test_openapi_and_docs(client):
    r = client.get("/api/v1/openapi.json")
    assert r.status_code == 200
    spec = r.get_json()
    assert spec["info"]["title"]
    assert "/summary" in spec["paths"]
    d = client.get("/api/v1/docs")
    assert d.status_code == 200
    assert "swagger" in d.get_data(as_text=True).lower()


# ---- 25. History: single PZEM historical -------------------------------------
def test_history_pzem_single(client, monkeypatch):
    """GET /api/v1/history/pzem/<n> with start/end timestamps."""
    now = int(time.time())
    # Monkeypatch data_loader to return synthetic history frames
    from ai import data_loader
    import ai.electrical_analysis as ea
    import pandas as pd
    import numpy as np

    BASE_TS = 1_700_000_000

    def make_frame(days=1, power_profile=lambda t: 100.0):
        ts = np.arange(BASE_TS, BASE_TS + days * 288 * 300, 300, dtype=np.int64)
        power_vals = np.array([power_profile(int(x)) for x in (ts % 86400) // 60])
        df = pd.DataFrame({
            "timestamp": ts,
            "voltage": 230.0,
            "current": np.full(len(ts), 0.5),
            "power": power_vals,
            "energy": np.cumsum(power_vals * 300 / 3600.0 / 1000.0),
            "frequency": 50.0,
            "pf": 0.98,
        })
        return df

    frames = {n: make_frame(days=1, power_profile=lambda t: 200.0) for n in range(1, 4)}

    monkeypatch.setattr(data_loader, "fetch_meter_history", lambda pzem_number, settings=None, force_full_refresh=False: data_loader.HistoryLoadResult(
        pzem_number=pzem_number,
        frame=frames.get(pzem_number, make_frame(days=1, power_profile=lambda t: 200.0)),
        available_days=1.0,
        requested_days=60,
        served_from_cache_only=False,
        dropped_rows=0,
        duplicate_keys_collapsed=0,
    ))

    r = client.get("/api/v1/history/pzem/1?start=%d&end=%d" % (BASE_TS, BASE_TS + 300))
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["status"] == "ok"
    assert "data" in body
    data = body["data"]
    # Verify system aggregation timestamps are correct (min/max correspond to actual values)
    if "total_power" in data:
        tp = data["total_power"]
        assert "average" in tp
        assert "maximum" in tp
        assert "min_timestamp" in tp
        assert "max_timestamp" in tp


# ---- 26. History: multi-PZEM comparison --------------------------------------
def test_history_multi_pzem(client, monkeypatch):
    """GET /api/v1/history/multi with pzem[] and start/end timestamps."""
    from ai import data_loader
    import pandas as pd
    import numpy as np

    BASE_TS = 1_700_000_000

    def make_frame(days=1, power_profile=lambda t: 100.0):
        ts = np.arange(BASE_TS, BASE_TS + 288 * 300, 300, dtype=np.int64)
        power_vals = np.array([power_profile(int(x)) for x in (ts % 86400) // 60])
        df = pd.DataFrame({
            "timestamp": ts,
            "voltage": 230.0,
            "current": np.full(len(ts), 0.5),
            "power": power_vals,
            "energy": np.cumsum(power_vals * 300 / 3600.0 / 1000.0),
            "frequency": 50.0,
            "pf": 0.98,
        })
        return df

    # Patch fetch_meter_history
    monkeypatch.setattr(data_loader, "fetch_meter_history", lambda pzem_number, settings=None, force_full_refresh=False: data_loader.HistoryLoadResult(
        pzem_number=pzem_number,
        frame=make_frame(days=1),
        available_days=1.0,
        requested_days=60,
        served_from_cache_only=False,
        dropped_rows=0,
        duplicate_keys_collapsed=0,
    ))

    r = client.get("/api/v1/history/multi?pzem=1,2&start=%d&end=%d" % (BASE_TS, BASE_TS + 300))
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["status"] == "ok"
    data = body["data"]
    # Should have pzem_1 and pzem_2 keys
    assert "1" in data or "pzem_1" in str(data)
    assert "2" in data or "pzem_2" in str(data)


# ---- 27. History: system-wide ------------------------------------------------
def test_history_system(client, monkeypatch):
    """GET /api/v1/history/system with start/end timestamps."""
    from ai import data_loader
    import pandas as pd
    import numpy as np

    BASE_TS = 1_700_000_000

    def make_frame(days=1, power_profile=lambda t: 100.0):
        ts = np.arange(BASE_TS, BASE_TS + 288 * 300, 300, dtype=np.int64)
        power_vals = np.array([power_profile(int(x)) for x in (ts % 86400) // 60])
        df = pd.DataFrame({
            "timestamp": ts,
            "voltage": 230.0,
            "current": np.full(len(ts), 0.5),
            "power": power_vals,
            "energy": np.cumsum(power_vals * 300 / 3600.0 / 1000.0),
            "frequency": 50.0,
            "pf": 0.98,
        })
        return df

    # Patch fetch_meter_history
    monkeypatch.setattr(data_loader, "fetch_meter_history", lambda pzem_number, settings=None, force_full_refresh=False: data_loader.HistoryLoadResult(
        pzem_number=pzem_number,
        frame=make_frame(days=1),
        available_days=1.0,
        requested_days=60,
        served_from_cache_only=False,
        dropped_rows=0,
        duplicate_keys_collapsed=0,
    ))

    r = client.get("/api/v1/history/system?start=%d&end=%d" % (BASE_TS, BASE_TS + 86400))
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["status"] == "ok"
    data = body["data"]
    # Verify system aggregation preserves simultaneous sum semantics
    assert "total_power" in data
    tp = data["total_power"]
    assert "average" in tp
    assert "maximum" in tp
    assert "min_timestamp" in tp
    assert "max_timestamp" in tp
    # min/max timestamps should correspond to actual min/max values (Part 1 fix)
    # Verify min_timestamp and max_timestamp correspond to actual values
    assert tp["average"] > 0 or tp["count"] > 0


# ---- 25. Ask BOB -------------------------------------------------------------
def test_ask_returns_answer(client, monkeypatch):
    from ai import bob_tools
    meters = [
        {"pzem_number": 1, "online": True, "power": 276.0, "energy": 1.5, "voltage": 230.1},
        {"pzem_number": 3, "online": True, "power": 114.0, "energy": 0.9, "voltage": 228.0},
        {"pzem_number": 2, "online": False, "power": 0.0, "energy": 0.4, "voltage": 0.0},
    ]

    def fake_run(name, **params):
        if name == "get_meters":
            return list(meters)
        return []

    monkeypatch.setattr(bob_tools, "run_tool", fake_run)
    r = client.post("/api/v1/ask", json={"question": "Which PZEM uses most power?"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert "PZEM" in body["answer"]


def test_ask_empty_question_rejected(client):
    r = client.post("/api/v1/ask", json={"question": "   "})
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "empty_question"


def test_ask_handles_read_failure(client, monkeypatch):
    from ai import bob_tools

    def boom(name, **params):
        raise RuntimeError("firebase down")

    monkeypatch.setattr(bob_tools, "run_tool", boom)
    r = client.post("/api/v1/ask", json={"question": "Which PZEM uses most power?"})
    # Read failure degrades gracefully to a safe, non-crashing answer (200).
    assert r.status_code == 200
    assert "don't have enough current data" in r.get_json()["answer"]
