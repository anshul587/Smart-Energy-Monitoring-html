"""
ai/api_server.py
----------------
Stage 15: REST API for existing AI / energy results (read-only).

Framework: Flask (already a project dependency). FastAPI was NOT used because
it is not installed and the spec forbids adding unnecessary dependencies; the
project already has Flask, so we reuse it and hand-serve OpenAPI JSON + a
Swagger-UI /docs page (no extra packages).

The API only READS existing data through ai.api_store — it never runs a model
and never writes. Stage 14's scheduler and this API are independent consumers
of the canonical AI modules / Firebase paths.

Auth: NONE. This service is local/dev-only. See security notes in /openapi.json
and the final report. Do NOT expose publicly without adding authentication,
HTTPS, and rate limiting.
"""

from __future__ import annotations

import os
import re
import datetime
from typing import Any, Optional, Tuple

from flask import Blueprint, Flask, jsonify, request, send_file

from . import api_store
from . import ask_bob
from .config import get_settings

API_VERSION = "v1"
MAX_LIMIT = 500
DEFAULT_LIMIT = 50
ALLOWED_HORIZONS = {"24h", "7d", "both"}

bp = Blueprint("api", __name__, url_prefix=f"/api/{API_VERSION}")


# ---------------------------------------------------------------------------
# Response envelopes
# ---------------------------------------------------------------------------

def _ok(data: Any, meta: Optional[dict] = None) -> Tuple[Any, int]:
    body = {"status": "ok", "data": data}
    if meta is not None:
        body["meta"] = meta
    return jsonify(body), 200


def _err(code: str, message: str, status: int = 400) -> Tuple[Any, int]:
    return jsonify({"status": "error", "error": {"code": code, "message": message}}), status


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _pzem_count() -> int:
    return get_settings().pzem_count


def _parse_pzem() -> Tuple[Optional[int], Optional[Tuple[Any, int]]]:
    raw = request.args.get("pzem_number")
    if raw is None or raw == "":
        return None, None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None, _err("invalid_pzem", "pzem_number must be an integer")
    if not (1 <= n <= _pzem_count()):
        return None, _err("invalid_pzem", f"pzem_number must be between 1 and {_pzem_count()}")
    return n, None


def _parse_ts(name: str) -> Tuple[Optional[int], Optional[Tuple[Any, int]]]:
    raw = request.args.get(name)
    if raw is None or raw == "":
        return None, None
    try:
        ts = int(raw)
    except (TypeError, ValueError):
        return None, _err("invalid_timestamp", f"{name} must be a unix timestamp (integer seconds)")
    if ts < 0:
        return None, _err("invalid_timestamp", f"{name} must be >= 0")
    return ts, None


def _parse_limit() -> Tuple[Optional[int], Optional[Tuple[Any, int]]]:
    raw = request.args.get("limit")
    if raw is None or raw == "":
        return DEFAULT_LIMIT, None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None, _err("invalid_limit", "limit must be an integer")
    if n < 0:
        return None, _err("invalid_limit", "limit must be >= 0")
    if n > MAX_LIMIT:
        return None, _err("limit_too_large", f"limit exceeds maximum of {MAX_LIMIT}")
    return n, None


def _filter_records(records: list[dict], pzem: Optional[int], start: Optional[int],
                    end: Optional[int], severity: Optional[str], severity_key: str,
                    risk: Optional[str], priority: Optional[str]) -> list[dict]:
    out = []
    for r in records:
        ts = api_store.record_timestamp(r)
        pz = api_store.record_pzem(r)
        if pzem is not None:
            # energy_saving records are fleet-wide (pzem None) -> match via recommendations
            if pz is None:
                recs = r.get("recommendations", []) or []
                if not any(int(x.get("pzem_number", 0)) == pzem for x in recs if isinstance(x, dict)):
                    continue
            elif pz != pzem:
                continue
        if start is not None and ts is not None and ts < start:
            continue
        if end is not None and ts is not None and ts > end:
            continue
        if severity is not None and str(r.get(severity_key, "")).lower() != severity.lower():
            continue
        if risk is not None and str(r.get("risk_level", "")).lower() != risk.lower():
            continue
        if priority is not None:
            recs = r.get("recommendations", []) or []
            if not any(str(x.get("priority", "")).lower() == priority.lower() for x in recs if isinstance(x, dict)):
                continue
        out.append(r)
    out.sort(key=lambda r: api_store.record_timestamp(r) or 0, reverse=True)
    return out


def _wrap(records: list[dict], limit: int, pzem=None, start=None, end=None,
          extra_meta: Optional[dict] = None) -> Tuple[Any, int]:
    total = len(records)
    page = records[:limit]
    meta = {"count": len(page), "limit": limit, "total": total}
    if pzem is not None:
        meta["pzem_number"] = pzem
    if start is not None:
        meta["start"] = start
    if end is not None:
        meta["end"] = end
    if extra_meta:
        meta.update(extra_meta)
    return _ok(page, meta)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@bp.route("/health", methods=["GET"])
def health():
    return _ok({"status": "ok", "service": "ai-results-api", "version": API_VERSION,
                "timestamp": int(datetime.datetime.now(datetime.timezone.utc).timestamp())})


# ---------------------------------------------------------------------------
# Meters
# ---------------------------------------------------------------------------

@bp.route("/meters", methods=["GET"])
def meters():
    try:
        raw = api_store.read_all_meters()
    except Exception:
        return _err("data_unavailable", "Meter data source is unavailable", 503)
    data = []
    for n in sorted(raw.keys()):
        m = raw[n]
        if not isinstance(m, dict):
            continue
        online, age_ms = api_store.meter_online(m)
        entry = {
            "pzem_number": n,
            "online": online,
            "voltage": m.get("voltage"),
            "current": m.get("current"),
            "power": m.get("power") if online else None,
            "energy": m.get("energy"),
            "power_factor": m.get("pf"),
            "frequency": m.get("frequency") if online else None,
            "last_seen": m.get("lastSeen", m.get("timestamp")),
            "age_ms": age_ms,
        }
        data.append(entry)
    return _ok(data, {"count": len(data), "pzem_count": _pzem_count()})


@bp.route("/meters/<int:pzem_number>", methods=["GET"])
def meter_detail(pzem_number: int):
    if not (1 <= pzem_number <= _pzem_count()):
        return _err("invalid_pzem", f"pzem_number must be between 1 and {_pzem_count()}")
    try:
        m = api_store.read_meter(pzem_number)
    except Exception:
        return _err("data_unavailable", "Meter data source is unavailable", 503)
    if not isinstance(m, dict):
        return _err("not_found", f"No data for PZEM {pzem_number}", 404)
    online, age_ms = api_store.meter_online(m)
    return _ok({
        "pzem_number": pzem_number,
        "online": online,
        "voltage": m.get("voltage"),
        "current": m.get("current"),
        "power": m.get("power") if online else None,
        "energy": m.get("energy"),
        "power_factor": m.get("pf"),
        "frequency": m.get("frequency") if online else None,
        "last_seen": m.get("lastSeen", m.get("timestamp")),
        "age_ms": age_ms,
    })


# ---------------------------------------------------------------------------
# AI result endpoints
# ---------------------------------------------------------------------------

def _ai_route(reader, severity_key="severity", risk_key="risk_level", priority=False):
    pzem, perr = _parse_pzem()
    if perr:
        return perr
    start, serr = _parse_ts("start")
    if serr:
        return serr
    end, eerr = _parse_ts("end")
    if serr or eerr:
        return eerr or serr
    if start is not None and end is not None and start > end:
        return _err("invalid_timestamp", "start must be <= end")
    limit, lerr = _parse_limit()
    if lerr:
        return lerr
    severity = request.args.get("severity")
    risk = request.args.get("risk")
    pr = request.args.get("priority") if priority else None
    try:
        records = reader()
    except Exception:
        return _err("data_unavailable", "AI data source is unavailable", 503)
    filtered = _filter_records(records, pzem, start, end, severity, severity_key,
                               risk, pr)
    return _wrap(filtered, limit, pzem=pzem, start=start, end=end)


@bp.route("/anomalies", methods=["GET"])
def anomalies():
    return _ai_route(api_store.read_anomalies, severity_key="anomaly_severity_provisional")


@bp.route("/faults", methods=["GET"])
def faults():
    return _ai_route(api_store.read_faults, severity_key="severity")


@bp.route("/peaks", methods=["GET"])
def peaks():
    return _ai_route(api_store.read_peaks)


@bp.route("/maintenance", methods=["GET"])
def maintenance():
    return _ai_route(api_store.read_maintenance, severity_key="risk_level", risk_key="risk_level")


@bp.route("/forecast", methods=["GET"])
def forecast():
    pzem, perr = _parse_pzem()
    if perr:
        return perr
    start, serr = _parse_ts("start")
    if serr:
        return serr
    end, eerr = _parse_ts("end")
    if serr or eerr:
        return eerr or serr
    if start is not None and end is not None and start > end:
        return _err("invalid_timestamp", "start must be <= end")
    limit, lerr = _parse_limit()
    if lerr:
        return lerr
    horizon = (request.args.get("horizon") or "both").lower()
    if horizon not in ALLOWED_HORIZONS:
        return _err("invalid_horizon", f"horizon must be one of {sorted(ALLOWED_HORIZONS)}")
    try:
        records = api_store.read_forecast()
    except Exception:
        return _err("data_unavailable", "AI data source is unavailable", 503)
    filtered = _filter_records(records, pzem, start, end, None, "severity", None, None)
    if horizon in ("24h", "7d"):
        strip = "forecast_7d" if horizon == "24h" else "forecast_24h"
        for r in filtered:
            r.pop(strip, None)
    return _wrap(filtered, limit, pzem=pzem, start=start, end=end,
                 extra_meta={"horizon": horizon})


@bp.route("/bill-prediction", methods=["GET"])
def bill_prediction():
    limit, lerr = _parse_limit()
    if lerr:
        return lerr
    try:
        records = api_store.read_bill_prediction()
    except Exception:
        return _err("data_unavailable", "AI data source is unavailable", 503)
    records.sort(key=lambda r: api_store.record_timestamp(r) or 0, reverse=True)
    return _wrap(records, limit)


@bp.route("/energy-saving", methods=["GET"])
def energy_saving():
    return _ai_route(api_store.read_energy_saving, priority=True)


# ---------------------------------------------------------------------------
# Monthly reports
# ---------------------------------------------------------------------------

_MONTHLY_FILE_RE = re.compile(r"^report-\d{4}-\d{2}\.pdf$")
_SAFE_NAMES = {"latest.pdf"}


def _monthly_dir() -> str:
    root = os.environ.get("REPORTS_DIR")
    if root:
        return os.path.join(root, "monthly")
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.normpath(os.path.join(here, "Dashboard Smart-Monitoring-System", "reports", "monthly"))


@bp.route("/reports/monthly", methods=["GET"])
def reports_monthly():
    d = _monthly_dir()
    files = []
    try:
        entries = os.listdir(d) if os.path.isdir(d) else []
    except OSError:
        entries = []
    for fn in entries:
        if fn == "latest.pdf" or _MONTHLY_FILE_RE.match(fn):
            path = os.path.join(d, fn)
            m = _MONTHLY_FILE_RE.match(fn)
            year = month = None
            if m:
                year, month = (int(x) for x in m.group(0)[len("report-"):-len(".pdf")].split("-"))
            try:
                size = os.path.getsize(path)
            except OSError:
                size = None
            files.append({
                "filename": fn,
                "year": year,
                "month": month,
                "size_bytes": size,
                "url": f"/api/{API_VERSION}/reports/monthly/{fn}",
            })
    files.sort(key=lambda f: (f["year"] or 0, f["month"] or 0), reverse=True)
    return _ok(files, {"count": len(files), "storage": "local_disk_only",
                       "note": "Reports are NOT stored in Firebase RTDB."})


@bp.route("/reports/monthly/<path:filename>", methods=["GET"])
def report_download(filename: str):
    # Path-traversal safe: only whitelisted basenames under the fixed monthly dir.
    if filename != "latest.pdf" and not _MONTHLY_FILE_RE.match(filename):
        return _err("forbidden", "File not allowed", 403)
    path = os.path.normpath(os.path.join(_monthly_dir(), filename))
    allowed = os.path.normpath(_monthly_dir())
    if not path.startswith(allowed + os.sep) and path != allowed:
        return _err("forbidden", "File not allowed", 403)
    if not os.path.isfile(path):
        return _err("not_found", "Report not found", 404)
    return send_file(path, mimetype="application/pdf")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@bp.route("/summary", methods=["GET"])
def summary():
    try:
        data = api_store.build_summary()
    except Exception:
        return _err("data_unavailable", "Summary data source is unavailable", 503)
    return _ok(data)


@bp.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    # CORS preflight is handled by the after_request hook; answer OPTIONS here.
    if request.method == "OPTIONS":
        return _ok({"status": "ok"})
    try:
        payload = request.get_json(silent=True) or {}
        history = payload.get("history") or None
        result = ask_bob.ask_bob(payload.get("question", ""), history)
    except Exception:
        return _err("ask_unavailable", "Ask BOB is temporarily unavailable", 503)
    if result.get("status") == "error":
        return jsonify(result), 400
    return jsonify(result), 200


# ---------------------------------------------------------------------------
# OpenAPI / docs
# ---------------------------------------------------------------------------

def _qp(name: str, type_: str) -> dict:
    return {"name": name, "in": "query", "required": False, "schema": {"type": type_}}


@bp.route("/openapi.json", methods=["GET"])
def openapi():
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "Smart Energy Monitoring — AI Results API",
            "version": API_VERSION,
            "description": (
                "Read-only REST API exposing existing AI and energy results. "
                "SECURITY: this service has NO authentication and is intended for "
                "local/development use only. Do not expose it publicly without adding "
                "authentication, HTTPS, and rate limiting."
            ),
        },
        "servers": [{"url": f"/api/{API_VERSION}"}],
        "components": {
            "schemas": {
                "Error": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "example": "error"},
                        "error": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "message": {"type": "string"},
                            },
                        },
                    },
                },
                "ListResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "example": "ok"},
                        "data": {"type": "array", "items": {"type": "object"}},
                        "meta": {"type": "object"},
                    },
                },
            }
        },
        "paths": {
            f"/health": {"get": {"summary": "Health check", "responses": {"200": {"description": "ok"}}}},
            f"/meters": {"get": {"summary": "All meters (live)", "responses": {"200": {"description": "ok"}}}},
            f"/meters/{{pzem_number}}": {"get": {"summary": "One meter", "parameters": [{"name": "pzem_number", "in": "path", "required": True, "schema": {"type": "integer"}}], "responses": {"200": {"description": "ok"}, "404": {"description": "not found"}}}},
            f"/anomalies": {"get": {"summary": "Anomaly results", "parameters": [_qp("pzem_number", "integer"), _qp("start", "integer"), _qp("end", "integer"), _qp("limit", "integer"), _qp("severity", "string")], "responses": {"200": {"description": "ok"}}}},
            f"/faults": {"get": {"summary": "Fault results", "parameters": [_qp("pzem_number", "integer"), _qp("start", "integer"), _qp("end", "integer"), _qp("limit", "integer"), _qp("severity", "string")], "responses": {"200": {"description": "ok"}}}},
            f"/peaks": {"get": {"summary": "Peak results", "parameters": [_qp("pzem_number", "integer"), _qp("start", "integer"), _qp("end", "integer"), _qp("limit", "integer")], "responses": {"200": {"description": "ok"}}}},
            f"/maintenance": {"get": {"summary": "Maintenance risk results", "parameters": [_qp("pzem_number", "integer"), _qp("start", "integer"), _qp("end", "integer"), _qp("limit", "integer"), _qp("risk", "string")], "responses": {"200": {"description": "ok"}}}},
            f"/forecast": {"get": {"summary": "Forecast results", "parameters": [_qp("pzem_number", "integer"), _qp("start", "integer"), _qp("end", "integer"), _qp("limit", "integer"), _qp("horizon", "string")], "responses": {"200": {"description": "ok"}}}},
            f"/bill-prediction": {"get": {"summary": "Latest bill prediction", "parameters": [_qp("limit", "integer")], "responses": {"200": {"description": "ok"}}}},
            f"/energy-saving": {"get": {"summary": "Energy-saving recommendations", "parameters": [_qp("pzem_number", "integer"), _qp("start", "integer"), _qp("end", "integer"), _qp("limit", "integer"), _qp("priority", "string")], "responses": {"200": {"description": "ok"}}}},
            f"/reports/monthly": {"get": {"summary": "Monthly report metadata", "responses": {"200": {"description": "ok"}}}},
            f"/summary": {"get": {"summary": "System summary", "responses": {"200": {"description": "ok"}}}},
            f"/ask": {"post": {"summary": "Ask BOB a question", "parameters": [{"name": "question", "in": "query", "required": True, "schema": {"type": "string"}}], "responses": {"200": {"description": "ok"}}}},
        },
    }
    return jsonify(spec)


@bp.route("/docs", methods=["GET"])
def docs():
    html = (
        "<!DOCTYPE html><html><head><title>AI Results API — Docs</title>"
        "<link rel='stylesheet' href='https://unpkg.com/swagger-ui-dist@5/swagger-ui.css'></head>"
        "<body><div id='swagger'></div>"
        "<script src='https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js'></script>"
        "<script>SwaggerUIBundle({url:'/api/v1/openapi.json', dom_id:'#swagger'});</script>"
        "</body></html>"
    )
    return html


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(bp)

    # Allow the dashboard (served from a different local origin/port) to call the
    # API. This is a development convenience; lock down the origin before any
    # public deployment (see security notes in /openapi.json).
    @app.after_request
    def _cors(resp):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    return app


if __name__ == "__main__":
    # Run as a script (python ai/api_server.py) or via run_api.py. Absolute
    # import keeps the relative imports above working in both contexts.
    from ai.api_server import create_app

    create_app().run(host=os.environ.get("API_HOST", "127.0.0.1"),
                     port=int(os.environ.get("API_PORT", "8000")))
