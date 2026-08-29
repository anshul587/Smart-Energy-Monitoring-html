"""
ai/ask_bob.py
-------------
Stage 16 (enhanced): Ask BOB as a data-aware AI agent.

Flow:
  User question
    -> intent routing (casual / project / energy / mixed)
    -> for energy: select the minimum required registered tools
    -> call the Stage 15 read layer through ai.bob_tools (verified data)
    -> compose an answer from that data (Claude if configured, else rules)

The agent can only invoke the registered, validated tools in ai.bob_tools — it
never touches Firebase, SQL, the filesystem, or arbitrary URLs directly, and the
LLM only ever sees verified tool output (it never calls tools itself).

Public contract unchanged: ask_bob(question, history) returns
{"status","answer","source","intent"}, and the /api/v1/ask endpoint is untouched.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from . import bob_tools
from .config import get_settings

logger = logging.getLogger("ai.ask_bob")

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
_KNOWLEDGE_PATH = Path(__file__).resolve().parent / "project_knowledge.json"

_SYSTEM_PZEM_COUNT = 9  # fallback only; real count comes from build_summary


# ---------------------------------------------------------------------------
# Knowledge + credentials
# ---------------------------------------------------------------------------

def _load_knowledge() -> dict:
    try:
        with open(_KNOWLEDGE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ask BOB could not load project knowledge: %s", exc)
        return {}


_KNOWLEDGE = _load_knowledge()


def _get_api_key() -> str:
    try:
        settings = get_settings()
        return getattr(settings, "anthropic_api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    except Exception:  # config missing -> no key, deterministic path only
        return ""


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

_CASUAL_RE = re.compile(
    r"\b(hi|hello|hey|howdy|good morning|good afternoon|good evening)\b"
    r"|how are you|who are you|what can you do|\b(thanks|thank you|ty)\b"
    r"|\b(bye|goodbye|see you)\b", re.I)

_PROJECT_RE = re.compile(
    r"project|team member|team|developer|developed|who made|who built|who created"
    r"|purpose|problem|architecture|how does it work|how it works|how does the system"
    r"|explain|hardware|software|tech stack|technolog|ai feature|ai capabilit"
    r"|advantage|capabilit|feature|\bdashboard\b|data flow|esp32|firebase|rest api"
    r"|ai backend|offline|introduction|designed and developed", re.I)

_ENERGY_RE = re.compile(
    r"pzem\s*_?\s*\d+|meter\s*_?\s*\d+|power|fault|peak|forecast|bill|maintenance|needs attention"
    r"|save energy|energy saving|anomal|consumption|usage|voltage|current"
    r"|energy|watt|kW|offline|status|system status|condition|report|monthly", re.I)


def _detect(question: str) -> dict:
    q = question.strip()
    return {
        "casual": bool(_CASUAL_RE.search(q)),
        "project": bool(_PROJECT_RE.search(q)),
        "energy": bool(_ENERGY_RE.search(q)),
    }


def _last_pzem(history: list) -> Optional[int]:
    if not history:
        return None
    text = " ".join(str(t.get("content", "")) for t in history)
    nums = re.findall(r"pzem\s*_?\s*(\d+)", text, re.I)
    if nums:
        return int(nums[-1])
    return None


def _pzem_from_text(text: str) -> Optional[int]:
    m = re.search(r"pzem\s*_?\s*(\d+)", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"meter\s*_?\s*(\d+)", text, re.I)
    if m:
        return int(m.group(1))
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9}
    m = re.search(r"\b(?:meter|pzem)\s+(one|two|three|four|five|six|seven|eight|nine)\b",
                  text, re.I)
    if m:
        return words.get(m.group(1).lower())
    return None


def _resolve_followup(question: str, history: list) -> str:
    q = question.strip()
    if re.search(r"pzem\s*_?\s*\d+|meter\s*_?\s*\d+", q, re.I) or not history:
        return q
    if re.match(r"^(how much|how many|which one|what about|and|why|how|more|"
                r"tell me more|who|what|when|where)\b", q, re.I) and len(q) < 60:
        pz = _last_pzem(history)
        if pz:
            return f"What is the power and energy of PZEM {pz}? (follow-up: {q})"
    return q


# ---------------------------------------------------------------------------
# Tool selection (deterministic; picks the minimum required tools)
# ---------------------------------------------------------------------------

def _select_tools(question: str, history: list) -> list[tuple[str, dict]]:
    q = question.lower()
    pz = _pzem_from_text(q) or _last_pzem(history)
    plan: list[tuple[str, dict]] = []

    def add(name: str, **params: Any) -> None:
        sig = bob_tools._TOOL_PARAMS.get(name, ())
        if pz is not None and "pzem_number" in sig and "pzem_number" not in params:
            params["pzem_number"] = pz
        params = {k: v for k, v in params.items() if k in sig}
        plan.append((name, params))

    has = lambda *ws: any(w in q for w in ws)

    want_report = has("monthly report", "report")
    want_saving = has("save energy", "energy saving", "energy-saving", "recommend",
                      "reduce", "lower my bill", "lower the bill", "cut energy",
                      "save electricity", "save power")
    want_bill = has("bill", "invoice")
    want_forecast = has("forecast", "tomorrow", "next 24", "next 7", "next seven",
                        "predicted usage", "future usage", "upcoming", "what will")
    want_peaks = has("peak", "surge", "spike", "highest load")
    want_faults = has("fault", "breakdown", "error", "failure", "tripped")
    want_anomalies = has("anomal", "unusual", "abnormal", "strange", "odd")
    want_maint = has("maintenance", "risk", "attention", "watch", "health", "degrade")
    want_status = has("status", "summary", "overview", "how is", "how's", "how are the",
                      "condition", "system health", "state of", "system status")
    want_reading = has("voltage", "current", "reading", "offline", "online", "frequency")
    compare = has("most power", "highest", "uses most", "which pzem", "which meter",
                  "consume more", "consuming more", "more than", "compare", "comparison",
                  "all meter", "all meters", "rank", "difference between", "difference")
    want_power = has("power", "consum", "usage", "using", "watt", "kw", "electricity",
                     "load", "energy used", "draw")
    if compare:  # comparing meters is fundamentally a power/usage question
        want_power = True
    reason_why = has("why", "reason", "consuming more", "using more", "more power",
                     "higher power", "what happened", "happened", "what's wrong",
                     "what is wrong", "wrong", "matter", "concern", "worried",
                     "concerned", "should i", "should we")

    if want_report:
        add("get_monthly_reports")
        return _dedupe(plan)

    if want_saving:
        add("get_energy_saving")
    if want_bill:
        add("get_bill_prediction")
    if want_forecast:
        horizon = ("24h" if has("tomorrow", "next 24", "24 hours", "24h")
                   else "7d" if has("next 7", "next seven", "7 days", "7d", "week")
                   else "both")
        add("get_forecast", horizon=horizon)
    if want_peaks:
        add("get_peaks")
    if want_faults:
        add("get_faults")
    if want_anomalies:
        add("get_anomalies")

    if reason_why:
        if want_power or compare or has("consum", "power", "load"):
            if pz is not None:
                add("get_meter")
                add("get_peaks")
                add("get_anomalies")
                add("get_maintenance")
            else:
                add("get_meters")
                add("get_maintenance")
        elif want_faults or has("problem", "issue", "fault", "breakdown"):
            add("get_faults")
            add("get_anomalies")
        else:
            add("get_maintenance")
            if pz is not None:
                add("get_faults")
                add("get_anomalies")

    if want_maint:
        add("get_maintenance")
    if want_reading and pz is not None:
        add("get_meter")
    if want_power:
        if compare or pz is None:
            add("get_meters")
        elif pz is not None and not any(t == "get_meter" for t, _ in plan):
            add("get_meter")
    if want_status:
        add("get_system_summary")

    if not plan:
        add("get_system_summary")

    return _dedupe(plan)


def _dedupe(plan: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    seen = set()
    out = []
    for name, params in plan:
        key = (name, tuple(sorted(params.items())))
        if key not in seen:
            seen.add(key)
            out.append((name, params))
    return out


def _run_plan(plan: list[tuple[str, dict]]) -> dict:
    ctx = bob_tools.ToolContext()
    return {name: ctx.call(name, **params) for name, params in plan}


def _ok_results(ctx: dict) -> dict:
    return {name: r["data"] for name, r in ctx.items()
            if r.get("ok") and r.get("data") is not None}


# ---------------------------------------------------------------------------
# Casual responses
# ---------------------------------------------------------------------------

def _casual_response(question: str) -> str:
    q = question.strip().lower()
    if re.search(r"\b(hi|hello|hey|howdy|good morning|good afternoon|good evening)\b", q):
        return ("Hi! I'm BOB, your energy monitoring assistant. Ask me about your "
                "PZEM meters, faults, forecasts, bills, or energy-saving recommendations.")
    if "how are you" in q:
        return "I'm running well, thanks for asking! I'm here to help you understand your energy system."
    if "who are you" in q:
        return ("I'm BOB, the AI assistant for the Smart Monitoring System. I can explain "
                "the project, answer questions about your PZEM meters, faults, forecasts, "
                "bills, and energy-saving opportunities.")
    if "what can you do" in q:
        return ("I can help you understand your energy data, PZEM status, faults, peaks, "
                "maintenance risk, forecasts, bill predictions, and energy-saving "
                "opportunities. I can also tell you about this project and the team behind it.")
    if re.search(r"thanks|thank you", q):
        return "You're welcome!"
    if re.search(r"bye|goodbye|see you", q):
        return "Goodbye! Reach out anytime you need help with your energy data."
    return "Hi! I'm BOB, your energy monitoring assistant. How can I help?"


# ---------------------------------------------------------------------------
# Project responses (authoritative, from knowledge)
# ---------------------------------------------------------------------------

_NO_INFO = "I don't have verified information about that part of the project."

_INTRO = (
    "The Smart Monitoring System is an industrial energy-monitoring platform built by "
    "Anshul Ninawe with team members Yash Kawale, Yash Dahake, Swapnil Shendre, "
    "Chetan Bokade, and Sanjog Godbole. ESP32 boards poll PZEM energy meters across 9 "
    "circuits and publish readings to Firebase; a Python AI backend analyses the data "
    "for anomalies, faults, peaks, forecasts, bill prediction and maintenance risk, and "
    "a web dashboard shows it all live. It helps sites cut energy waste, catch faults "
    "early, and plan maintenance.")


def _project_response(question: str, k: dict) -> str:
    q = question.lower()
    team = k.get("team_members", [])
    dev = k.get("dashboard_developer") or k.get("developer", "Anshul Ninawe")

    if re.search(r"project guide|project supervisor|guide\b", q):
        return f"The project guide is {k.get('project_guide', _NO_INFO)}."
    if re.search(r"project name|name of (this|the) project|what.*project.*called", q):
        return f"The project is called {k.get('project_name', 'Smart Monitoring System')}."
    if re.search(r"who (made|built|created|programm|developed|designed).*dashboard"
                 r"|dashboard.*(made|built|programm|developed|designed)|programm", q):
        return f"The dashboard was designed, developed and programmed by {dev}."
    if re.search(r"who.*hardware|hardware.*(team|member|people|worked|setup|assembly|integration)", q):
        ht = k.get("hardware_team", team)
        return ("The hardware setup, assembly and integration was handled by: "
                + ", ".join(ht) + ".")

    if re.search(r"who made|who built|who created|developer|developed|team member|team\b", q):
        members = ", ".join(team)
        return (f"This project was designed and developed by {dev} with team members "
                f"{members}.")
    if "purpose" in q:
        return k.get("purpose", _NO_INFO)
    if "problem" in q:
        return k.get("problem_solved", _NO_INFO)
    if re.search(r"architecture|how does it work|how it works|how does the system|data flow|explain the system", q):
        return (k.get("architecture", "") + " " + k.get("data_flow", "")).strip() or _NO_INFO
    if "hardware" in q:
        return k.get("hardware", _NO_INFO)
    if re.search(r"software|tech|technolog|stack", q):
        return k.get("software_stack", _NO_INFO)
    if re.search(r"ai feature|ai capabilit", q):
        return k.get("ai_capabilities", _NO_INFO)
    if "advantage" in q:
        return k.get("advantages", _NO_INFO)
    if re.search(r"capabilit|dashboard|feature", q):
        return k.get("dashboard_capabilities", _NO_INFO)
    if "esp32" in q:
        return k.get("why_esp32", _NO_INFO)
    if "firebase" in q:
        return k.get("why_firebase", _NO_INFO)
    if "rest api" in q or "api role" in q:
        return k.get("rest_api_role", _NO_INFO)
    if "ai backend" in q:
        return k.get("ai_backend_role", _NO_INFO)
    if "offline" in q:
        return k.get("offline_behavior", _NO_INFO)
    if re.search(r"how many pzem|pzem.*used|pzem.*count|number of.*pzem", q):
        return f"The system monitors {k.get('pzem_count', _SYSTEM_PZEM_COUNT)} PZEM energy meters."
    if re.search(r"introduction|about (this|your|the) project|tell me about|describe|summar"
                 r"|what is (this|the) project|explain (this|the) project", q):
        return _INTRO
    if re.search(r"limitation|drawback|weakness", q):
        return k.get("limitations", _NO_INFO)
    return _NO_INFO


# ---------------------------------------------------------------------------
# Verified-data composer (deterministic, evidence-based)
# ---------------------------------------------------------------------------

def _fmt_ts(ms: Any) -> str:
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ms)


def _render_system_summary(s: dict, question: str = "") -> Optional[str]:
    if not isinstance(s, dict):
        return None
    parts = [f"System status is {s.get('system_status', 'unknown')} with "
             f"{s.get('online_meter_count', 0)} of {s.get('total_meter_count', 0)} meters online."]
    if s.get("total_power_w") is not None:
        parts.append(f"Total active power is {s['total_power_w']} W.")
    if s.get("total_energy_kwh") is not None:
        parts.append(f"Total energy is {s['total_energy_kwh']} kWh.")
    if s.get("average_voltage_v") is not None:
        parts.append(f"Average voltage is {s['average_voltage_v']} V.")
    if s.get("active_fault_count") is not None:
        parts.append(f"There are {s['active_fault_count']} active faults.")
    return " ".join(parts)


def _render_meters(meters: list, question: str) -> str:
    if not meters:
        return "No PZEM meter data is currently available."
    q = question.lower()
    online = [m for m in meters if m.get("online") is True]
    if "most" in q or "highest" in q or "uses most" in q or "which pzem" in q \
            or "which meter" in q or "consume more" in q or "consuming more" in q \
            or "compare" in q or "all meter" in q or "rank" in q:
        if not online:
            return "No PZEM meters are currently online, so I can't rank power use right now."
        top = max(online, key=lambda m: float(m.get("power") or 0))
        suffix = f", out of {len(online)} online meters." if len(online) > 1 else "."
        return (f"PZEM {top['pzem_number']} is using the most power right now at "
                f"{top.get('power')} W{suffix}")
    lines = [f"{len(online)} of {len(meters)} meters are online."]
    for m in sorted(online, key=lambda m: m["pzem_number"]):
        p = m.get("power")
        lines.append(f"- PZEM {m['pzem_number']}: {p} W" if p is not None else f"- PZEM {m['pzem_number']}: online")
    return "\n".join(lines)


def _render_meter(m: dict, question: str = "") -> str:
    n = m.get("pzem_number")
    if m.get("online") is not True:
        return f"PZEM {n} is currently offline (no recent reading)."
    parts = [f"PZEM {n} is online."]
    if m.get("power") is not None:
        parts.append(f"Power: {m['power']} W.")
    if m.get("energy") is not None:
        parts.append(f"Energy: {m['energy']} kWh.")
    if m.get("voltage") is not None:
        parts.append(f"Voltage: {m['voltage']} V.")
    if m.get("current") is not None:
        parts.append(f"Current: {m['current']} A.")
    if m.get("frequency") is not None:
        parts.append(f"Frequency: {m['frequency']} Hz.")
    return " ".join(parts)


def _render_faults(records: list, question: str = "") -> str:
    if not records:
        return "There are no active faults recorded."
    lines = ["Active faults:"]
    for f in records[:5]:
        n = f.get("pzem_number")
        ft = f.get("fault_type") or "fault"
        ts = _fmt_ts(f.get("timestamp"))
        lines.append(f"- PZEM {n}: {ft} (at {ts})")
    return " ".join(lines)


def _render_anomalies(records: list, question: str = "") -> str:
    if not records:
        return "No anomalies are recorded right now."
    lines = [f"{len(records)} anomaly record(s) found."]
    for a in records[:5]:
        n = a.get("pzem_number")
        label = a.get("anomaly_label") or "anomaly"
        ts = _fmt_ts(a.get("timestamp"))
        lines.append(f"- PZEM {n}: {label} (at {ts})" if n is not None else f"- {label} (at {ts})")
    return " ".join(lines)


def _render_peaks(records: list, question: str = "") -> str:
    if not records:
        return "No peak data is available right now."
    p = records[0]
    tp = p.get("total_peak_power_w")
    if tp is None:
        return "No peak data is available right now."
    dom = p.get("dominant_pzems")
    ts = _fmt_ts(p.get("timestamp"))
    dom_s = f" (dominant: PZEM {dom})" if dom else ""
    return f"Latest system peak was {tp} W at {ts}{dom_s}."


def _render_maintenance(records: list, question: str = "") -> str:
    if not records:
        return "No maintenance-risk data is available right now."
    sys_rec = next((r for r in records if r.get("pzem_number") is None), None)
    if sys_rec:
        hr = len(sys_rec.get("high_risk_meters", []) or [])
        wm = len(sys_rec.get("watch_meters", []) or [])
        hr_pz = sys_rec.get("highest_risk_pzem")
        parts = [f"Maintenance: {hr} high-risk and {wm} watch meters."]
        if hr_pz is not None:
            parts.append(f"Highest-risk meter is PZEM {hr_pz}.")
        return " ".join(parts)
    lines = ["Maintenance indicators:"]
    for r in records[:5]:
        n = r.get("pzem_number")
        lvl = r.get("risk_level")
        lines.append(f"- PZEM {n}: {lvl}")
    return " ".join(lines)


def _render_forecast(records: list, question: str = "") -> str:
    if not records:
        return "No forecast is available right now."
    r = records[0]
    parts = ["A power forecast is available."]
    f24 = r.get("forecast_24h")
    f7 = r.get("forecast_7d")
    if isinstance(f24, (int, float)):
        parts.append(f"24h forecast around {f24} W.")
    if isinstance(f7, (int, float)):
        parts.append(f"7d forecast around {f7} W.")
    return " ".join(parts)


def _render_bill(records: list, question: str = "") -> str:
    if not records:
        return "No bill prediction is available right now."
    b = records[0]
    est = b.get("estimated_bill")
    if est is None:
        return "No bill prediction is available right now."
    ts = _fmt_ts(b.get("anchor_timestamp"))
    return f"Latest predicted bill is {est} (as of {ts})."


def _render_energy_saving(records: list, question: str = "") -> str:
    if not records:
        return "No energy-saving recommendations are available right now."
    r = records[0]
    cnt = r.get("recommendation_count")
    recs = r.get("recommendations") or []
    parts = []
    if cnt:
        parts.append(f"There are {cnt} energy-saving recommendations.")
    for x in [x for x in recs if isinstance(x, dict)][:3]:
        pz = x.get("pzem_number")
        txt = x.get("recommendation") or x.get("text") or x.get("action")
        if txt:
            parts.append(f"- PZEM {pz}: {txt}" if pz else f"- {txt}")
    return " ".join(parts) if parts else "Energy-saving recommendations are available."


def _render_monthly_reports(files: list, question: str = "") -> str:
    if not files:
        return "No monthly reports are available."
    names = [f.get("filename") for f in files]
    return "Available monthly reports: " + ", ".join(names) + "."


_RENDERERS = {
    "get_system_summary": _render_system_summary,
    "get_meters": _render_meters,
    "get_meter": _render_meter,
    "get_faults": _render_faults,
    "get_anomalies": _render_anomalies,
    "get_peaks": _render_peaks,
    "get_maintenance": _render_maintenance,
    "get_forecast": _render_forecast,
    "get_bill_prediction": _render_bill,
    "get_energy_saving": _render_energy_saving,
    "get_monthly_reports": _render_monthly_reports,
}

_ORDER = ["get_system_summary", "get_meters", "get_meter", "get_faults",
          "get_anomalies", "get_peaks", "get_maintenance", "get_forecast",
          "get_bill_prediction", "get_energy_saving", "get_monthly_reports"]


def _has_data(v: Any) -> bool:
    if isinstance(v, list):
        return len(v) > 0
    if isinstance(v, dict):
        return len(v) > 0
    return v is not None


def _compose(question: str, results: dict) -> str:
    results = {k: v for k, v in results.items() if _has_data(v)}
    if not results:
        return "I don't have enough current data to answer that."
    pieces = []
    for name in _ORDER:
        if name not in results:
            continue
        rendered = _RENDERERS[name](results[name], question)
        if rendered:
            pieces.append(rendered)
    if not pieces:
        return "I don't have enough current data to answer that."
    return " ".join(pieces)


# ---------------------------------------------------------------------------
# LLM (server-side, optional) — composes from verified tool data only
# ---------------------------------------------------------------------------

def _llm_compose(question: str, results: dict, history: list, api_key: str) -> Optional[str]:
    try:
        import anthropic
    except ImportError:
        return None
    try:
        system = (
            "You are BOB, the conversational AI assistant for the Smart Monitoring System, "
            "an industrial energy-monitoring project. Answer naturally and professionally. "
            "Use ONLY the provided verified tool-data JSON. Never invent meter readings, "
            "faults, predictions, timestamps, savings, or project facts. When quoting data, "
            "include the PZEM number, value and timestamp as evidence. If the data is empty "
            "or missing, say you don't have enough current data. Keep responses concise "
            "(under 180 words). Avoid saying 'according to my database'."
        )
        messages = []
        for turn in (history or [])[-4:]:
            role = "assistant" if turn.get("role") == "bot" else "user"
            content = turn.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        ctx_text = json.dumps(results, default=str)
        messages.append({"role": "user",
                         "content": f"Question: {question}\n\nVerified tool data:\n{ctx_text}"})
        client = anthropic.Anthropic(api_key=api_key)
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        resp = client.messages.create(model=model, max_tokens=500, system=system, messages=messages)
        answer = "".join(getattr(b, "text", "") for b in resp.content).strip()
        return answer or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ask BOB Claude compose failed; using deterministic path: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def ask_bob(question: str, history: Optional[list] = None) -> dict[str, Any]:
    question = (question or "").strip()
    if not question:
        return {"status": "error", "error": {"code": "empty_question",
                "message": "Please enter a question."}, "answer": None}

    resolved = _resolve_followup(question, history or [])
    flags = _detect(resolved)
    is_energy = flags["energy"] or "follow-up" in resolved.lower()
    is_project = flags["project"]
    is_casual = flags["casual"]

    # Mixed: project knowledge + verified live data.
    if is_project and is_energy:
        project_part = _project_response(resolved, _KNOWLEDGE)
        ctx = _run_plan(_select_tools(resolved, history or []))
        results = _ok_results(ctx)
        api_key = _get_api_key()
        if api_key and results:
            energy_part = _llm_compose(resolved, results, history or [], api_key) or _compose(resolved, results)
        else:
            energy_part = _compose(resolved, results)
        if not energy_part.strip():
            energy_part = "I don't have enough current data to answer that."
        return {"status": "ok", "answer": f"{project_part}\n\n{energy_part}",
                "source": "mixed", "intent": "project+energy"}

    # Energy / live data.
    if is_energy:
        ctx = _run_plan(_select_tools(resolved, history or []))
        results = _ok_results(ctx)
        api_key = _get_api_key()
        if api_key and results:
            ans = _llm_compose(resolved, results, history or [], api_key)
            if ans:
                return {"status": "ok", "answer": ans, "source": "llm", "intent": "energy"}
        return {"status": "ok", "answer": _compose(resolved, results),
                "source": "tool", "intent": "energy"}

    # Project knowledge only.
    if is_project:
        return {"status": "ok", "answer": _project_response(resolved, _KNOWLEDGE),
                "source": "project", "intent": "project"}

    # Casual.
    if is_casual:
        return {"status": "ok", "answer": _casual_response(question),
                "source": "casual", "intent": "casual"}

    # Unmatched: safe, non-fabricating helper.
    return {"status": "ok",
            "answer": ("I'm BOB, your energy assistant. I can answer questions about your "
                       "PZEM meters, faults, forecasts, bills and energy saving, or tell you "
                       "about this project and the team behind it."),
            "source": "casual", "intent": "unknown"}
