"""
ai/ask_bob.py
-------------
Stage 16 (enhanced): Ask BOB as a genuinely conversational AI agent.

Three-layer response architecture:
  A. GENERAL CONVERSATION — natural, ChatGPT-like responses for greetings,
     casual chat, general knowledge, follow-ups. Uses LLM when configured,
     deterministic fallback otherwise.
  B. VERIFIED PROJECT KNOWLEDGE — authoritative facts from project_knowledge.json.
     Never invented. Covers: team, guide, purpose, architecture, hardware, etc.
  C. VERIFIED LIVE ENERGY DATA — current sensor values, faults, forecasts, bills,
     maintenance, energy-saving. Only via registered bob_tools. Never fabricated.

Routing is lightweight and semantic. No fixed question list. Mixed questions
(project + live, casual + project, etc.) are composed naturally.

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

_KNOWLEDGE_PATH = Path(__file__).resolve().parent / "project_knowledge.json"

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
# Semantic intent classification (lightweight, no giant keyword dict)
# ---------------------------------------------------------------------------

# Pre-compiled patterns for fast routing hints (not exhaustive filters)
_CASUAL_HINTS = re.compile(
    r"\b(hi|hello|hey|howdy|good\s+(morning|afternoon|evening|night))\b"
    r"|how are you|who are you|what can you do|\b(thanks|thank you|ty)\b"
    r"|\b(bye|goodbye|see you)\b"
    r"|what is (ai|artificial intelligence|iot|internet of things|energy efficiency)"
    r"|explain|tell me something"
    r"|why is .* important"
    r"|interesting", re.I)

_PROJECT_HINTS = re.compile(
    r"project|team member|team|developer|developed|who (made|built|created|programm|designed)"
    r"|purpose|problem|architecture|how does it work|how it works|how does the system"
    r"|explain.*(project|system)|hardware|software|tech stack|technolog|ai feature|ai capabilit"
    r"|advantage|capabilit|feature|\bdashboard\b|data flow|esp32|firebase|rest api"
    r"|ai backend|offline|introduction|designed and developed"
    r"|guide|supervisor|Anshul|Yash|Swapnil|Chetan|Sanjog", re.I)

_ENERGY_HINTS = re.compile(
    r"pzem\s*_?\s*\d+|meter\s*_?\s*\d+|power|fault|peak|forecast|bill|maintenance|needs attention"
    r"|save energy|energy saving|energy-saving|recommend|reduce|lower.*bill|cut energy"
    r"|save electricity|save power|anomal|consumption|usage|voltage|current"
    r"|energy|watt|kw|offline|status|condition|report|monthly|which (pzem|meter)"
    r"|most power|highest|compare|comparison|rank|consuming|using|draw|load"
    r"|how much|how many", re.I)

_FOLLOWUP_HINTS = re.compile(
    r"^(why|how|what|when|where|who|which|how much|how many|tell me more|more|how much)"
    r"|^(and|but|also|then|so)"
    r"|^(it|he|she|they|that|this)\b", re.I)


def _detect_intent(question: str, history: list) -> dict[str, bool]:
    """Lightweight semantic intent detection. Returns flags for each layer."""
    q = question.strip()
    q_lower = q.lower()

    # Check for follow-up first (short, context-dependent)
    is_followup = bool(
        _FOLLOWUP_HINTS.search(q_lower)
        and len(q) < 80
        and history
    )

    # Primary intent hints
    casual = bool(_CASUAL_HINTS.search(q))
    project = bool(_PROJECT_HINTS.search(q))
    energy = bool(_ENERGY_HINTS.search(q))

    # Follow-ups inherit energy/project context from history
    if is_followup and not (casual or project or energy):
        last_q = ""
        for turn in reversed(history):
            if turn.get("role") == "user":
                last_q = turn.get("content", "")
                break
        if last_q:
            casual |= bool(_CASUAL_HINTS.search(last_q))
            project |= bool(_PROJECT_HINTS.search(last_q))
            energy |= bool(_ENERGY_HINTS.search(last_q))

    return {
        "casual": casual,
        "project": project,
        "energy": energy,
        "followup": is_followup,
    }


def _last_mentioned_pzem(history: list) -> Optional[int]:
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
    """Enrich short follow-up questions with context from history."""
    q = question.strip()
    if re.search(r"pzem\s*_?\s*\d+|meter\s*_?\s*\d+", q, re.I) or not history:
        return q

    # Short follow-up that likely refers to previous context
    if _FOLLOWUP_HINTS.search(q) and len(q) < 80:
        pz = _last_mentioned_pzem(history)
        if pz:
            return f"Regarding PZEM {pz}: {q}"
        # Check if last bot answer mentioned a specific meter
        for turn in reversed(history):
            if turn.get("role") == "bot":
                content = turn.get("content", "")
                pz = _pzem_from_text(content)
                if pz:
                    return f"Regarding PZEM {pz}: {q}"
                break
    return q


# ---------------------------------------------------------------------------
# Tool selection (deterministic; picks minimum required tools)
# ---------------------------------------------------------------------------

def _select_tools(question: str, history: list) -> list[tuple[str, dict]]:
    q = question.lower()
    pz = _pzem_from_text(q) or _last_mentioned_pzem(history)
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
    want_reading = has("voltage", "current", "reading", "offline", "online", "frequency",
                       "how much", "how many")
    compare = has("most power", "highest", "uses most", "which pzem", "which meter",
                  "consume more", "consuming more", "more than", "compare", "comparison",
                  "all meter", "all meters", "rank", "difference between", "difference")
    want_power = has("power", "consum", "usage", "using", "watt", "kw", "electricity",
                     "load", "energy used", "draw", "how much", "how many")
    if compare:
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
# Layer A: General Conversation (LLM + deterministic fallback)
# ---------------------------------------------------------------------------

_CASUAL_RESPONSES = {
    "greeting": (
        "Hi! I'm BOB, your energy monitoring assistant. Ask me about your "
        "PZEM meters, faults, forecasts, bills, or energy-saving recommendations."
    ),
    "how_are_you": "I'm running well, thanks for asking! I'm here to help you understand your energy system.",
    "who_are_you": (
        "I'm BOB, the AI assistant for the Smart Energy Monitoring System. I can explain "
        "the project, answer questions about your PZEM meters, faults, forecasts, "
        "bills, and energy-saving opportunities."
    ),
    "what_can_you_do": (
        "I can help you understand your energy data, PZEM status, faults, peaks, "
        "maintenance risk, forecasts, bill predictions, and energy-saving "
        "opportunities. I can also tell you about this project and the team behind it."
    ),
    "thanks": "You're welcome!",
    "goodbye": "Goodbye! Reach out anytime you need help with your energy data.",
    "default": (
        "Hi! I'm BOB, your energy monitoring assistant. How can I help?"
    ),
}


def _classify_casual(question: str) -> str:
    q = question.strip().lower()
    if re.search(r"\b(hi|hello|hey|howdy|good\s+(morning|afternoon|evening|night))\b", q):
        return "greeting"
    if "how are you" in q:
        return "how_are_you"
    if "who are you" in q:
        return "who_are_you"
    if "what can you do" in q:
        return "what_can_you_do"
    if re.search(r"thanks|thank you", q):
        return "thanks"
    if re.search(r"bye|goodbye|see you", q):
        return "goodbye"
    return "default"


def _casual_response(question: str) -> str:
    """Deterministic fallback for casual conversation when LLM unavailable."""
    return _CASUAL_RESPONSES[_classify_casual(question)]


def _llm_general_conversation(question: str, history: list, api_key: str) -> Optional[str]:
    """Use LLM for natural general conversation. Only for non-project, non-energy topics."""
    try:
        import anthropic
    except ImportError:
        return None
    try:
        system = (
            "You are BOB, a friendly and knowledgeable AI assistant for the Smart Energy Monitoring System. "
            "Answer naturally and conversationally. Keep responses concise (under 150 words). "
            "You can discuss general topics: greetings, how things work, energy concepts, IoT, AI, etc. "
            "Do NOT invent project-specific facts, sensor values, team members, or hardware specs. "
            "If asked about the project, meters, or live data, say you'll check the verified sources. "
            "Be helpful, concise, and natural."
        )
        messages = []
        for turn in (history or [])[-4:]:
            role = "assistant" if turn.get("role") == "bot" else "user"
            content = turn.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})
        client = anthropic.Anthropic(api_key=api_key)
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        resp = client.messages.create(model=model, max_tokens=400, system=system, messages=messages)
        answer = "".join(getattr(b, "text", "") for b in resp.content).strip()
        return answer or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ask BOB LLM general conversation failed; using fallback: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Layer B: Verified Project Knowledge (authoritative, from knowledge)
# ---------------------------------------------------------------------------

_NO_INFO = "I don't have verified information about that part of the project."

_INTRO = (
    "The Smart Energy Monitoring System is an industrial energy-monitoring platform built by "
    "Anshul Ninawe with team members Yash Kawale, Yash Dahake, Swapnil Shendre, "
    "Chetan Bokade, and Sanjog Godbole. ESP32 boards poll PZEM energy meters across 9 "
    "circuits and publish readings to Firebase; a Python AI backend analyses the data "
    "for anomalies, faults, peaks, forecasts, bill prediction and maintenance risk, and "
    "a web dashboard shows it all live. It helps sites cut energy waste, catch faults "
    "early, and plan maintenance."
)


def _project_response(question: str, k: dict) -> str:
    q = question.lower()
    team = k.get("team_members", [])
    dev = k.get("dashboard_developer") or k.get("developer", "Anshul Ninawe")

    # "Who built/created YOU?" — distinguish assistant from project
    if re.search(r"who (built|created|developed|made) you\b", q):
        return (
            "I'm BOB, the AI assistant integrated into the Smart Energy Monitoring System. "
            "This project and dashboard were designed, developed and programmed by Anshul Ninawe. "
            "The underlying AI model is provided by Anthropic (Claude)."
        )

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

    # "Who is [Name]?" for team members
    if re.search(r"who is (anshul|yash|swapnil|chetan|sanjog)", q):
        name_match = re.search(r"who is (anshul|yash|swapnil|chetan|sanjog)", q)
        if name_match:
            name = name_match.group(1).title()
            if name == "Anshul":
                return f"{name} Ninawe is the developer who designed, developed and programmed this dashboard and system."
            elif name in ["Yash", "Swapnil", "Chetan", "Sanjog"]:
                return f"{name} is a team member who worked on the hardware setup and integration."
        return _NO_INFO

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
        return f"The system monitors {k.get('pzem_count', 9)} PZEM energy meters."
    if re.search(r"introduction|about (this|your|the) project|tell me about|describe|summar"
                 r"|what is (this|the) project|explain (this|the) project", q):
        return _INTRO
    if re.search(r"limitation|drawback|weakness", q):
        return k.get("limitations", _NO_INFO)

    return _NO_INFO


# ---------------------------------------------------------------------------
# Layer C: Verified Live Energy Data (deterministic, evidence-based)
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


def _compose_energy(question: str, results: dict) -> str:
    """Deterministic composer for live energy data."""
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


def _llm_compose_energy(question: str, results: dict, history: list, api_key: str) -> Optional[str]:
    """LLM composes natural answer from verified tool data only."""
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


# Backward-compat alias for tests
_llm_compose = _llm_compose_energy


# ---------------------------------------------------------------------------
# Orchestration: Three-layer architecture
# ---------------------------------------------------------------------------

def ask_bob(question: str, history: Optional[list] = None) -> dict[str, Any]:
    question = (question or "").strip()
    if not question:
        return {"status": "error", "error": {"code": "empty_question",
                "message": "Please enter a question."}, "answer": None}

    history = history or []
    resolved = _resolve_followup(question, history)
    flags = _detect_intent(resolved, history)

    is_casual = flags["casual"]
    is_project = flags["project"]
    is_energy = flags["energy"]
    is_followup = flags["followup"]

    api_key = _get_api_key()
    has_llm = bool(api_key)

    # ---------------------------------------------------------
    # MIXED: Project knowledge + Live energy data
    # ---------------------------------------------------------
    if is_project and is_energy:
        project_part = _project_response(resolved, _KNOWLEDGE)
        ctx = _run_plan(_select_tools(resolved, history))
        results = _ok_results(ctx)

        if has_llm and results:
            energy_part = _llm_compose_energy(resolved, results, history, api_key) or _compose_energy(resolved, results)
        else:
            energy_part = _compose_energy(resolved, results)

        if not energy_part.strip():
            energy_part = "I don't have enough current data to answer that."

        return {"status": "ok", "answer": f"{project_part}\n\n{energy_part}",
                "source": "mixed", "intent": "project+energy"}

    # ---------------------------------------------------------
    # MIXED: Casual + Project knowledge
    # ---------------------------------------------------------
    if is_casual and is_project:
        casual_part = _casual_response(question) if not has_llm else None
        project_part = _project_response(resolved, _KNOWLEDGE)
        if has_llm:
            # Use LLM to compose natural mixed response
            ans = _llm_general_conversation(resolved, history, api_key)
            if ans:
                return {"status": "ok", "answer": ans, "source": "llm", "intent": "casual+project"}
        # Fallback: compose manually
        if casual_part and project_part != _NO_INFO:
            return {"status": "ok", "answer": f"{casual_part}\n\n{project_part}",
                    "source": "mixed", "intent": "casual+project"}
        elif project_part != _NO_INFO:
            return {"status": "ok", "answer": project_part, "source": "project", "intent": "project"}

    # ---------------------------------------------------------
    # LIVE ENERGY DATA only
    # ---------------------------------------------------------
    if is_energy or (is_followup and not is_casual and not is_project):
        ctx = _run_plan(_select_tools(resolved, history))
        results = _ok_results(ctx)

        if has_llm and results:
            ans = _llm_compose_energy(resolved, results, history, api_key)
            if ans:
                return {"status": "ok", "answer": ans, "source": "llm", "intent": "energy"}

        return {"status": "ok", "answer": _compose_energy(resolved, results),
                "source": "tool", "intent": "energy"}

    # ---------------------------------------------------------
    # PROJECT KNOWLEDGE only
    # ---------------------------------------------------------
    if is_project:
        answer = _project_response(resolved, _KNOWLEDGE)
        if answer == _NO_INFO:
            # Not in knowledge base - try LLM for general knowledge if available
            if has_llm:
                ans = _llm_general_conversation(resolved, history, api_key)
                if ans:
                    return {"status": "ok", "answer": ans, "source": "llm", "intent": "general"}
            return {"status": "ok", "answer": _NO_INFO, "source": "project", "intent": "project"}
        return {"status": "ok", "answer": answer, "source": "project", "intent": "project"}

    # ---------------------------------------------------------
    # GENERAL CONVERSATION (casual, general knowledge)
    # ---------------------------------------------------------
    if is_casual or not (is_project or is_energy):
        # Try LLM first for natural conversation
        if has_llm:
            ans = _llm_general_conversation(resolved, history, api_key)
            if ans:
                return {"status": "ok", "answer": ans, "source": "llm", "intent": "casual"}

        # Deterministic fallback
        return {"status": "ok", "answer": _casual_response(question),
                "source": "casual", "intent": "casual"}

    # ---------------------------------------------------------
    # Should not reach here, but safe fallback
    # ---------------------------------------------------------
    return {"status": "ok",
            "answer": ("I'm BOB, your energy assistant. I can answer questions about your "
                       "PZEM meters, faults, forecasts, bills and energy saving, or tell you "
                       "about this project and the team behind it."),
            "source": "casual", "intent": "unknown"}