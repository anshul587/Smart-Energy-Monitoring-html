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
        provider = getattr(settings, "llm_provider", "openrouter")
        if provider == "anthropic":
            return getattr(settings, "anthropic_api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        # openrouter (default) ---
        return getattr(settings, "open_router_api_key", "") or os.environ.get("OPENROUTER_API_KEY", "")
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
    r"pzem[\s_-]*\d+|meter[\s_-]*\d+|power|fault|peak|forecast|bill|maintenance|needs attention"
    r"|save energy|energy saving|energy-saving|recommend|reduce|lower.*bill|cut energy"
    r"|save electricity|save power|anomal|consumption|usage|voltage|current"
    r"|energy|watt|kw|offline|status|condition|report|monthly|which (pzem|meter)"
    r"|most power|highest|compare|comparison|rank|consuming|using|draw|load"
    r"|how much|how many|diagnostic|recommendation|what to do|what should|maintenance required",
    re.I)

_HISTORICAL_HINTS = re.compile(
    r"\b(last|previous|yesterday|kal|aaj|today|this\s+week|last\s+week|previous\s+week|last\s+month|previous\s+month|"
    r"\d+\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|"
    r"july|august|september|october|november|december)|"
    r"\d+\s*(day|week|month|year)|"
    r"\d+\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|"
    r"10\s*september|10\s*sep|yesterday|kal|aaj)\b", re.I)

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
    if not energy:
        energy = bool(re.search(r"action\s*(lena|karna|kari|karna|chahiye)", q, re.I))

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
    m = re.search(r"pzem[\s_-]*(\d+)", text, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"meter[\s_-]*(\d+)", text, re.I)
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
# Date extraction for historical queries
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}


def _date_from_text(text: str) -> Optional[dict]:
    """Extract date info from natural language text.

    Returns dict with 'start' and 'end' unix timestamps, or None if unresolved.
    Supports: '10 September', '10 Sep', 'yesterday'/'kal', 'today'/'aaj'.
    """
    from datetime import datetime, timezone, timedelta
    q = text.strip().lower()
    now = datetime.now(timezone.utc)

    if q in ("yesterday", "kal"):
        d = now.date() - timedelta(days=1)
        return {"start": int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()),
                "end": int(datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc).timestamp())}

    if q in ("today", "aaj"):
        d = now.date()
        return {"start": int(datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc).timestamp()),
                "end": int(datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc).timestamp())}

    m = re.search(r"(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)", q, re.I)
    if m:
        day = int(m.group(1))
        month_name = m.group(2).lower()
        month = _MONTH_MAP.get(month_name)
        if month:
            try:
                d = datetime(now.year, month, day, tzinfo=timezone.utc)
                return {"start": int(datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc).timestamp()),
                        "end": int(datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc).timestamp())}
            except ValueError:
                return None

    m = re.search(r"last\s+(\d+)\s*day", q, re.I)
    if m:
        n = int(m.group(1))
        end = now.date()
        start = end - timedelta(days=n)
        return {"start": int(datetime(start.year, start.month, start.day, 0, 0, 0, tzinfo=timezone.utc).timestamp()),
                "end": int(datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc).timestamp())}

    return None


def _resolve_date_range(q: str, history: list) -> Optional[dict]:
    """Resolve date range from question, checking history if needed."""
    result = _date_from_text(q)
    if result:
        return result
    for turn in reversed(history):
        if turn.get("role") == "user":
            hr = _date_from_text(turn.get("content", ""))
            if hr:
                return hr
    return None


# ---------------------------------------------------------------------------
# Tool selection (deterministic; picks minimum required tools)
# ---------------------------------------------------------------------------

def _select_tools(question: str, history: list) -> list[tuple[str, dict]]:
    q = question.lower()
    pz = _pzem_from_text(q) or _last_mentioned_pzem(history)
    date_range = _resolve_date_range(q, history)
    plan: list[tuple[str, dict]] = []

    def add(name: str, **params: Any) -> None:
        sig = bob_tools._TOOL_PARAMS.get(name, ())
        if pz is not None and "pzem_number" in sig and "pzem_number" not in params:
            params["pzem_number"] = pz
        params = {k: v for k, v in params.items() if k in sig}
        plan.append((name, params))

    has = lambda *ws: any(w in q for w in ws)

    want_historical = bool(date_range) or has(
        "last", "previous", "yesterday", "kal", "aaj", "this week",
        "last week", "previous week", "last month", "previous month",
        "10 september", "10 sep", "average power", "max power", "min power",
        "energy consumption", "consumed", "maximum current", "maximum power",
        "highest power", "lowest power", "daily", "hourly", "trend",
    )

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
                        "concerned", "should i", "should we", "kyu", "kyu hai",
                        "kyu hua", "kyu tha", "kyu increase", "kya kyu",
                        "kaise reduce", "kaise kam", "kaise kare")
    want_diagnostic = has("diagnostic", "recommendation", "what to do", "what should",
                           "maintenance required", "probable cause", "probable_cause",
                           "what to check", "what_to_check", "corrective action",
                           "corrective_action", "urgency", "fault kya hai",
                           "problem kya hai", "kya problem hai", "ka fault",
                           "mein kya karna", "mujhe kya karna", "kya karna chahiye",
                           "action lena", "kya maintenance", "reason kya hai",
                           "ka reason", "kyu problem", "problem kyu",
                           "kya karu", "kya action lena chahiye",
                           "fault kya hai", "mein koi fault")
    if not want_diagnostic and reason_why and (pz is not None or want_faults
            or has("problem", "issue", "fault", "breakdown", "error")):
        want_diagnostic = True

    if want_report:
        add("get_monthly_reports")
        return _dedupe(plan)

    if want_historical and pz is not None:
        add("get_historical_analysis",
            pzem_number=pz,
            start=date_range.get("start") if date_range else None,
            end=date_range.get("end") if date_range else None)
    elif want_historical and pz is None:
        add("get_historical_analysis",
            start=date_range.get("start") if date_range else None,
            end=date_range.get("end") if date_range else None)

    if want_saving:
        add("get_energy_saving")
    if want_bill:
        add("get_bill_prediction")
    if want_forecast:
        horizon = ("24h" if has("tomorrow", "next 24", "24 hours", "24h")
                   else "7d" if has("next 7", "next seven", "7 days", "7d", "week")
                   else "both")
        add("get_forecast", horizon=horizon)
    if want_peaks and not want_historical:
        add("get_peaks")
    if want_faults:
        add("get_faults")
        add("get_diagnostic_recommendations")
    if want_anomalies:
        add("get_anomalies")
    if want_diagnostic:
        add("get_diagnostic_recommendations")
        if pz is not None and not want_historical:
            add("get_meter")
            add("get_faults")
            add("get_anomalies")

    if reason_why and not want_historical and not want_diagnostic and not want_bill and not want_saving and not want_forecast:
        if want_power or compare or has("consum", "power", "load"):
            if pz is not None:
                add("get_meter")
                add("get_peaks")
                add("get_anomalies")
                add("get_maintenance")
                add("get_diagnostic_recommendations")
            else:
                add("get_meters")
                add("get_maintenance")
                add("get_diagnostic_recommendations")
                add("get_system_summary")
        elif want_faults or has("problem", "issue", "fault", "breakdown"):
            add("get_faults")
            add("get_anomalies")
            if pz is not None:
                add("get_diagnostic_recommendations")
        else:
            add("get_maintenance")
            if pz is not None:
                add("get_faults")
                add("get_anomalies")
                add("get_diagnostic_recommendations")
            else:
                add("get_system_summary")
                add("get_diagnostic_recommendations")

    if want_maint and not want_historical:
        add("get_maintenance")
    if want_reading and pz is not None and not want_historical:
        add("get_meter")
    if want_power and not want_historical:
        if compare or pz is None:
            add("get_meters")
        elif pz is not None and not any(t == "get_meter" for t, _ in plan):
            add("get_meter")
    if want_status and not want_historical:
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


def _llm_general_conversation(question: str, history: list, api_key: str, provider: str = "openrouter") -> Optional[str]:
    """Use LLM for natural general conversation. Only for non-project, non-energy topics."""
    try:
        import openai as _openai
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
        if provider == "openrouter":
            client = _openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            model = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
            sys_msg = {"role": "system", "content": system}
            user_msgs = [msg for msg in messages if msg.get("role") == "user"]
            combined = [sys_msg] + user_msgs if user_msgs else [sys_msg] + messages
            resp = client.chat.completions.create(model=model, max_tokens=400, messages=combined)  # type: ignore
            answer = resp.choices[0].message.content.strip()
        else:
            # anthropic fallback
            import anthropic  # type: ignore
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
    "The Smart Energy Monitoring System is an energy-monitoring platform built by "
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


def _fmt_ts_s(sec: Any) -> str:
    """Format a Unix-seconds timestamp into a human-readable date string."""
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(int(sec), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(sec)


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


def _render_historical(result: dict, question: str = "") -> str:
    """Render Stage 3 historical analysis results naturally."""
    if not result or not isinstance(result, dict):
        return "No historical data is available."
    status = result.get("status")
    if status == "ERROR":
        return f"Historical analysis could not be completed: {result.get('reason', 'unknown error')}."
    if status == "NO_DATA":
        return "No historical data is available for the requested period."
    if status == "INSUFFICIENT_DATA":
        return f"Insufficient historical data for the requested period: {result.get('reason', '')}".strip()
    if status != "OK":
        return "Historical data is unavailable."

    parts = []
    pz = result.get("pzem_number")
    if pz:
        parts.append(f"PZEM {pz} historical analysis:")
    else:
        parts.append("System-wide historical analysis:")

    start = result.get("requested_start")
    end = result.get("requested_end")
    if start and end:
        parts.append(f"Period: {_fmt_ts_s(start)} to {_fmt_ts_s(end)}")

    if "available_days" in result and result["available_days"]:
        parts.append(f"Data span: {result['available_days']:.1f} days ({result.get('sample_count', 0)} samples)")

    power = result.get("power")
    if power:
        parts.append(f"Average power: {power.get('average', 'N/A')} W")
        if power.get("maximum") is not None:
            parts.append(f"Maximum power: {power['maximum']} W")
        if power.get("minimum") is not None:
            parts.append(f"Minimum power: {power['minimum']} W")
        if power.get("max_timestamp") is not None:
            parts.append(f"Peak at: {_fmt_ts(power['max_timestamp'])}")

    voltage = result.get("voltage")
    if voltage and voltage.get("average") is not None:
        parts.append(f"Average voltage: {voltage['average']} V")

    current = result.get("current")
    if current and current.get("average") is not None:
        parts.append(f"Average current: {current['average']} A")

    freq = result.get("frequency")
    if freq and freq.get("average") is not None:
        parts.append(f"Average frequency: {freq['average']} Hz")

    pf = result.get("pf")
    if pf and pf.get("average") is not None:
        parts.append(f"Average power factor: {pf['average']:.2f}")

    energy = result.get("energy_consumption")
    if energy and energy.get("consumption_kwh") is not None:
        parts.append(f"Energy consumed: {energy['consumption_kwh']:.4f} kWh")

    trend = result.get("trend")
    if trend and trend.get("power_trend_per_hour") is not None:
        parts.append(f"Power trend: {trend['power_trend_per_hour']:.3f} W/h")

    daily = result.get("daily")
    if daily:
        parts.append(f"Daily breakdown ({len(daily)} day(s)):")
        for d in daily[:7]:
            day_line = f"  {d['date']}: avg={d.get('power_avg', 'N/A')}W max={d.get('power_max', 'N/A')}W"
            if d.get('energy_consumption_kwh') is not None:
                day_line += f" energy={d['energy_consumption_kwh']:.4f}kWh"
            parts.append(day_line)

    hourly = result.get("hourly")
    if hourly:
        parts.append(f"Hourly breakdown ({len(hourly)} hour(s) with data):")
        for h in hourly[:12]:
            parts.append(f"  Hour {h['hour']:02d}: avg={h.get('power_avg', 'N/A')}W max={h.get('power_max', 'N/A')}W (n={h['sample_count']})")

    return " ".join(parts) if parts else "No detailed historical data available."


def _render_monthly_reports(files: list, question: str = "") -> str:
    if not files:
        return "No monthly reports are available."
    names = [f.get("filename") for f in files]
    return "Available monthly reports: " + ", ".join(names) + "."


def _render_diagnostic_recommendations(records: list, question: str = "") -> str:
    if not records:
        return "No verified diagnostic recommendation is currently available."
    parts = []
    for rec in records[:5]:
        pz = rec.get("pzem_number")
        pz_label = f"PZEM {pz}" if pz is not None else "SYSTEM"
        parts.append(f"{pz_label}:")
        if rec.get("fault_type") is not None:
            parts.append(f"  Fault / Condition: {rec['fault_type']}")
        if rec.get("severity") is not None:
            parts.append(f"  Severity: {rec['severity']}")
        if rec.get("priority") is not None:
            parts.append(f"  Priority: {rec['priority']}")
        if rec.get("probable_cause") is not None:
            parts.append(f"  Probable Cause: {rec['probable_cause']}")
        if rec.get("evidence") is not None:
            parts.append(f"  Evidence: {rec['evidence']}")
        if rec.get("confidence") is not None:
            parts.append(f"  Confidence: {rec['confidence']}")
        if rec.get("what_to_check") is not None:
            parts.append(f"  What to Check: {rec['what_to_check']}")
        if rec.get("what_to_do_now") is not None:
            parts.append(f"  What to Do Now: {rec['what_to_do_now']}")
        if rec.get("corrective_action") is not None:
            parts.append(f"  Corrective Action: {rec['corrective_action']}")
        if rec.get("urgency") is not None:
            parts.append(f"  Urgency: {rec['urgency']}")
        if rec.get("maintenance_required") is not None:
            mr = rec['maintenance_required']
            parts.append(f"  Maintenance Required: {mr}")
        if rec.get("maintenance_timing") is not None:
            parts.append(f"  Maintenance Timing: {rec['maintenance_timing']}")
        if rec.get("energy_impact_kwh") is not None:
            parts.append(f"  Energy Impact: {rec['energy_impact_kwh']} kWh")
        if rec.get("cost_impact") is not None:
            parts.append(f"  Cost Impact: {rec['cost_impact']}")
        if rec.get("source_stages") is not None:
            parts.append(f"  Source Stages: {', '.join(rec['source_stages'])}")
    return "\n".join(parts) if parts else "No verified diagnostic recommendation is currently available."


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
    "get_historical_analysis": _render_historical,
    "get_monthly_reports": _render_monthly_reports,
    "get_diagnostic_recommendations": _render_diagnostic_recommendations,
}

_ORDER = ["get_system_summary", "get_meters", "get_meter", "get_faults",
          "get_anomalies", "get_peaks", "get_maintenance", "get_forecast",
          "get_bill_prediction", "get_energy_saving",
          "get_historical_analysis", "get_monthly_reports",
          "get_diagnostic_recommendations"]


def _has_data(v: Any) -> bool:
    if isinstance(v, list):
        return len(v) > 0
    if isinstance(v, dict):
        return len(v) > 0
    return v is not None


def _compose_combined_evidence(question: str, results: dict) -> dict[str, list]:
    """Build structured evidence from verified tool results.

    Returns:
        {
            "measured": [...],
            "observed": [...],
            "probable": [...],
            "action": [...],
            "maintenance": [...],
            "impact": [...],
        }
    All values are strings or dicts pulled only from tool output.
    Nothing is invented.
    """
    evidence: dict[str, list] = {
        "measured": [],
        "observed": [],
        "probable": [],
        "action": [],
        "maintenance": [],
        "impact": [],
    }

    # ---- MEASURED: meter + historical data ----
    meter_data = results.get("get_meter")
    if isinstance(meter_data, dict) and meter_data.get("pzem_number") is not None:
        pz = meter_data["pzem_number"]
        parts = [f"PZEM {pz} live reading:"]
        if meter_data.get("voltage") is not None:
            parts.append(f"Voltage: {meter_data['voltage']} V")
        if meter_data.get("current") is not None:
            parts.append(f"Current: {meter_data['current']} A")
        if meter_data.get("power") is not None:
            parts.append(f"Power: {meter_data['power']} W")
        if meter_data.get("energy") is not None:
            parts.append(f"Energy: {meter_data['energy']} kWh")
        if meter_data.get("frequency") is not None:
            parts.append(f"Frequency: {meter_data['frequency']} Hz")
        if meter_data.get("power_factor") is not None:
            parts.append(f"PF: {meter_data['power_factor']}")
        if meter_data.get("online") is False:
            parts.append("(offline)")
        if len(parts) > 1:
            evidence["measured"].append(" ".join(parts))

    meters_data = results.get("get_meters")
    if isinstance(meters_data, list) and meters_data:
        online = [m for m in meters_data if m.get("online") is True]
        parts = [f"{len(online)} of {len(meters_data)} meters online."]
        for m in sorted(online, key=lambda m: m["pzem_number"])[:5]:
            p = m.get("power")
            parts.append(f"PZEM {m['pzem_number']}: {p} W" if p is not None else f"PZEM {m['pzem_number']}: online")
        evidence["measured"].append(" ".join(parts))

    hist_data = results.get("get_historical_analysis")
    if isinstance(hist_data, dict):
        status = hist_data.get("status")
        if status == "OK":
            pz = hist_data.get("pzem_number")
            parts = []
            if pz:
                parts.append(f"PZEM {pz} historical analysis ({_fmt_ts_s(hist_data.get('requested_start', 0))} to {_fmt_ts_s(hist_data.get('requested_end', 0))}):")
            else:
                parts.append("System-wide historical analysis:")
            power = hist_data.get("power")
            if power:
                parts.append(f"Average power: {power.get('average', 'N/A')} W")
                if power.get("maximum") is not None:
                    parts.append(f"Maximum power: {power['maximum']} W")
                if power.get("minimum") is not None:
                    parts.append(f"Minimum power: {power['minimum']} W")
            voltage = hist_data.get("voltage")
            if voltage and voltage.get("average") is not None:
                parts.append(f"Average voltage: {voltage['average']} V")
            current = hist_data.get("current")
            if current and current.get("average") is not None:
                parts.append(f"Average current: {current['average']} A")
            energy = hist_data.get("energy_consumption")
            if energy and energy.get("consumption_kwh") is not None:
                parts.append(f"Energy consumed: {energy['consumption_kwh']:.4f} kWh")
            if hist_data.get("available_days"):
                parts.append(f"Data span: {hist_data['available_days']:.1f} days")
            if parts:
                evidence["measured"].append(" ".join(parts))
        elif status == "NO_DATA":
            evidence["measured"].append("No historical data available for the requested period.")
        elif status == "INSUFFICIENT_DATA":
            evidence["measured"].append(f"Insufficient historical data: {hist_data.get('reason', '')}".strip())

    sys_summary = results.get("get_system_summary")
    if isinstance(sys_summary, dict) and sys_summary.get("system_status") is not None:
        parts = [f"System status: {sys_summary['system_status']}."]
        if sys_summary.get("total_power_w") is not None:
            parts.append(f"Total power: {sys_summary['total_power_w']} W")
        if sys_summary.get("total_energy_kwh") is not None:
            parts.append(f"Total energy: {sys_summary['total_energy_kwh']} kWh")
        if sys_summary.get("average_voltage_v") is not None:
            parts.append(f"Average voltage: {sys_summary['average_voltage_v']} V")
        if sys_summary.get("active_fault_count") is not None:
            parts.append(f"Active faults: {sys_summary['active_fault_count']}")
        if parts:
            evidence["measured"].append(" ".join(parts))

    # ---- OBSERVED: faults, anomalies, peaks, trends ----
    faults = results.get("get_faults")
    if isinstance(faults, list) and faults:
        for f in faults[:5]:
            n = f.get("pzem_number")
            ft = f.get("fault_type") or "fault"
            ts = _fmt_ts(f.get("timestamp"))
            evidence["observed"].append(f"Fault: PZEM {n}: {ft} at {ts}" if n is not None else f"Fault: {ft} at {ts}")

    anomalies = results.get("get_anomalies")
    if isinstance(anomalies, list) and anomalies:
        for a in anomalies[:5]:
            n = a.get("pzem_number")
            label = a.get("anomaly_label") or "anomaly"
            ts = _fmt_ts(a.get("timestamp"))
            evidence["observed"].append(f"Anomaly: PZEM {n}: {label} at {ts}" if n is not None else f"Anomaly: {label} at {ts}")

    peaks = results.get("get_peaks")
    if isinstance(peaks, list) and peaks:
        for p in peaks[:3]:
            tp = p.get("total_peak_power_w")
            ts = _fmt_ts(p.get("timestamp"))
            dom = p.get("dominant_pzems")
            dom_s = f" (dominant: PZEM {dom})" if dom else ""
            evidence["observed"].append(f"Peak: {tp} W at {ts}{dom_s}" if tp is not None else f"Peak at {ts}")

    maint_data = results.get("get_maintenance")
    if isinstance(maint_data, list) and maint_data:
        sys_rec = next((r for r in maint_data if r.get("pzem_number") is None), None)
        if sys_rec:
            hr = len(sys_rec.get("high_risk_meters", []) or [])
            wm = len(sys_rec.get("watch_meters", []) or [])
            hr_pz = sys_rec.get("highest_risk_pzem")
            evidence["observed"].append(f"Maintenance: {hr} high-risk and {wm} watch meters."
                                        + (f" Highest-risk meter is PZEM {hr_pz}." if hr_pz is not None else ""))
        else:
            for r in maint_data[:5]:
                n = r.get("pzem_number")
                lvl = r.get("risk_level")
                evidence["observed"].append(f"Maintenance risk: PZEM {n}: {lvl}" if n is not None and lvl is not None else f"Maintenance risk: {lvl}")

    # ---- PROBABLE: from Stage 4 diagnostic recommendations ----
    diag = results.get("get_diagnostic_recommendations")
    if isinstance(diag, list) and diag:
        for rec in diag[:5]:
            pz = rec.get("pzem_number")
            label = f"PZEM {pz}" if pz is not None else "SYSTEM"
            pc = rec.get("probable_cause")
            if pc is not None:
                evidence["probable"].append(f"{label}: {pc}")
            conf = rec.get("confidence")
            if conf is not None:
                evidence["probable"].append(f"{label}: confidence = {conf}")

    # ---- ACTION: what_to_check, what_to_do_now, corrective_action ----
    if isinstance(diag, list) and diag:
        for rec in diag[:5]:
            pz = rec.get("pzem_number")
            label = f"PZEM {pz}" if pz is not None else "SYSTEM"
            wtc = rec.get("what_to_check")
            if wtc is not None:
                evidence["action"].append(f"{label} — What to Check: {wtc}")
            wtdn = rec.get("what_to_do_now")
            if wtdn is not None:
                evidence["action"].append(f"{label} — What to Do Now: {wtdn}")
            ca = rec.get("corrective_action")
            if ca is not None:
                evidence["action"].append(f"{label} — Corrective Action: {ca}")

    # ---- MAINTENANCE: urgency, maintenance_required, maintenance_timing ----
    if isinstance(diag, list) and diag:
        for rec in diag[:5]:
            pz = rec.get("pzem_number")
            label = f"PZEM {pz}" if pz is not None else "SYSTEM"
            urgency = rec.get("urgency")
            if urgency is not None:
                evidence["maintenance"].append(f"{label}: Urgency = {urgency}")
            mr = rec.get("maintenance_required")
            if mr is not None:
                evidence["maintenance"].append(f"{label}: Maintenance Required = {mr}")
            mt = rec.get("maintenance_timing")
            if mt is not None:
                evidence["maintenance"].append(f"{label}: Maintenance Timing = {mt}")

    # ---- IMPACT: energy_impact_kwh, cost_impact ----
    if isinstance(diag, list) and diag:
        for rec in diag[:5]:
            pz = rec.get("pzem_number")
            label = f"PZEM {pz}" if pz is not None else "SYSTEM"
            eik = rec.get("energy_impact_kwh")
            if eik is not None:
                evidence["impact"].append(f"{label}: Energy Impact = {eik} kWh")
            ci = rec.get("cost_impact")
            if ci is not None:
                evidence["impact"].append(f"{label}: Cost Impact = {ci}")

    # If measured is empty but historical+diagnostic (historical question),
    # add a label noting no live meter data is available
    if not evidence["measured"] and ("get_historical_analysis" in results) and not ("get_meter" in results):
        hist = results.get("get_historical_analysis")
        if isinstance(hist, dict) and hist.get("status") == "OK":
            evidence["measured"].append("Historical data only (no live meter reading available for this period).")

    return evidence


def _correlate_evidence(results: dict, evidence: dict[str, list]) -> dict:
    """Analyze relationships between evidence sections (Stage 5D Phase 3B).

    Produces a structured reasoning context that tells BOB what is measured,
    what was observed, which observations are related, which Stage 4
    conclusions are supported, what relationships are NOT supported,
    whether evidence is sufficient/partial/conflicting/insufficient,
    and what can safely be stated.

    Never invents causal conclusions.
    Preserves Phase 2 evidence as authoritative.
    """
    correlation: dict[str, Any] = {
        "status": "INSUFFICIENT",
        "scope": "LIVE",
        "supported_findings": [],
        "relationships": [],
        "causal_support": [],
        "uncertainties": [],
        "evidence_quality": [],
        "pzem_scope": [],
        "has_live_data": False,
        "has_historical_data": False,
        "has_diagnostic": False,
        "has_system_data": False,
        "conflicts": [],
        "time_alignment_ok": True,
    }

    has_meter = isinstance(results.get("get_meter"), dict) and results.get("get_meter", {}).get("pzem_number") is not None
    has_meters = isinstance(results.get("get_meters"), list) and len(results.get("get_meters", [])) > 0
    has_hist = isinstance(results.get("get_historical_analysis"), dict)
    has_diag = isinstance(results.get("get_diagnostic_recommendations"), list) and len(results.get("get_diagnostic_recommendations", [])) > 0
    has_sys = isinstance(results.get("get_system_summary"), dict) and results.get("get_system_summary", {}).get("system_status") is not None
    has_faults = isinstance(results.get("get_faults"), list) and len(results.get("get_faults", [])) > 0
    has_anomalies = isinstance(results.get("get_anomalies"), list) and len(results.get("get_anomalies", [])) > 0
    has_peaks = isinstance(results.get("get_peaks"), list) and len(results.get("get_peaks", [])) > 0
    has_maintenance = isinstance(results.get("get_maintenance"), list) and len(results.get("get_maintenance", [])) > 0

    correlation["has_live_data"] = has_meter or has_meters
    correlation["has_historical_data"] = has_hist
    correlation["has_diagnostic"] = has_diag
    correlation["has_system_data"] = has_sys
    correlation["has_faults"] = has_faults
    correlation["has_anomalies"] = has_anomalies
    correlation["has_peaks"] = has_peaks
    correlation["has_maintenance"] = has_maintenance

    # ---- Determine scope ----
    if has_hist and not has_meter:
        correlation["scope"] = "HISTORICAL"
    elif has_sys and not has_meter and not has_hist:
        correlation["scope"] = "SYSTEM"
    elif has_meter:
        correlation["scope"] = "LIVE"
    elif has_hist:
        correlation["scope"] = "HISTORICAL"
    elif has_diag:
        correlation["scope"] = "LIVE"
    else:
        correlation["scope"] = "UNKNOWN"

    # ---- Determine PZEM scope ----
    pzems = set()
    if has_meter:
        pzems.add(results["get_meter"].get("pzem_number"))
    if has_meters:
        for m in results["get_meters"]:
            pz = m.get("pzem_number")
            if pz is not None:
                pzems.add(pz)
    if has_hist:
        pz = results["get_historical_analysis"].get("pzem_number")
        if pz is not None:
            pzems.add(pz)
    if has_diag:
        for rec in results["get_diagnostic_recommendations"][:5]:
            pz = rec.get("pzem_number")
            if pz is not None:
                pzems.add(pz)
    if has_sys:
        pzems.add(None)  # system-level, no specific PZEM
    correlation["pzem_scope"] = sorted(pzems, key=lambda x: x if x is not None else -1)

    # ---- Conflicts ----
    # Conflict: measured high voltage but diagnostic says no voltage fault
    if has_meter and has_diag:
        meter = results["get_meter"]
        voltage = meter.get("voltage")
        if voltage is not None:
            for rec in results["get_diagnostic_recommendations"][:5]:
                ft = rec.get("fault_type")
                sev = rec.get("severity")
                if ft is not None and sev == "NORMAL" and voltage > 240:
                    correlation["conflicts"].append(
                        f"Measured voltage={voltage}V but diagnostic reports {ft} as NORMAL"
                    )
                if ft is not None and sev == "NORMAL" and ft in ("high_power", "overcurrent"):
                    correlation["conflicts"].append(
                        f"Measured {ft} but diagnostic severity is NORMAL"
                    )

    # ---- Supported findings ----
    if has_meter:
        meter = results["get_meter"]
        parts = [f"PZEM {meter.get('pzem_number')} measured"]
        for field, label in [("voltage", "voltage"), ("current", "current"),
                              ("power", "power"), ("energy", "energy"),
                              ("frequency", "frequency"), ("power_factor", "PF")]:
            val = meter.get(field)
            if val is not None:
                parts.append(f"{label}={val}")
        correlation["supported_findings"].append(" ".join(parts))

    if has_hist:
        hist = results["get_historical_analysis"]
        pz = hist.get("pzem_number")
        pz_label = f"PZEM {pz}" if pz else "System"
        correlation["supported_findings"].append(f"{pz_label} historical data available")
        if hist.get("status") == "OK":
            power = hist.get("power")
            if power and power.get("average") is not None:
                correlation["supported_findings"].append(f"{pz_label} historical average power={power['average']}W")

    if has_sys:
        correlation["supported_findings"].append("System summary available")

    # ---- Causal support from Stage 4 ----
    if has_diag:
        for rec in results["get_diagnostic_recommendations"][:5]:
            pz = rec.get("pzem_number")
            label = f"PZEM {pz}" if pz is not None else "SYSTEM"
            pc = rec.get("probable_cause")
            conf = rec.get("confidence")
            if pc is not None:
                correlation["causal_support"].append(f"{label}: probable_cause={pc}")
            if conf is not None:
                correlation["causal_support"].append(f"{label}: confidence={conf}")
            wtc = rec.get("what_to_check")
            if wtc is not None:
                correlation["causal_support"].append(f"{label}: what_to_check={wtc}")
            wtdn = rec.get("what_to_do_now")
            if wtdn is not None:
                correlation["causal_support"].append(f"{label}: what_to_do_now={wtdn}")

    # ---- Relationships ----
    if has_meter and has_diag:
        meter_pz = results["get_meter"].get("pzem_number")
        diag_pzems = {rec.get("pzem_number") for rec in results["get_diagnostic_recommendations"][:5]}
        if meter_pz in diag_pzems or None in diag_pzems:
            correlation["relationships"].append(f"Live meter PZEM-{meter_pz} has corresponding diagnostic evidence")
        elif diag_pzems:
            correlation["relationships"].append(f"Live meter PZEM-{meter_pz} has no matching diagnostic recommendation")

    if has_meter and has_peaks:
        correlation["relationships"].append("Live meter readings and peak data are both available for correlation")

    if has_anomalies and has_diag:
        correlation["relationships"].append("Anomaly observations may support Stage 4 diagnostic conclusions")

    # ---- Evidence quality ----
    evidence_count = sum(1 for v in evidence.values() if v)
    if evidence_count >= 4:
        correlation["evidence_quality"].append("HIGH: multiple evidence sections populated")
    elif evidence_count >= 2:
        correlation["evidence_quality"].append("MEDIUM: partial evidence sections populated")
    elif evidence_count >= 1:
        correlation["evidence_quality"].append("LOW: minimal evidence sections populated")
    else:
        correlation["evidence_quality"].append("NONE: no evidence sections populated")

    # ---- Uncertainties ----
    if has_meter and not has_diag and not has_hist:
        correlation["uncertainties"].append("Measured data exists but no diagnostic cause is available")
    if has_diag and not has_meter and not has_hist:
        correlation["uncertainties"].append("Diagnostic conclusions exist but no measurement data to confirm")
    if has_meter and has_diag:
        meter_pz = results["get_meter"].get("pzem_number")
        diag_pzems = {rec.get("pzem_number") for rec in results["get_diagnostic_recommendations"][:5]}
        if meter_pz not in diag_pzems and None not in diag_pzems:
            correlation["uncertainties"].append(f"Diagnostic evidence does not cover PZEM-{meter_pz}")
    if has_hist and not has_diag:
        correlation["uncertainties"].append("Historical data exists but no diagnostic cause analysis")
    if has_peaks and not has_diag:
        correlation["uncertainties"].append("Peak data exists but no diagnostic explanation")

    # ---- Status determination ----
    if has_diag and (has_meter or has_sys or has_hist):
        correlation["status"] = "SUPPORTED"
    elif has_diag and not has_meter and not has_sys and not has_hist:
        correlation["status"] = "PARTIAL"
    elif has_meter and not has_diag:
        correlation["status"] = "PARTIAL"
    elif has_sys and not has_diag and not has_meter:
        correlation["status"] = "PARTIAL"
    elif has_hist and not has_diag:
        correlation["status"] = "PARTIAL"
    elif has_faults and not has_diag and not has_meter and not has_sys:
        correlation["status"] = "PARTIAL"
    elif has_maintenance and not has_diag and not has_meter and not has_sys:
        correlation["status"] = "PARTIAL"
    elif evidence_count == 0:
        correlation["status"] = "INSUFFICIENT"
    elif evidence_count == 1:
        correlation["status"] = "INSUFFICIENT"
    else:
        correlation["status"] = "PARTIAL"

    # Check for conflicts that downgrade status
    if correlation["conflicts"]:
        correlation["status"] = "CONFLICTING"

    return correlation


def _format_evidence(evidence: dict[str, list]) -> str:
    """Format structured evidence into a readable deterministic string."""
    if not any(evidence.values()):
        return "I don't have enough verified data to answer that."
    sections = []
    if evidence["measured"]:
        sections.append("Measured\n- " + "\n- ".join(evidence["measured"]))
    if evidence["observed"]:
        sections.append("Observed\n- " + "\n- ".join(evidence["observed"]))
    if evidence["probable"]:
        sections.append("Probable Cause\n- " + "\n- ".join(evidence["probable"]))
    if evidence["action"]:
        sections.append("What to Check / Do Now\n- " + "\n- ".join(evidence["action"]))
    if evidence["maintenance"]:
        sections.append("Urgency / Maintenance\n- " + "\n- ".join(evidence["maintenance"]))
    if evidence["impact"]:
        sections.append("Impact\n- " + "\n- ".join(evidence["impact"]))
    return "\n\n".join(sections) if sections else "I don't have enough verified data to answer that."


def _reason_response(pieces: list[str], evidence: dict[str, list], correlation: Optional[dict],
                         question: str) -> str:
    """Phase 3C: Structure and enhance rendered evidence into an engineer-style response.

    Takes rendered pieces from _RENDERERS + structured evidence + correlation
    and produces a clear, concise, evidence-grounded response following:
    1. Directly measured facts first
    2. Relevant observed events/patterns
    3. Supported diagnosis from Stage 4
    4. What the evidence means
    5. What should be checked
    6. What should be done now
    7. Maintenance/urgency if supported
    8. Energy/cost impact if supplied
    9. Explicit uncertainty when evidence is insufficient

    Never invents measurements, causes, or actions beyond what evidence supplies.
    """
    q = question.strip().lower()
    is_why = bool(re.search(r"\bwhy\b|reason|reason kya|kyu|kyu hai|kyu tha|kyu hua|kaise", q))
    is_action = bool(re.search(r"\bwhat to do|what should|action lena|kya karna|kya karu|corrective|what_to_check|what_to_do_now", q))
    is_historical = correlation is not None and correlation.get("scope") == "HISTORICAL"
    is_system = correlation is not None and correlation.get("scope") == "SYSTEM"

    status = correlation.get("status", "INSUFFICIENT") if correlation else "INSUFFICIENT"
    uncertainties = correlation.get("uncertainties", []) if correlation else []
    conflicts = correlation.get("conflicts", []) if correlation else []
    scope = correlation.get("scope", "UNKNOWN") if correlation else "UNKNOWN"

    # Determine what evidence sections are present
    has_measured = bool(evidence.get("measured"))
    has_observed = bool(evidence.get("observed"))
    has_probable = bool(evidence.get("probable"))
    has_action = bool(evidence.get("action"))
    has_maintenance = bool(evidence.get("maintenance"))
    has_impact = bool(evidence.get("impact"))

    # Start with rendered pieces as the base content
    result_pieces = list(pieces)

    # ---- CONFLICTING evidence: add conflict notice ----
    if conflicts and pieces:
        result_pieces.append("Note: " + "; ".join(conflicts[:2]))

    # ---- UNCERTAINTY section: only for why/action questions when diagnosis unavailable ----
    if uncertainties and (is_why or is_action) and status in ("INSUFFICIENT", "PARTIAL"):
        result_pieces.append("Available data does not establish a specific cause.")

    # ---- ACTION section for WHY/ACTION questions ----
    if (is_why or is_action) and has_action and not any("What to Check" in p for p in pieces):
        action_parts = []
        for item in evidence["action"][:5]:
            action_parts.append(item)
        if action_parts:
            result_pieces.extend(action_parts)

    # ---- If pieces is empty, try to build from evidence ----
    if not result_pieces:
        if has_measured:
            result_pieces.extend(evidence["measured"][:5])
        if has_observed:
            result_pieces.extend(evidence["observed"][:5])
        if has_probable:
            result_pieces.extend(evidence["probable"][:5])
        if not result_pieces:
            return "Available data does not establish a specific cause."

    # ---- System scope: remove PZEM-specific attribution from pieces if needed ----
    if is_system:
        filtered = []
        for p in result_pieces:
            if "PZEM-" in p and correlation and correlation.get("pzem_scope") and \
               None not in correlation.get("pzem_scope", []):
                continue
            filtered.append(p)
        result_pieces = filtered if filtered else result_pieces

    # ---- Historical: keep only historical references ----
    if is_historical:
        filtered = []
        for p in result_pieces:
            if "is online" in p or p.startswith("Voltage:") or p.startswith("Current:") or \
               p.startswith("Power:") or "measured" in p.lower():
                if "historical" not in p.lower() and "data span" not in p.lower() and \
                   "Average power" not in p.lower():
                    continue
            filtered.append(p)
        result_pieces = filtered if filtered else result_pieces

    # ---- Always pass through all pieces; conciseness handled by LLM prompt ----
    return "\n\n".join(result_pieces) if result_pieces else "Available data does not establish a specific cause."


# ---------------------------------------------------------------------------
# Phase 3D: Final Validation / Safety Gate
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(
    r"(?:voltage|current|power|energy|PF|frequency|kWh|kwh|\u20b9|₹|percent|%"
    r"|confidence|W\b|V\b|A\b|Hz\b)"
)

_DEBUG_TAG_RE = re.compile(
    r"\[Evidence status:|\[Note:|DEBUG:|correlation=|status=|raw tool output|internal metadata="
)

_DIAGNOSTIC_FIELDS = frozenset({
    "probable_cause", "confidence", "what_to_check", "what_to_do_now",
    "corrective_action", "urgency", "maintenance_required", "maintenance_timing",
    "energy_impact_kwh", "cost_impact", "fault_type", "severity", "priority",
})


def _validate_response(
    response: str,
    question: str,
    results: dict,
    evidence: dict[str, list],
    correlation: Optional[dict],
) -> dict[str, Any]:
    """Phase 3D: Deterministic validation of the final user-facing response.

    Validates the response against authoritative evidence and returns a
    structured report. Never mutates source data.

    Returns:
        {
            "status": "VALID" | "REPAIRABLE" | "UNSAFE",
            "violations": [...],
            "warnings": [...],
            "scope_ok": bool,
            "time_ok": bool,
            "diagnosis_ok": bool,
            "action_ok": bool,
            "numeric_values_ok": bool,
            "debug_tags_ok": bool,
        }
    """
    report: dict[str, Any] = {
        "status": "VALID",
        "violations": [],
        "warnings": [],
        "scope_ok": True,
        "time_ok": True,
        "diagnosis_ok": True,
        "action_ok": True,
        "numeric_values_ok": True,
        "debug_tags_ok": True,
    }

    resp_lower = response.lower()
    q_lower = question.strip().lower()
    is_why = bool(re.search(r"\bwhy\b|reason|reason kya|kyu|kyu hai|kyu tha|kyu hua|kaise", q_lower))
    is_action = bool(re.search(r"\bwhat to do|what should|action lena|kya karna|kya karu|corrective", q_lower))
    is_historical = correlation is not None and correlation.get("scope") == "HISTORICAL"
    is_system = correlation is not None and correlation.get("scope") == "SYSTEM"
    status = correlation.get("status", "INSUFFICIENT") if correlation else "INSUFFICIENT"

    # ---- RULE 15: Debug / internal metadata ----
    if _DEBUG_TAG_RE.search(response):
        report["status"] = "REPAIRABLE"
        report["violations"].append("debug_tag")
        report["debug_tags_ok"] = False
        report["warnings"].append("Internal metadata tags found in response")

    # ---- RULE 13: Insufficient evidence ----
    if status in ("INSUFFICIENT",) and not evidence.get("probable"):
        has_definitive_cause = bool(re.search(
            r"(caused by|is the cause|caused this|reason is|because of|due to|is causing|cause of|root cause|motor overload|transformer issue|bad cable)",
            resp_lower
        ))
        if has_definitive_cause:
            report["status"] = "UNSAFE"
            report["violations"].append("unsupported_cause")
            report["diagnosis_ok"] = False
            report["warnings"].append("Definitive cause stated with INSUFFICIENT evidence")

    # ---- RULE 13: PARTIAL without causal support ----
    if status == "PARTIAL" and not evidence.get("probable") and is_why:
        has_definitive_cause = bool(re.search(
            r"(caused by|is the cause|caused this|reason is|because of|due to|is causing|cause of)",
            resp_lower
        ))
        if has_definitive_cause:
            report["status"] = "UNSAFE"
            report["violations"].append("unsupported_cause_partial")
            report["diagnosis_ok"] = False

    # ---- RULE 5: Confidence protection ----
    if evidence.get("probable"):
        all_confidences = []
        for src in evidence["probable"]:
            cm = re.search(r"confidence\s*=\s*([\d.]+)", src)
            if cm:
                all_confidences.append(float(cm.group(1)))
        if all_confidences:
            avg_conf = sum(all_confidences) / len(all_confidences)
            if avg_conf < 0.8 and re.search(r"\bhigh confidence\b", resp_lower):
                report["status"] = "REPAIRABLE"
                report["violations"].append("confidence_mismatch")
                report["warnings"].append("High confidence claimed but evidence confidence is lower")
            if re.search(r"\bconfirmed\b|\bdefinite\b|\bcertain\b", resp_lower) and avg_conf < 0.9:
                report["status"] = "REPAIRABLE"
                report["violations"].append("confidence_mismatch")
                report["warnings"].append("Definitive wording used with moderate confidence evidence")

    # ---- RULE 4: Stage 4 diagnosis authority ----
    # Check that any stated diagnosis fields are in evidence
    if evidence.get("probable"):
        diag_text_source = " ".join(evidence["probable"]).lower()
    else:
        diag_text_source = ""

    # Check for diagnosed causes not in evidence
    cause_keywords = ["causing", "is the cause", "cause is", "root cause", "reason is"]
    for kw in cause_keywords:
        if kw in resp_lower:
            if not diag_text_source or not any(kw in diag_text_source.lower() for kw in ["overvoltage", "overload", "high_power", "excessive", "fault"]):
                report["status"] = "UNSAFE"
                report["violations"].append("unsupported_cause")
                report["diagnosis_ok"] = False
                report["warnings"].append("Definitive cause stated without evidence")
                break

    # Specific transformer/cable/equipment cause without evidence
    for bad_cause in ["transformer issue", "motor overload", "bad cable", "faulty wiring"]:
        if bad_cause in resp_lower and bad_cause not in diag_text_source:
            report["status"] = "UNSAFE"
            report["violations"].append("unsupported_cause")
            report["diagnosis_ok"] = False
            report["warnings"].append(f"Cause '{bad_cause}' not in authoritative evidence")
            break

    # ---- RULE 1: No unsupported numbers ----
    # Extract numeric values mentioned alongside energy keywords
    numeric_keywords = [
        r"voltage[^.]*?(\d+(?:\.\d+)?)\s*v\b",
        r"voltage[^.]*?(\d+(?:\.\d+)?)\s*kv\b",
        r"current[^.]*?(\d+(?:\.\d+)?)\s*a\b",
        r"power[^.]*?(\d+(?:\.\d+)?)\s*w\b",
        r"(?:energy|consumption)[^.]*?(\d+(?:\.\d+)?)\s*kwh\b",
        r"frequency[^.]*?(\d+(?:\.\d+)?)\s*hz\b",
    ]
    measured_values: dict[str, float] = {}
    meter_data = results.get("get_meter")
    if isinstance(meter_data, dict):
        for field in ("voltage", "current", "power", "energy", "frequency"):
            val = meter_data.get(field)
            if isinstance(val, (int, float)):
                measured_values[field] = float(val)

    hist_data = results.get("get_historical_analysis")
    if isinstance(hist_data, dict):
        power = hist_data.get("power")
        if isinstance(power, dict):
            for field, val in power.items():
                if isinstance(val, (int, float)):
                    measured_values[f"hist_power_{field}"] = float(val)
        voltage = hist_data.get("voltage")
        if isinstance(voltage, dict) and isinstance(voltage.get("average"), (int, float)):
            measured_values["hist_voltage_avg"] = float(voltage["average"])
        current = hist_data.get("current")
        if isinstance(current, dict) and current.get("average") is not None:
            measured_values["hist_current_avg"] = float(current["average"])
        energy = hist_data.get("energy_consumption")
        if isinstance(energy, dict) and energy.get("consumption_kwh") is not None:
            measured_values["hist_consumption_kwh"] = float(energy["consumption_kwh"])

    sys_data = results.get("get_system_summary")
    if isinstance(sys_data, dict):
        for field in ("total_power_w", "total_energy_kwh", "average_voltage_v"):
            val = sys_data.get(field)
            if isinstance(val, (int, float)):
                measured_values[f"sys_{field}"] = float(val)

    for pat in numeric_keywords:
        for m in re.finditer(pat, resp_lower):
            claimed_val = float(m.group(1))
            matched_field = pat.split("[^")[0]
            if matched_field == "power":
                matched_field = "power"
            if matched_field == "energy":
                matched_field = "energy"
            found = False
            for ev_key, ev_val in measured_values.items():
                if abs(ev_val - claimed_val) < 0.01:
                    found = True
                    break
            if not found and matched_field not in ("kwh",):
                for ev_val in measured_values.values():
                    if abs(ev_val - claimed_val) < 0.01:
                        found = True
                        break
            if not found:
                report["status"] = "UNSAFE"
                report["violations"].append(f"unsupported_number:{matched_field}")
                report["numeric_values_ok"] = False
                report["warnings"].append(f"Numeric value {claimed_val} {matched_field} not in evidence")
                break

    # ---- RULE 6: Historical safety ----
    if is_historical:
        # Check that "is online", "live", "current" don't appear in historical context
        if re.search(r"\bis online\b|live reading|current reading|live data", resp_lower):
            report["status"] = "REPAIRABLE"
            report["violations"].append("historical_live_mix")
            report["time_ok"] = False
            report["warnings"].append("Live data reference in historical response")

        # Check timestamps match the historical period
        if hist_data and isinstance(hist_data, dict):
            req_start = hist_data.get("requested_start")
            req_end = hist_data.get("requested_end")
            if req_start and req_end:
                import datetime as _dt
                from ai.ask_bob import _fmt_ts_s
                period_start_str = _fmt_ts_s(req_start)
                period_end_str = _fmt_ts_s(req_end)
                if period_start_str not in response and period_end_str not in response:
                    report["warnings"].append("Historical period not explicitly stated")

    # ---- RULE 7: System / PZEM scope ----
    if is_system:
        pzem_refs = re.findall(r"PZEM[-_\s]*\d+", response)
        if pzem_refs and correlation and correlation.get("pzem_scope") and \
           None in correlation.get("pzem_scope", []):
            report["status"] = "REPAIRABLE"
            report["violations"].append("system_pzem_attribution")
            report["scope_ok"] = False
            report["warnings"].append("PZEM attribution in system-level response")

    # ---- RULE 10: Energy/cost impact ----
    impact_patterns = [
        (r"save[^.]*?(\d+(?:\.\d+)?)\s*(?:%|percent)", "percentage_savings"),
        (r"(\d+(?:\.\d+)?)\s*(?:%|percent)[^.]*?(?:saving|reduce|save)", "percentage_savings"),
        (r"\u20b9\s*\d+", "currency_savings"),
        (r"rs\s*\d+", "currency_savings"),
        (r"saved\s*\u20b9", "currency_savings"),
    ]
    for pat, vtype in impact_patterns:
        if re.search(pat, resp_lower):
            if not evidence.get("impact"):
                report["status"] = "UNSAFE"
                report["violations"].append(f"unsupported_{vtype}")
                report["warnings"].append(f"{vtype} claimed without evidence")
                break

    # ---- RULE 11: Maintenance timing ----
    maint_timing_patterns = [
        r"maintenance required\s+(tomorrow|next week|next month|in \d+ (days|hours|weeks))",
        r"inspection\s+(tomorrow|next week|in \d+ (days|hours|weeks))",
        r"schedule\s+(for|on)\s+(tomorrow|next week|in \d+ (days|hours|weeks))",
    ]
    for pat in maint_timing_patterns:
        if re.search(pat, resp_lower):
            if not evidence.get("maintenance"):
                report["status"] = "UNSAFE"
                report["violations"].append("unsupported_maintenance_timing")
                report["warnings"].append("Maintenance timing claimed without evidence")
                break
            mt_items = evidence["maintenance"]
            has_explicit_timing = any("tomorrow" in m.lower() or "next" in m.lower()
                                      for m in mt_items)
            if not has_explicit_timing:
                report["status"] = "REPAIRABLE"
                report["violations"].append("unsupported_maintenance_timing")
                report["warnings"].append("Maintenance timing not explicitly in evidence")
                break

    # ---- RULE 12: Conflict validation ----
    if correlation and correlation.get("conflicts"):
        if "definitely" in resp_lower or "certainly" in resp_lower or "confirmed" in resp_lower:
            if "NORMAL" in resp_lower and any("high" in c.lower() or "voltage" in c.lower() for c in correlation.get("conflicts", [])):
                report["status"] = "REPAIRABLE"
                report["violations"].append("conflict_suppressed")
                report["warnings"].append("Conflicting evidence presented as definitive")

    # ---- RULE 14: Simple question validation ----
    is_simple = not is_why and not is_action and not re.search(r"why|what.*fault|what.*cause|what.*problem|diagnostic|maintenance|recommend", q_lower)
    if is_simple:
        if len(resp_lower) > 300 and not any(k in q_lower for k in ["compare", "most", "highest", "rank", "all"]):
            report["status"] = "REPAIRABLE"
            report["violations"].append("simple_question_too_long")
            report["warnings"].append("Simple question produced overly lengthy response")
        # Simple questions should not have uncertainty
        if "does not establish a specific cause" in resp_lower:
            report["status"] = "REPAIRABLE"
            report["violations"].append("simple_question_uncertainty")
            report["warnings"].append("Simple question has unnecessary uncertainty")

    # ---- RULE 9: Action validation ----
    if is_action or is_why:
        action_patterns = [
            r"replace[^.]*?(transformer|cable|meter|breaker|panel)",
            r"install[^.]*?(device|sensor|panel)",
            r"check[^.]*?voltage",
            r"change[^.]*?(wiring|breaker|fuse)",
        ]
        for pat in action_patterns:
            m = re.search(pat, resp_lower)
            if m:
                claimed_action = m.group(0).strip().lower()
                if evidence.get("action"):
                    action_texts = " ".join(evidence["action"]).lower()
                    if claimed_action not in action_texts:
                        report["status"] = "UNSAFE"
                        report["violations"].append("unsupported_action")
                        report["action_ok"] = False
                        report["warnings"].append(f"Action '{claimed_action}' not in evidence")
                        break
                else:
                    report["status"] = "UNSAFE"
                    report["violations"].append("unsupported_action")
                    report["action_ok"] = False
                    report["warnings"].append(f"Action '{claimed_action}' not in evidence")
                    break

    # ---- Determine final status ----
    if any(v.startswith("unsupported_") for v in report["violations"]):
        report["status"] = "UNSAFE"
    elif report["violations"]:
        report["status"] = "REPAIRABLE"

    return report


def _safe_fallback(
    evidence: dict[str, list],
    correlation: Optional[dict],
    question: str,
) -> str:
    """Generate a deterministic safe fallback from authoritative evidence only.

    Never calls LLM. Never invents content. Uses only values in evidence.
    """
    status = correlation.get("status", "INSUFFICIENT") if correlation else "INSUFFICIENT"
    q = question.strip().lower()

    parts = []

    # Add measured evidence
    if evidence.get("measured"):
        parts.extend(evidence["measured"][:3])

    # Add observed evidence
    if evidence.get("observed"):
        parts.extend(evidence["observed"][:2])

    # Add probable cause if available
    if evidence.get("probable"):
        parts.extend(evidence["probable"][:2])

    # Add action if available
    if evidence.get("action"):
        parts.extend(evidence["action"][:2])

    # Add maintenance if available
    if evidence.get("maintenance"):
        parts.extend(evidence["maintenance"][:2])

    # Add impact if available
    if evidence.get("impact"):
        parts.extend(evidence["impact"][:2])

    if not parts:
        if correlation and correlation.get("scope") == "HISTORICAL":
            return "Historical data is available but does not establish a specific cause."
        return "Available data does not establish a specific cause."

    result = "\n\n".join(parts)

    # Add uncertainty for why/action questions
    is_why = bool(re.search(r"\bwhy\b|reason|reason kya|kyu|kyu hai|kyu tha|kyu hua|kaise", q))
    is_action = bool(re.search(r"\bwhat to do|what should|action lena|kya karna|kya karu|corrective", q))
    if (is_why or is_action) and status in ("INSUFFICIENT", "PARTIAL"):
        result += "\n\nAvailable data does not establish a specific cause."

    return result


def _apply_phase3d(
    response: str,
    question: str,
    results: dict,
    evidence: dict[str, list],
    correlation: Optional[dict],
) -> str:
    """Apply Phase 3D validation gate. Returns validated or safely-fallback response.

    Deterministic: no LLM calls. Never mutates source data.
    """
    if not response or not response.strip():
        return "I don't have enough current data to answer that."

    report = _validate_response(response, question, results, evidence, correlation)

    if report["status"] == "VALID":
        return response

    if report["status"] == "REPAIRABLE":
        # Try to repair by removing debug tags and excessive content
        repaired = response
        if not report["debug_tags_ok"]:
            repaired = _DEBUG_TAG_RE.sub("", repaired)
            repaired = re.sub(r"\n{3,}", "\n\n", repaired)
            repaired = repaired.strip()
        if not repaired.strip():
            repaired = _safe_fallback(evidence, correlation, question)
        return repaired

    # UNSAFE: replace with safe fallback
    return _safe_fallback(evidence, correlation, question)


def _compose_energy(question: str, results: dict, correlation: Optional[dict] = None) -> str:
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
    evidence = _compose_combined_evidence(question, results)
    response = _reason_response(pieces, evidence, correlation, question)
    return _apply_phase3d(response, question, results, evidence, correlation)


def _llm_compose_energy(question: str, results: dict, history: list, api_key: str, provider: str = "openrouter", correlation: Optional[dict] = None) -> Optional[str]:
    """LLM composes natural answer from structured combined evidence only."""
    try:
        import openai as _openai
    except ImportError:
        return None
    try:
        evidence = _compose_combined_evidence(question, results)
        if correlation is None:
            correlation = _correlate_evidence(results, evidence)
        system = (
            "You are BOB, the conversational AI assistant for the Smart Monitoring System, "
            "an energy-monitoring project. Answer naturally and professionally. "
            "Use ONLY the provided verified structured evidence below. Never invent meter readings, "
            "faults, predictions, timestamps, savings, or project facts. When quoting data, "
            "include the PZEM number, value and timestamp as evidence. If a section is empty, "
            "do not mention it. Keep responses concise (under 180 words). "
            "Avoid saying 'according to my database'. "
            "If evidence contains a 'status' field set to 'NO_DATA', say historical data is unavailable. "
            "If 'status' is 'INSUFFICIENT_DATA', say there is insufficient historical data. "
            "Do not substitute live readings for historical readings. "
            "CLEARLY LABEL historical data vs live data when both are present. "
            "STRUCTURED EVIDENCE RULES: "
            "The evidence is organized into: Measured, Observed, Probable Cause, "
            "What to Check / Do Now, Urgency / Maintenance, Impact. "
            "Stage 4 diagnostic fields are authoritative for probable cause, corrective action, "
            "urgency, maintenance recommendation, confidence, and energy/cost impact. "
            "NEVER invent any of these fields. If a recommendation has None/null values, "
            "do NOT fill them in. If the diagnostic list is empty, state no verified diagnostic "
            "recommendation is currently available. "
            "Preserve confidence values exactly (do not upgrade 0.5 to 'high confidence'). "
            "Do not convert observations into diagnoses. Do not infer unsupported fault causes. "
            "Do not invent appliance identities. Do not invent savings. "
            "EVIDENCE CORRELATION RULES: "
            "The correlation context provides analysis of what evidence supports what. "
            "Follow these correlation rules strictly: "
            "1. MEASUREMENT IS FACT - only state measured values, never infer causes from them alone. "
            "2. OBSERVATION IS NOT CAUSE - do not claim a specific cause unless Stage 4 diagnostic provides probable_cause. "
            "3. STAGE 4 IS AUTHORITATIVE - use only supplied probable_cause, confidence, what_to_check, what_to_do_now, corrective_action, urgency. "
            "4. NULL MEANS UNKNOWN - do not convert None to normal/0/unknown. "
            "5. HISTORICAL SAFETY - never mix historical and live data as if same time. "
            "6. SYSTEM SCOPE - do not attribute system-level evidence to individual PZEMs without explicit evidence. "
            "7. CONFLICTING EVIDENCE - if measurements conflict with diagnostics, report both separately without forcing one conclusion. "
            "8. INSUFFICIENT EVIDENCE - if cause is not supported, explicitly state the evidence does not establish a specific cause. "
            f"Correlation status: {correlation.get('status', 'INSUFFICIENT')}, scope: {correlation.get('scope', 'UNKNOWN')}. "
            f"Supported findings: {json.dumps(correlation.get('supported_findings', []), default=str)}. "
            f"Uncertainties: {json.dumps(correlation.get('uncertainties', []), default=str)}. "
            f"Conflicts: {json.dumps(correlation.get('conflicts', []), default=str)}. "
            f"PZEM scope: {correlation.get('pzem_scope', [])}. "
            "PHASE 3C RESPONSE RULES: "
            "You are ONLY a language/response layer, NOT an electrical calculation engine. "
            "STRUCTURE your response according to the question type: "
            "For simple questions (what is power/current/voltage), give a direct answer with value + unit. "
            "For diagnostic WHY questions, use this structure: Measured facts first, then Observed events, "
            "then Probable cause from Stage 4, then What to check, then What to do now, then Maintenance, then Impact. "
            "Omit sections that have no evidence. Do NOT invent missing sections. "
            "KEEP responses concise - match reasoning depth to question complexity. "
            "RULE: Start with actual measured evidence, never with a conclusion. "
            "RULE: Stage 4 diagnostic fields are authoritative - never upgrade confidence or reword causes. "
            "RULE: Clearly separate Measured / Observed / Probable Cause / Action / Maintenance / Impact. "
            "RULE: Action must come from Stage 4 what_to_check, what_to_do_now, corrective_action only. "
            "RULE: If no diagnosis exists, explicitly state 'Available data does not establish a specific cause.' "
            "RULE: Never say overload, motor fault, bad cable, transformer issue unless Stage 4 supplies it. "
            "RULE: If correlation status is CONFLICTING, state measured condition AND conflicting observation, "
            "do not select one side silently. Do not force a final diagnosis. "
            "RULE: Historical questions must only use historical evidence - never insert current/live values. "
            "RULE: System questions must keep system-level scope - never say PZEM-X caused system peak without explicit evidence. "
            "RULE: PZEM questions must keep scope limited to that specific PZEM - do not generalize to whole system. "
            "RULE: Only mention energy/cost impact when supplied - never calculate or invent savings. "
            "RULE: Preserve None/0/0.0 correctly - do not convert None to normal or 0. "
            "RULE: Never mix historical and live data as if same time period. "
            "RULE: Never create system-to-PZEM attribution without explicit evidence. "
            "PHASE 3C DETECTED QUESTION TYPE: "
        )
        messages = []
        for turn in (history or [])[-4:]:
            role = "assistant" if turn.get("role") == "bot" else "user"
            content = turn.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
        ctx_text = json.dumps(evidence, default=str)
        messages.append({"role": "user",
                         "content": f"Question: {question}\n\nVerified structured evidence:\n{ctx_text}"})
        if provider == "openrouter":
            client = _openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            model = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
            sys_msg = {"role": "system", "content": system}
            user_msgs = [msg for msg in messages if msg.get("role") == "user"]
            combined = [sys_msg] + user_msgs if user_msgs else [sys_msg] + messages
            resp = client.chat.completions.create(model=model,max_tokens=500, messages=combined)  # type: ignore
            answer = resp.choices[0].message.content.strip()
        else:
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=api_key)
            model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
            resp = client.messages.create(model=model, max_tokens=500, system=system, messages=messages)
            answer = resp.content[0].text.strip()
        validated = _apply_phase3d(answer, question, results, evidence, correlation)
        return validated or None
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
        evidence = _compose_combined_evidence(resolved, results)
        correlation = _correlate_evidence(results, evidence)

        if has_llm and results:
            energy_part = _llm_compose_energy(resolved, results, history, api_key,
                                              provider=getattr(get_settings(), "llm_provider", "openrouter"),
                                              correlation=correlation) or _compose_energy(resolved, results, correlation)
        else:
            energy_part = _compose_energy(resolved, results, correlation)

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
            ans = _llm_general_conversation(resolved, history, api_key,
                                            provider=getattr(get_settings(), "llm_provider", "openrouter"))
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
        evidence = _compose_combined_evidence(resolved, results)
        correlation = _correlate_evidence(results, evidence)

        if has_llm and results:
            ans = _llm_compose_energy(resolved, results, history, api_key,
                                      provider=getattr(get_settings(), "llm_provider", "openrouter"),
                                      correlation=correlation)
            if ans:
                return {"status": "ok", "answer": ans, "source": "llm", "intent": "energy"}

        return {"status": "ok", "answer": _compose_energy(resolved, results, correlation),
                "source": "tool", "intent": "energy"}

    # ---------------------------------------------------------
    # PROJECT KNOWLEDGE only
    # ---------------------------------------------------------
    if is_project:
        answer = _project_response(resolved, _KNOWLEDGE)
        if answer == _NO_INFO:
            # Not in knowledge base - try LLM for general knowledge if available
            if has_llm:
                ans = _llm_general_conversation(resolved, history, api_key,
                                                provider=getattr(get_settings(), "llm_provider", "openrouter"))
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