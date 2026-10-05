"""Deterministic probe of the BOB recent-fault path (no LLM, no Firebase writes).

Run: python tools/probe_bob_recent_faults.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ai-backend"))
os.environ.setdefault("LLM_API_KEY", "")

from ai import api_store, ask_bob, bob_tools  # noqa: E402
from ai import mapping  # noqa: E402

# 2026-09-10 11:33:20 UTC in Unix SECONDS (project convention / Firebase key).
TS_2026 = 1789040000
assert time.strftime("%Y", time.gmtime(TS_2026)) == "2026", "fixture must be a 2026 second"

FAULTS = [
    {"pzem_number": 1, "fault_type": "power_factor_drop", "severity": "WARNING",
     "measured_value": 0.62, "timestamp": TS_2026},
    {"pzem_number": 2, "fault_type": "overvoltage", "severity": "EMERGENCY",
     "measured_value": 265.0, "timestamp": TS_2026},
]
DIAG = [
    {"pzem_number": 1, "fault_type": "power_factor_drop", "severity": "WARNING",
     "timestamp": TS_2026, "probable_cause": "Load drawing lagging power factor",
     "confidence": 0.72, "what_to_check": "Check capacitor bank terminals",
     "what_to_do_now": "Log PF at the meter for 30 minutes",
     "urgency": "MEDIUM", "maintenance_required": False},
]
MAINT = [{"pzem_number": 1, "risk_level": "WATCH", "timestamp": TS_2026}]

api_store.read_faults = lambda: list(FAULTS)
api_store.read_diagnostic_recommendations = lambda: list(DIAG)
api_store.read_maintenance = lambda: list(MAINT)
api_store.read_anomalies = lambda: []
api_store.read_peaks = lambda: []
api_store.read_forecast = lambda: []
api_store.read_energy_saving = lambda: []
api_store.read_bill_prediction = lambda: []
api_store.read_all_meters = lambda: {}
api_store.meter_online = lambda m: (False, None)
api_store.record_timestamp = api_store.record_timestamp

mapping._REMOTE = {"pzem_1": {"load_name": "Fan 1", "location": "Classroom"}}
mapping._REMOTE_LOADED = True

print("=== _pzem_from_text on load-name queries ===")
for q in ("Fan 1 ka current kitna hai?", "PZEM-1 ka current?", "Fan 1 ka fault?"):
    print(f"  {q!r} -> {ask_bob._pzem_from_text(q)}")

print("\n=== _select_tools('Any recent faults?') ===")
for name, params in ask_bob._select_tools("Any recent faults?", []):
    print(f"  {name} {params}")

print("\n=== timestamp helpers ===")
print("  _fmt_ts(TS_2026)     =", ask_bob._fmt_ts(TS_2026))
print("  _fmt_ts_s(TS_2026)   =", ask_bob._fmt_ts_s(TS_2026))
print("  _fmt_ts(ms)          =", ask_bob._fmt_ts(TS_2026 * 1000))

print("\n=== _compose_energy (deterministic, no LLM) ===")
results = ask_bob._ok_results(bob_tools.ToolContext().__class__() and {}) if False else None
ctx = bob_tools.ToolContext()
results = ask_bob._ok_results({name: ctx.call(name, **params)
                              for name, params in ask_bob._select_tools("Any recent faults?", [])})
evidence = ask_bob._compose_combined_evidence("Any recent faults?", results)
corr = ask_bob._correlate_evidence(results, evidence)
print(ask_bob._compose_energy("Any recent faults?", results, corr))

print("\n=== FULL ask_bob('Any recent faults?') ===")
out = ask_bob.ask_bob("Any recent faults?", [])
print("source:", out.get("source"), "| intent:", out.get("intent"))
print(out["answer"])