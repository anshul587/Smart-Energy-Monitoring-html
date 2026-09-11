#!/usr/bin/env python3
"""Run real BOB test with OpenRouter FREE provider."""
import os
import sys
import json

# Load .env manually
env_path = "ai backend/.env"
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

# Verify env vars
print("OPENROUTER_API_KEY:", repr(os.environ.get("OPENROUTER_API_KEY", "NOT SET")))
print("LLM_PROVIDER:", repr(os.environ.get("LLM_PROVIDER", "NOT SET")))
print("OPENROUTER_MODEL:", repr(os.environ.get("OPENROUTER_MODEL", "NOT SET")))

sys.path.insert(0, "ai backend")

from ai.config import get_settings
from ai.ask_bob import ask_bob

settings = get_settings()
print("\nSettings:")
print("  llm_provider:", settings.llm_provider)
print("  open_router_api_key:", repr(settings.open_router_api_key))
print("  open_router_model:", settings.open_router_model)
print("  data_source:", settings.data_source)

# Run real BOB tests
questions = [
    "What is this project about?",                              # Project question
    "Hello BOB, how are you today?",                            # General question
    "What is the current power reading?",                       # Live meter/data question
    "Will there be any AI results today?",                      # AI-result question
    "What happens if data is unavailable?",                     # Unavailable-data question
]

print("\n=== Running REAL BOB provider smoke tests ===\n")

for i, question in enumerate(questions, 1):
    print(f"--- Test {i}: {question} ---")
    try:
        result = ask_bob(question)
        print(f"  status: {result.get('status')}")
        print(f"  source: {result.get('source')}")
        print(f"  intent: {result.get('intent')}")
        answer = result.get('answer', '')
        # Show first 200 chars of answer
        if answer:
            print(f"  answer: {answer[:200]}...")
        else:
            print(f"  answer: (none)")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()

print("=== BOB real test complete ===")