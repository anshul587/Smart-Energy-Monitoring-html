import sys, json
sys.path.insert(0, r'E:\smart energy monitoring sys\ai backend')
from ai.config import get_settings
from ai.ask_bob import _get_api_key

settings = get_settings()
print("LLM_PROVIDER from settings:", getattr(settings, 'llm_provider', 'NOT SET'))
print("open_router_api_key from settings:", getattr(settings, 'open_router_api_key', 'NOT SET'))
print("ANTHROPIC_API_KEY from settings:", getattr(settings, 'anthropic_api_key', 'NOT SET'))
print()
api_key = _get_api_key()
print("_get_api_key() result:", repr(api_key[:30]) + "..." if len(api_key) > 30 else repr(api_key))
print("has_llm:", bool(api_key))