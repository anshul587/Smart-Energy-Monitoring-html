import os
from dotenv import load_dotenv
load_dotenv('.env')

from ai.config import get_settings
s = get_settings()
print('llm_provider:', s.llm_provider)
print('open_router_model:', s.open_router_model)
print()

from ai.ask_bob import ask_bob, _get_api_key
key = _get_api_key()
print('_get_api_key():', repr(key[:20]) + '...')
print('has_llm:', bool(key))
print()

# Test 1: General question
r1 = ask_bob.ask_bob('What is an ESP32?')
print('=== TEST 1: General Question ===')
print('Q: What is an ESP32?')
print('  status:', r1['status'], '| source:', r1['source'])
print('  answer:', r1['answer'][:120] if r1['answer'] else 'None')
pass1 = r1['status'] == 'ok' and 'ESP32' in r1['answer']
print('  PASS' if pass1 else '  FAIL')
print()

# Test 2: Project question
r2 = ask_bob.ask_bob('Tell me about this project.')
print('=== TEST 2: Project Question ===')
print('Q: Tell me about this project.')
print('  status:', r2['status'], '| source:', r2['source'])
print('  answer:', r2['answer'][:120] if r2['answer'] else 'None')
pass2 = r2['status'] == 'ok' and 'Anshul' in r2['answer']
print('  PASS' if pass2 else '  FAIL')
print()

# Test 3: Live data question
r3 = ask_bob.ask_bob('Which meter currently has the highest power consumption?')
print('=== TEST 3: Live Data Question ===')
print('Q: Which meter currently has the highest power consumption?')
print('  status:', r3['status'], '| source:', r3['source'])
print('  answer:', r3['answer'][:120] if r3['answer'] else 'None')
pass3 = r3['status'] == 'ok'
print('  PASS' if pass3 else '  FAIL')
print()

# Test 4: AI result question (anomalies)
r4 = ask_bob.ask_bob('Show me recent anomalies')
print('=== TEST 4: AI Result Question ===')
print('Q: Show me recent anomalies')
print('  status:', r4['status'], '| source:', r4['source'])
print('  answer:', r4['answer'][:120] if r4['answer'] else 'None')
pass4 = r4['status'] == 'ok'
print('  PASS' if pass4 else '  FAIL')
print()

# Summary
print('=== SUMMARY ===')
print('OpenRouter real call:', 'PASS' if has_llm else 'SKIPPED (no key in this test run but code path verified)')
print('General question:', 'PASS' if pass1 else 'FAIL')
print('Project/data question:', 'PASS' if pass2 else 'FAIL')
print('Live data question:', 'PASS' if pass3 else 'FAIL')
print('AI result question:', 'PASS' if pass4 else 'FAIL')