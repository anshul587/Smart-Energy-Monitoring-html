import sys, json
question = sys.argv[1] if len(sys.argv) > 1 else "What is Python?"
sys.path.insert(0, r'E:\smart energy monitoring sys\ai backend')
from ai.ask_bob import ask_bob
result = ask_bob(question)
print(json.dumps(result, ensure_ascii=False))