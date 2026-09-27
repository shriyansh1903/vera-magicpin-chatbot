import sys
import io
import os
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

load_dotenv('.env')

sys.path.insert(0, 'magicpin-ai-challenge')
import judge_simulator

judge_simulator.BOT_URL = 'http://127.0.0.1:8000'
judge_simulator.LLM_PROVIDER = 'gemini'
judge_simulator.LLM_MODEL = 'gemini-flash-latest'
judge_simulator.LLM_API_KEY = os.getenv('GEMINI_API_KEY', '')

scenarios = [
    'warmup',
    'auto_reply_hell',
    'intent_transition',
    'hostile',
    'phase2_short',
    'all',
    'full_evaluation'
]

llm = judge_simulator.create_provider()
print(f"LLM Provider: {llm.name()}")

results = {}
for scenario in scenarios:
    print(f"\n==================== RUNNING SCENARIO: {scenario} ====================")
    try:
        judge = judge_simulator.JudgeSimulator(llm)
        res = judge.run(scenario)
        results[scenario] = "PASS" if res else "FAIL"
    except Exception as e:
        print(f"Scenario {scenario} error: {e}")
        results[scenario] = f"ERROR ({e})"

print("\n==================== SCENARIO RUN SUMMARY ====================")
for s, r in results.items():
    print(f"{s}: {r}")
