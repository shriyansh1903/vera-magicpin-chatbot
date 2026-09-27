import time
import requests
import json
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

def run_performance_test():
    # 1. Warmup context
    requests.post(f"{BASE_URL}/v1/context", json={
        "scope": "merchant", "context_id": "m_perf_01", "version": 1,
        "payload": {"name": "Perf Salon", "category_slug": "salons"}
    })
    requests.post(f"{BASE_URL}/v1/context", json={
        "scope": "customer", "context_id": "c_perf_01", "version": 1,
        "payload": {"name": "Perf Customer", "consent": True}
    })

    tick_latencies = []
    for i in range(10):
        trig_id = f"trig_perf_{i}"
        requests.post(f"{BASE_URL}/v1/context", json={
            "scope": "trigger", "context_id": trig_id, "version": 1,
            "payload": {"type": "curious_ask_due", "merchant_id": "m_perf_01", "urgency": "medium"}
        })
        start = time.time()
        res = requests.post(f"{BASE_URL}/v1/tick", json={"now": "2026-09-27T16:00:00Z", "available_triggers": [trig_id]})
        lat_ms = (time.time() - start) * 1000
        tick_latencies.append(lat_ms)
        print(f"Tick {i+1}: {lat_ms:.2f}ms (Status {res.status_code})")

    reply_latencies = []
    messages = [
        "Hi, what is your offer?",
        "Yes I am interested",
        "How much does it cost?",
        "Tell me more details",
        "What are your opening hours?",
        "I want to book an appointment",
        "Sure let's do it",
        "Can you send details?",
        "Thanks",
        "Stop messaging me"
    ]
    for i, msg in enumerate(messages):
        start = time.time()
        res = requests.post(f"{BASE_URL}/v1/reply", json={
            "conversation_id": "conv_perf_01",
            "merchant_id": "m_perf_01",
            "customer_id": "c_perf_01",
            "from_role": "customer",
            "message": msg,
            "turn_number": i + 2
        })
        lat_ms = (time.time() - start) * 1000
        reply_latencies.append(lat_ms)
        print(f"Reply {i+1}: {lat_ms:.2f}ms (Status {res.status_code}, Action: {res.json().get('action')})")

    print("\n=== LATENCY DISTRIBUTION SUMMARY ===")
    print(f"Tick Latencies (10 runs): Min={min(tick_latencies):.2f}ms | Avg={sum(tick_latencies)/10:.2f}ms | Max={max(tick_latencies):.2f}ms | p95={np.percentile(tick_latencies, 95):.2f}ms")
    print(f"Reply Latencies (10 runs): Min={min(reply_latencies):.2f}ms | Avg={sum(reply_latencies)/10:.2f}ms | Max={max(reply_latencies):.2f}ms | p95={np.percentile(reply_latencies, 95):.2f}ms")

if __name__ == "__main__":
    run_performance_test()
