import time
import requests
import json

BASE_URL = "http://127.0.0.1:8080"

def test_production_endpoints():
    print(f"Testing production endpoint server at {BASE_URL}...\n")

    # 1. Healthz
    r1 = requests.get(f"{BASE_URL}/v1/healthz")
    assert r1.status_code == 200, f"healthz failed: {r1.status_code}"
    print("GET /v1/healthz -> 200 OK")
    print("Response:", json.dumps(r1.json(), indent=2))
    print()

    # 2. Metadata
    r2 = requests.get(f"{BASE_URL}/v1/metadata")
    assert r2.status_code == 200, f"metadata failed: {r2.status_code}"
    print("GET /v1/metadata -> 200 OK")
    print("Response:", json.dumps(r2.json(), indent=2))
    print()

    # 3. Context
    r3 = requests.post(f"{BASE_URL}/v1/context", json={
        "scope": "merchant",
        "context_id": "m_prod_test",
        "version": 1,
        "payload": {"name": "Prod Test Salon", "category_slug": "salons"}
    })
    assert r3.status_code == 200, f"context failed: {r3.status_code}"
    res3 = r3.json()
    assert res3.get("accepted") is True, f"accepted is not True: {res3}"
    print("POST /v1/context -> 200 OK (accepted=True)")
    print("Response:", json.dumps(res3, indent=2))
    print()

    # Push trigger context
    requests.post(f"{BASE_URL}/v1/context", json={
        "scope": "trigger",
        "context_id": "trig_prod_test",
        "version": 1,
        "payload": {"type": "curious_ask_due", "merchant_id": "m_prod_test"}
    })

    # 4. Tick
    start_tick = time.time()
    r4 = requests.post(f"{BASE_URL}/v1/tick", json={
        "now": "2026-09-27T16:45:00Z",
        "available_triggers": ["trig_prod_test"]
    })
    tick_lat = (time.time() - start_tick) * 1000
    assert r4.status_code == 200, f"tick failed: {r4.status_code}"
    res4 = r4.json()
    assert "actions" in res4, "missing actions in tick response"
    if res4["actions"]:
        assert res4["actions"][0]["body"], "tick action body is empty"
    print(f"POST /v1/tick -> 200 OK ({tick_lat:.2f}ms)")
    print("Response:", json.dumps(res4, indent=2))
    print()

    # 5. Reply
    start_reply = time.time()
    r5 = requests.post(f"{BASE_URL}/v1/reply", json={
        "conversation_id": "conv_prod_01",
        "merchant_id": "m_prod_test",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Yes I am interested",
        "turn_number": 2
    })
    reply_lat = (time.time() - start_reply) * 1000
    assert r5.status_code == 200, f"reply failed: {r5.status_code}"
    res5 = r5.json()
    assert res5.get("action") in ["send", "wait", "end"], "invalid reply action"
    if res5.get("action") == "send":
        assert res5.get("body"), "send action has empty body"
    print(f"POST /v1/reply -> 200 OK ({reply_lat:.2f}ms)")
    print("Response:", json.dumps(res5, indent=2))
    print()

    print("ALL 5 PRODUCTION ENDPOINTS VERIFIED SUCCESSFULLY ON PORT 8080!")

if __name__ == "__main__":
    test_production_endpoints()
