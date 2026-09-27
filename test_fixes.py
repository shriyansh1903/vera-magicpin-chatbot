from fastapi.testclient import TestClient
from app import app, context_store, conversation_store
from composer import LLMComposer

client = TestClient(app)

def test_fix1_context_response_accepted_true():
    resp = client.post("/v1/context", json={
        "scope": "category",
        "context_id": "cat_test_fix",
        "version": 1,
        "payload": {"name": "Test Category"}
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] is True
    assert data["status"] == "accepted"
    assert data["scope"] == "category"
    assert data["context_id"] == "cat_test_fix"
    assert data["version"] == 1

def test_fix2_positive_commitment_never_null_body():
    conv_id = "conv_fix2_test"
    context_store.save_context("merchant", "m_fix2", 1, {"name": "Fix2 Salon", "active_offers": ["10% OFF"]})
    conversation_store.init_conversation(conv_id, "m_fix2", initial_message="Shall we launch campaign?")

    commitment_phrases = [
        "I want to join",
        "Yes, do it",
        "Ok lets do it. Whats next?"
    ]

    for turn_idx, phrase in enumerate(commitment_phrases, start=2):
        resp = client.post("/v1/reply", json={
            "conversation_id": f"{conv_id}_{turn_idx}",
            "merchant_id": "m_fix2",
            "from_role": "merchant",
            "message": phrase,
            "turn_number": turn_idx
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "send"
        assert data["body"] is not None
        assert isinstance(data["body"], str)
        assert len(data["body"].strip()) > 0
        assert "great" in data["body"].lower() or "next step" in data["body"].lower() or "proceeding" in data["body"].lower()

def test_fix3_fallback_messages_grounding():
    composer = LLMComposer()
    
    # 1. Missing full CategoryContext, but category_slug and payload available
    merchant = {
        "merchant_id": "m_gym_01",
        "name": "Fitness First",
        "category_slug": "gyms",
        "active_offers": ["Free 1 Week Pass"]
    }
    trigger = {
        "type": "perf_dip",
        "payload": {"dip_amount": "25%", "metric": "membership inquiries"}
    }

    res = composer.compose(decision={"selected_trigger": trigger}, merchant=merchant, trigger=trigger)
    assert res is not None
    assert "Fitness First" in res["body"]
    assert "25%" in res["body"]
    assert "membership inquiries" in res["body"]
    assert "Hi Merchant, here is a relevant update for Category" not in res["body"]

if __name__ == "__main__":
    test_fix1_context_response_accepted_true()
    test_fix2_positive_commitment_never_null_body()
    test_fix3_fallback_messages_grounding()
    print("ALL REGRESSION TESTS FOR FIXES 1, 2, AND 3 PASSED!")
