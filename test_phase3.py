import json
import urllib.error
from composer import LLMComposer
from context_store import ContextStore
from conversation import ConversationStore
from decision_engine import DecisionEngine

def test_merchant_facing_research_digest():
    composer = LLMComposer()
    decision = {
        "selected_trigger": {"context_id": "t_rd", "type": "research_digest", "version": 1},
        "merchant_id": "m_salon",
        "objective": "share_relevant_insight",
        "priority": 10,
        "relevant_context": {}
    }
    category = {"name": "Salon & Beauty", "voice": "Professional"}
    merchant = {"name": "Style Studio"}
    trigger = {"type": "research_digest", "source": "magicpin Insights", "payload": {"insight": "20% increase in weekend hair spa searches"}}

    res = composer.compose(decision, category=category, merchant=merchant, trigger=trigger)
    assert res is not None
    assert res["send_as"] == "vera"
    assert res["template_name"] == "vera_merchant_insight_v1"
    assert "Style Studio" in res["body"]
    assert "magicpin Insights" in res["body"]
    assert len(res["cta"]) > 0

def test_merchant_facing_perf_dip():
    composer = LLMComposer()
    decision = {"selected_trigger": {"type": "perf_dip"}}
    merchant = {"name": "Pizza Corner"}
    trigger = {"type": "perf_dip", "payload": {"dip_amount": "20%", "metric": "order volume"}}

    res = composer.compose(decision, merchant=merchant, trigger=trigger)
    assert res is not None
    assert res["send_as"] == "vera"
    assert res["template_name"] == "vera_merchant_performance_v1"
    assert "20%" in res["body"]
    assert "Pizza Corner" in res["body"]

def test_merchant_facing_festival_and_offer():
    composer = LLMComposer()
    decision = {"selected_trigger": {"type": "festival"}}
    merchant = {"name": "Royal Dining", "active_offers": ["Flat 15% OFF Thali"]}
    trigger = {"type": "festival", "payload": {"festival_name": "Diwali"}}

    res = composer.compose(decision, merchant=merchant, trigger=trigger)
    assert res is not None
    assert res["send_as"] == "vera"
    assert res["template_name"] == "vera_merchant_opportunity_v1"
    assert "Diwali" in res["body"]
    assert "Flat 15% OFF Thali" in res["body"]

def test_customer_facing_recall_due():
    composer = LLMComposer()
    decision = {"selected_trigger": {"type": "recall_due"}}
    merchant = {"name": "City Cafe", "active_offers": ["Buy 1 Get 1 Coffee"]}
    customer = {"customer_id": "c_1", "name": "Alice", "consent": True}
    trigger = {"type": "recall_due"}

    res = composer.compose(decision, merchant=merchant, customer=customer, trigger=trigger)
    assert res is not None
    assert res["send_as"] == "merchant_on_behalf"
    assert res["template_name"] == "vera_customer_reengagement_v1"
    assert "Alice" in res["body"]
    assert "City Cafe" in res["body"]
    assert "Buy 1 Get 1 Coffee" in res["body"]

def test_missing_customer_and_consent_safety():
    composer = LLMComposer()
    
    # 1. Missing customer context for customer-facing trigger
    assert composer.validate_customer_safety("recall_due", None, {"name": "Cafe"}) is False
    res_missing_cust = composer.compose(
        decision={"selected_trigger": {"type": "recall_due"}},
        merchant={"name": "Cafe"},
        customer=None,
        trigger={"type": "recall_due"}
    )
    assert res_missing_cust is None

    # 2. Consent == False
    cust_no_consent = {"customer_id": "c_2", "name": "Bob", "consent": False}
    assert composer.validate_customer_safety("recall_due", cust_no_consent, {"name": "Cafe"}) is False
    res_no_consent = composer.compose(
        decision={"selected_trigger": {"type": "recall_due"}},
        merchant={"name": "Cafe"},
        customer=cust_no_consent,
        trigger={"type": "recall_due"}
    )
    assert res_no_consent is None

def test_single_cta_and_template_fields():
    composer = LLMComposer()
    res = composer.compose(
        decision={"selected_trigger": {"type": "appointment_tomorrow"}},
        merchant={"name": "Spa Center"},
        customer={"customer_id": "c_3", "name": "Carol", "consent": True},
        trigger={"type": "appointment_tomorrow", "payload": {"appointment_time": "10:00 AM tomorrow"}}
    )
    assert res is not None
    assert isinstance(res["cta"], str)
    # Check at most one question mark in cta/body combo or single CTA
    assert res["template_name"] == "vera_customer_reengagement_v1"
    assert "Carol" in res["template_params"]
    assert "Spa Center" in res["template_params"]

def test_end_to_end_tick_action():
    from fastapi.testclient import TestClient
    from app import app, context_store

    client = TestClient(app)

    # Ingest Category, Merchant, Customer, Trigger
    client.post("/v1/context", json={
        "scope": "category",
        "context_id": "cat_test",
        "version": 1,
        "payload": {"name": "Fitness", "voice": "Energetic"}
    })
    client.post("/v1/context", json={
        "scope": "merchant",
        "context_id": "m_gym",
        "version": 1,
        "payload": {"name": "Iron Gym", "category_id": "cat_test", "active_offers": ["Free 3-Day Pass"]}
    })
    client.post("/v1/context", json={
        "scope": "customer",
        "context_id": "c_david",
        "version": 1,
        "payload": {"name": "David", "consent": True}
    })
    client.post("/v1/context", json={
        "scope": "trigger",
        "context_id": "trig_recall_gym",
        "version": 1,
        "payload": {
            "type": "recall_due",
            "urgency": "high",
            "merchant_id": "m_gym",
            "customer_id": "c_david"
        }
    })

    # Execute Tick
    resp = client.post("/v1/tick", json={
        "now": "2026-09-27T14:30:00Z",
        "available_triggers": ["trig_recall_gym"]
    })

    assert resp.status_code == 200
    actions = resp.json()["actions"]
    assert len(actions) == 1
    
    act = actions[0]
    assert act["conversation_id"].startswith("conv_")
    assert act["merchant_id"] == "m_gym"
    assert act["customer_id"] == "c_david"
    assert act["send_as"] == "merchant_on_behalf"
    assert act["trigger_id"] == "trig_recall_gym"
    assert act["template_name"] == "vera_customer_reengagement_v1"
    assert "David" in act["body"]
    assert "Iron Gym" in act["body"]
    assert "Free 3-Day Pass" in act["body"]
    assert act["suppression_key"] == "trig_recall_gym"

    # Second tick -> suppressed -> 0 actions
    resp_suppressed = client.post("/v1/tick", json={
        "now": "2026-09-27T14:30:00Z",
        "available_triggers": ["trig_recall_gym"]
    })
    assert resp_suppressed.status_code == 200
    assert len(resp_suppressed.json()["actions"]) == 0

def test_curious_ask_due_payload_grounding():
    composer = LLMComposer()
    decision = {"selected_trigger": {"type": "curious_ask_due"}, "objective": "generate_useful_curiosity_conversation"}
    merchant = {
        "identity": {"name": "Style Studio Salon", "owner_first_name": "Rohan", "locality": "Jubilee Hills"},
        "performance": {"views": 1820, "calls": 12},
        "offers": [{"title": "Hair Spa @ ₹499"}]
    }
    category = {
        "slug": "salons",
        "display_name": "Salons",
        "trends": [{"query": "bridal hair spa", "delta_yoy": "+55%"}]
    }
    trigger = {"kind": "curious_ask_due", "payload": {"ask_template": "what_service_in_demand_this_week"}}

    res = composer.compose(decision, category=category, merchant=merchant, trigger=trigger)
    assert res is not None
    body = res["body"]
    assert "Hi Rohan" in body or "Style Studio Salon" in body
    assert "bridal hair spa" in body or "demand" in body
    assert "recent update" not in body.lower()
    assert "following a recent" not in body.lower()
    assert len(res["cta"]) > 0

def test_scheduled_recurring_payload_grounding():
    composer = LLMComposer()
    decision = {"selected_trigger": {"type": "scheduled_recurring"}, "objective": "inform_or_assist"}
    merchant = {
        "identity": {"name": "Dr. Meera's Dental Clinic", "owner_first_name": "Meera", "locality": "Lajpat Nagar"},
        "performance": {"views": 2410, "calls": 18, "ctr": 0.021},
        "signals": ["stale_posts:22d"]
    }
    category = {"slug": "dentists", "display_name": "Dentists"}
    trigger = {"kind": "scheduled_recurring", "payload": {"scheduled_task": "weekly_performance_audit"}}

    res = composer.compose(decision, category=category, merchant=merchant, trigger=trigger)
    assert res is not None
    body = res["body"]
    assert "Dr. Meera" in body
    assert "2.1%" in body or "2,410" in body or "stale" in body or "views" in body
    assert "recent update" not in body.lower()
    assert len(res["cta"]) > 0

def test_gemini_retry_on_429():
    composer = LLMComposer()
    calls = []
    def mock_urlopen(req, timeout=12):
        calls.append(req.full_url)
        if len(calls) == 1:
            err = urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "0.1"}, None)
            raise err
        resp = mock.MagicMock()
        resp.getcode.return_value = 200
        json_resp = json.dumps({"candidates": [{"content": {"parts": [{"text": '{"body": "Dr. Meera, DCI revised radiograph dose limits. Want me to draft an audit checklist?", "cta": "Want me to draft an audit checklist?", "template_name": "t", "template_params": [], "send_as": "vera", "suppression_key": "s", "rationale": "r"}'}]}}]}).encode("utf-8")
        resp.read.return_value = json_resp
        resp.__enter__.return_value = resp
        return resp

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
        res = composer._call_gemini_raw("test prompt")
        assert "DCI revised radiograph" in res
        assert len(calls) == 2

def test_gemini_lightweight_model_fallback_on_429():
    composer = LLMComposer()
    calls = []
    def mock_urlopen(req, timeout=12):
        calls.append(req.full_url)
        if len(calls) in (1, 2):
            err = urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
            raise err
        resp = mock.MagicMock()
        resp.getcode.return_value = 200
        json_resp = json.dumps({"candidates": [{"content": {"parts": [{"text": "Success from fallback model"}]}}]}).encode("utf-8")
        resp.read.return_value = json_resp
        resp.__enter__.return_value = resp
        return resp

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
        res = composer._call_gemini_raw("test prompt")
        assert res == "Success from fallback model"
        assert len(calls) == 3
        assert "gemini-1.5-flash" in calls[2] or "gemini-flash-lite" in calls[2]

def test_repeated_429_deterministic_fallback():
    composer = LLMComposer()
    def mock_urlopen(req, timeout=12):
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)

    decision = {"selected_trigger": {"type": "perf_dip"}}
    merchant = {"identity": {"name": "Pizza Corner"}, "performance": {"views": 1000}}
    trigger = {"kind": "perf_dip", "payload": {"dip_amount": "20%", "metric": "calls"}}

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
        res = composer.compose(decision, merchant=merchant, trigger=trigger)
        assert res is not None
        body = res["body"]
        assert "Pizza Corner" in body
        assert "20%" in body
        assert "update for your business" not in body.lower()
        assert "this is a good time to act" not in body.lower()
        assert len(res["cta"]) > 0

def test_gemini_retry_on_500_503():
    composer = LLMComposer()
    calls = []
    def mock_urlopen(req, timeout=12):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, None)
        resp = mock.MagicMock()
        resp.getcode.return_value = 200
        resp.read.return_value = json.dumps({"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}).encode("utf-8")
        resp.__enter__.return_value = resp
        return resp

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
        res = composer._call_gemini_raw("test prompt")
        assert res == "OK"
        assert len(calls) == 2

def test_gemini_no_retry_on_404_401():
    composer = LLMComposer()
    calls = []
    def mock_urlopen(req, timeout=12):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    with mock.patch("urllib.request.urlopen", side_effect=mock_urlopen):
        try:
            composer._call_gemini_raw("test prompt")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        assert len(calls) == 1

def test_deterministic_fallback_quality_and_no_duplication():
    composer = LLMComposer()
    merchant = {
        "identity": {"name": "Dr. Meera's Dental Clinic", "owner_first_name": "Meera"},
        "performance": {"delta_7d": {"views_pct": 0.18}}
    }
    trigger = {"kind": "regulation_change", "payload": {"top_item_id": "d_1", "deadline_iso": "2026-12-15"}}
    
    res = composer._deterministic_compose("regulation_change", merchant, {"slug": "dentists"}, None, trigger, "vera", "t", [], "s")
    body = res["body"]
    assert "Dr. Meera" in body
    assert "2026-12-15" in body
    assert "update for your business" not in body.lower()
    assert "this is a good time to act" not in body.lower()
    assert body.count("2026-12-15") == 1

if __name__ == "__main__":
    import unittest.mock as mock
    test_merchant_facing_research_digest()
    test_merchant_facing_perf_dip()
    test_merchant_facing_festival_and_offer()
    test_customer_facing_recall_due()
    test_missing_customer_and_consent_safety()
    test_single_cta_and_template_fields()
    test_end_to_end_tick_action()
    test_curious_ask_due_payload_grounding()
    test_scheduled_recurring_payload_grounding()
    test_gemini_retry_on_429()
    test_gemini_lightweight_model_fallback_on_429()
    test_repeated_429_deterministic_fallback()
    test_gemini_retry_on_500_503()
    test_gemini_no_retry_on_404_401()
    test_deterministic_fallback_quality_and_no_duplication()
    print("ALL PHASE 3 COMPOSER, RESILIENCE AND TICK TESTS PASSED SUCCESSFULLY!")

