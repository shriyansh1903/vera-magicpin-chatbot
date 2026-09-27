from datetime import datetime, timezone
from context_store import ContextStore
from conversation import ConversationStore
from decision_engine import DecisionEngine, SuppressionRegistry

def test_context_versioning_and_idempotency():
    store = ContextStore()
    
    # Ingest v1
    status, ver = store.save_context("merchant", "m1", 1, {"name": "Merchant One"})
    assert status == "accepted"
    assert ver == 1
    assert store.get_merchant("m1")["name"] == "Merchant One"

    # Same version -> idempotent no-op
    status_noop, ver_noop = store.save_context("merchant", "m1", 1, {"name": "Merchant One"})
    assert status_noop == "noop"
    assert ver_noop == 1

    # Higher version -> atomic replacement
    status_up, ver_up = store.save_context("merchant", "m1", 2, {"name": "Merchant One Updated"})
    assert status_up == "accepted"
    assert ver_up == 2
    assert store.get_merchant("m1")["name"] == "Merchant One Updated"

    # Lower version -> stale
    status_stale, ver_stale = store.save_context("merchant", "m1", 1, {"name": "Stale Payload"})
    assert status_stale == "stale"
    assert ver_stale == 2

def test_getters_and_scopes():
    store = ContextStore()
    store.save_context("category", "cat1", 1, {"name": "Food & Beverage"})
    store.save_context("merchant", "m1", 1, {"name": "Pizza Place", "category_id": "cat1"})
    store.save_context("customer", "c1", 1, {"name": "Alice"})
    store.save_context("trigger", "trig1", 1, {
        "type": "perf_dip",
        "merchant_id": "m1",
        "customer_id": "c1",
        "urgency": "high"
    })

    assert store.get_category("cat1")["name"] == "Food & Beverage"
    assert store.get_merchant("m1")["name"] == "Pizza Place"
    assert store.get_customer("c1")["name"] == "Alice"
    assert store.get_trigger("trig1")["type"] == "perf_dip"

    counts = store.get_counts()
    assert counts == {"category": 1, "merchant": 1, "customer": 1, "trigger": 1}

def test_get_all_context_for_trigger():
    store = ContextStore()
    store.save_context("category", "cat1", 1, {"name": "Dining"})
    store.save_context("merchant", "m1", 1, {"name": "Dhaba", "category_id": "cat1"})
    store.save_context("customer", "c1", 1, {"name": "Bob"})
    store.save_context("trigger", "trig1", 1, {
        "type": "recall_due",
        "merchant_id": "m1",
        "customer_id": "c1"
    })

    ctx = store.get_all_context_for_trigger("trig1")
    assert ctx["category"]["name"] == "Dining"
    assert ctx["merchant"]["name"] == "Dhaba"
    assert ctx["customer"]["name"] == "Bob"
    assert ctx["trigger"]["type"] == "recall_due"

def test_missing_context_handling():
    store = ContextStore()
    # Trigger with missing category, merchant, customer
    store.save_context("trigger", "trig_orphan", 1, {"type": "scheduled_recurring"})
    
    ctx = store.get_all_context_for_trigger("trig_orphan")
    assert ctx["trigger"]["type"] == "scheduled_recurring"
    assert ctx["merchant"] is None
    assert ctx["category"] is None
    assert ctx["customer"] is None

def test_objective_mapping():
    engine = DecisionEngine(ContextStore(), ConversationStore())
    
    assert engine.map_objective("recall_due") == "drive_booking_or_action"
    assert engine.map_objective("appointment_tomorrow") == "drive_booking_or_action"
    assert engine.map_objective("customer_lapsed_soft") == "reengage_lapsed_customer"
    assert engine.map_objective("customer_lapsed_hard") == "winback_lapsed_customer"
    assert engine.map_objective("perf_dip") == "diagnose_and_improve_performance"
    assert engine.map_objective("perf_spike") == "capitalize_on_momentum"
    assert engine.map_objective("milestone_reached") == "reinforce_achievement_and_next_action"
    assert engine.map_objective("review_theme_emerged") == "address_review_theme"
    assert engine.map_objective("research_digest") == "share_relevant_insight"
    assert engine.map_objective("competitor_opened") == "surface_competitive_signal_and_action"
    assert engine.map_objective("festival") == "create_timely_merchant_opportunity"
    assert engine.map_objective("category_trend_movement") == "connect_demand_signal_to_merchant"
    assert engine.map_objective("curious_ask_due") == "generate_useful_curiosity_conversation"
    assert engine.map_objective("unknown_family_xyz") == "inform_or_assist"

def test_expired_trigger_filtering():
    store = ContextStore()
    engine = DecisionEngine(store, ConversationStore())

    # Expired trigger
    store.save_context("trigger", "trig_expired", 1, {
        "type": "perf_dip",
        "urgency": "high",
        "expires_at": "2026-09-01T00:00:00Z"
    })
    # Valid trigger
    store.save_context("trigger", "trig_valid", 1, {
        "type": "perf_spike",
        "urgency": "low",
        "expires_at": "2026-10-01T00:00:00Z"
    })

    now = "2026-09-15T00:00:00Z"
    decision = engine.make_decision(now, ["trig_expired", "trig_valid"])
    assert decision.selected_trigger["context_id"] == "trig_valid"

def test_suppression_and_version_unsuppress():
    store = ContextStore()
    engine = DecisionEngine(store, ConversationStore())

    store.save_context("trigger", "trig_suppress", 1, {
        "type": "recall_due",
        "urgency": 5,
        "suppression_key": "supp_key_1"
    })

    now = "2026-09-27T14:00:00Z"
    
    # First evaluation selects trig_suppress and suppresses (supp_key_1, 1)
    d1 = engine.make_decision(now, ["trig_suppress"])
    assert d1.selected_trigger["context_id"] == "trig_suppress"

    # Second evaluation with same trigger version finds it suppressed
    d2 = engine.make_decision(now, ["trig_suppress"])
    assert d2.selected_trigger is None

    # New version of same trigger context (version 2) resets suppression
    store.save_context("trigger", "trig_suppress", 2, {
        "type": "recall_due",
        "urgency": 5,
        "suppression_key": "supp_key_1"
    })

    d3 = engine.make_decision(now, ["trig_suppress"])
    assert d3.selected_trigger["context_id"] == "trig_suppress"
    assert d3.selected_trigger["version"] == 2

def test_trigger_ranking_urgency_and_actionability():
    store = ContextStore()
    engine = DecisionEngine(store, ConversationStore())

    # Low urgency action trigger vs high urgency info trigger
    store.save_context("trigger", "info_high", 1, {
        "type": "research_digest",
        "urgency": 10
    })
    store.save_context("trigger", "action_medium", 1, {
        "type": "recall_due",
        "urgency": 50
    })

    decision = engine.make_decision(None, ["info_high", "action_medium"])
    assert decision.selected_trigger["context_id"] == "action_medium"
    assert decision.objective == "drive_booking_or_action"

if __name__ == "__main__":
    test_context_versioning_and_idempotency()
    test_getters_and_scopes()
    test_get_all_context_for_trigger()
    test_missing_context_handling()
    test_objective_mapping()
    test_expired_trigger_filtering()
    test_suppression_and_version_unsuppress()
    test_trigger_ranking_urgency_and_actionability()
    print("ALL PHASE 2 UNIT TESTS PASSED SUCCESSFULLY!")
