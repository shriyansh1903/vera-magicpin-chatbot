import os
import json
import glob
from context_store import ContextStore
from composer import LLMComposer

FORBIDDEN_GENERIC_PHRASES = [
    "weekly performance audit",
    "update for your business",
    "following a recent update",
    "following your recent update",
    "this is a good time to act",
    "regular profile optimization",
    "let me know if you'd like help"
]

EXPECTED_RENDERERS = {
    "research_digest": "vera_merchant_insight_v1",
    "regulation_change": "vera_merchant_alert_v1",
    "recall_due": "vera_customer_reengagement_v1",
    "perf_dip": "vera_merchant_performance_v1",
    "renewal_due": "vera_merchant_alert_v1",
    "festival_upcoming": "vera_merchant_opportunity_v1",
    "wedding_package_followup": "vera_customer_reengagement_v1",
    "curious_ask_due": "vera_merchant_insight_v1",
    "winback_eligible": "vera_merchant_opportunity_v1",
    "ipl_match_today": "vera_merchant_opportunity_v1",
    "review_theme_emerged": "vera_merchant_alert_v1",
    "milestone_reached": "vera_merchant_performance_v1",
    "active_planning_intent": "vera_merchant_opportunity_v1",
    "seasonal_perf_dip": "vera_merchant_performance_v1",
    "customer_lapsed_hard": "vera_customer_reengagement_v1",
    "trial_followup": "vera_customer_reengagement_v1",
    "supply_alert": "vera_merchant_alert_v1",
    "chronic_refill_due": "vera_customer_reengagement_v1",
    "category_seasonal": "vera_merchant_opportunity_v1",
    "gbp_unverified": "vera_merchant_alert_v1",
    "cde_opportunity": "vera_merchant_opportunity_v1",
    "competitor_opened": "vera_merchant_alert_v1",
    "perf_spike": "vera_merchant_performance_v1",
    "dormant_with_vera": "vera_proactive_notification_v1"
}


def load_all_dataset_contexts():
    cs = ContextStore()
    
    # Load categories
    cat_files = glob.glob("magicpin-ai-challenge/dataset/categories/*.json")
    for filepath in cat_files:
        with open(filepath, "r", encoding="utf-8") as f:
            cat_data = json.load(f)
            slug = cat_data.get("slug")
            if slug:
                cs.save_context("category", slug, 1, cat_data)

    # Load merchants
    if os.path.exists("magicpin-ai-challenge/dataset/merchants_seed.json"):
        with open("magicpin-ai-challenge/dataset/merchants_seed.json", "r", encoding="utf-8") as f:
            m_data = json.load(f)
            for merch in m_data.get("merchants", []):
                mid = merch.get("merchant_id")
                if mid:
                    cs.save_context("merchant", mid, 1, merch)

    # Load customers
    if os.path.exists("magicpin-ai-challenge/dataset/customers_seed.json"):
        with open("magicpin-ai-challenge/dataset/customers_seed.json", "r", encoding="utf-8") as f:
            c_data = json.load(f)
            for cust in c_data.get("customers", []):
                cid = cust.get("customer_id")
                if cid:
                    cs.save_context("customer", cid, 1, cust)

    # Load 25 seed triggers
    with open("magicpin-ai-challenge/dataset/triggers_seed.json", "r", encoding="utf-8") as f:
        trig_data = json.load(f)
        for trig in trig_data.get("triggers", []):
            tid = trig.get("id")
            if tid:
                cs.save_context("trigger", tid, 1, trig)
                
    return cs, trig_data.get("triggers", [])


def test_deterministic_fallback_all_25_triggers():
    cs, triggers = load_all_dataset_contexts()
    composer = LLMComposer()

    assert len(triggers) == 25, f"Expected 25 seed triggers, found {len(triggers)}"

    for trig in triggers:
        trig_id = trig["id"]
        trig_kind = trig["kind"]

        # Fetch context
        all_ctx = cs.get_all_context_for_trigger(trig_id)
        merchant = all_ctx.get("merchant")
        category = all_ctx.get("category")
        customer = all_ctx.get("customer")

        # Select template & compose deterministically
        send_as = "merchant_on_behalf" if (trig_kind in ["recall_due", "wedding_package_followup", "customer_lapsed_hard", "trial_followup", "chronic_refill_due"]) else "vera"
        template_name, template_params = composer.select_template(trig_kind, {})

        res = composer._deterministic_compose(
            trigger_type=trig_kind,
            merchant=merchant,
            category=category,
            customer=customer,
            trigger=trig,
            send_as=send_as,
            template_name=template_name,
            template_params=template_params,
            suppression_key=trig.get("suppression_key", trig_id)
        )

        assert res is not None, f"Composition result is None for trigger {trig_id}"
        body = res.get("body", "")
        cta = res.get("cta", "")

        assert len(body) > 10, f"Body too short for {trig_id}: {body}"
        assert len(cta) > 0, f"Missing CTA for {trig_id}"
        
        # 1. Check template/renderer selection
        expected_template = EXPECTED_RENDERERS.get(trig_kind, "vera_proactive_notification_v1")
        assert res.get("template_name") == expected_template, (
            f"Trigger {trig_id} ({trig_kind}) mapped to template {res.get('template_name')}, expected {expected_template}"
        )

        # 2. Check forbidden generic phrases
        body_lower = body.lower()
        for phrase in FORBIDDEN_GENERIC_PHRASES:
            assert phrase not in body_lower, (
                f"Trigger {trig_id} contains forbidden generic phrase '{phrase}' in body: '{body}'"
            )

        # 3. Check single CTA
        # CTA should start with or contain single action verb, no multiple CTAs
        assert not ("let me know" in cta.lower() and "would you like" in cta.lower()), f"Multiple CTAs in {trig_id}"

        # 4. Payload evidence verification
        payload = trig.get("payload", {})
        if "deadline_iso" in payload:
            assert payload["deadline_iso"] in body, f"Missing deadline {payload['deadline_iso']} in {trig_id}"
        if "days_remaining" in payload:
            assert str(payload["days_remaining"]) in body, f"Missing days_remaining in {trig_id}"
        if "renewal_amount" in payload:
            assert f"{payload['renewal_amount']:,}" in body or str(payload["renewal_amount"]) in body, f"Missing renewal_amount in {trig_id}"
        if "festival" in payload:
            assert payload["festival"] in body, f"Missing festival name in {trig_id}"
        if "competitor_name" in payload:
            assert payload["competitor_name"] in body, f"Missing competitor_name in {trig_id}"
        if "match" in payload:
            assert payload["match"] in body, f"Missing match name in {trig_id}"

        safe_body = body[:70].encode('ascii', errors='replace').decode('ascii')
        safe_cta = cta.encode('ascii', errors='replace').decode('ascii')
        print(f"PASS: {trig_id} ({trig_kind}) -> {safe_body}... [CTA: {safe_cta}]")


if __name__ == "__main__":
    test_deterministic_fallback_all_25_triggers()
    print("\nALL 25 DETERMINISTIC FALLBACK TRIGGER TESTS PASSED SUCCESSFULLY!")
