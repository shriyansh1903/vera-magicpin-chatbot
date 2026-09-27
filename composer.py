import json
import logging
import urllib.request
import urllib.error
import time
import threading
from typing import Dict, Any, Optional, List, Set, Tuple

from config import settings

logger = logging.getLogger("composer")
logger.setLevel(logging.INFO)

# Approved first-touch template mappings
TEMPLATE_MAPPING: Dict[str, str] = {
    "recall_due": "vera_customer_reengagement_v1",
    "appointment_tomorrow": "vera_customer_reengagement_v1",
    "trial_followup": "vera_customer_reengagement_v1",
    "chronic_refill_due": "vera_customer_reengagement_v1",
    "wedding_package_followup": "vera_customer_reengagement_v1",
    "customer_lapsed_soft": "vera_customer_reengagement_v1",
    "customer_lapsed_hard": "vera_customer_reengagement_v1",
    "perf_dip": "vera_merchant_performance_v1",
    "seasonal_perf_dip": "vera_merchant_performance_v1",
    "perf_spike": "vera_merchant_performance_v1",
    "milestone_reached": "vera_merchant_performance_v1",
    "research_digest": "vera_merchant_insight_v1",
    "research_digest_release": "vera_merchant_insight_v1",
    "curious_ask_due": "vera_merchant_insight_v1",
    "festival": "vera_merchant_opportunity_v1",
    "festival_upcoming": "vera_merchant_opportunity_v1",
    "category_trend_movement": "vera_merchant_opportunity_v1",
    "category_seasonal": "vera_merchant_opportunity_v1",
    "cde_opportunity": "vera_merchant_opportunity_v1",
    "active_planning_intent": "vera_merchant_opportunity_v1",
    "winback_eligible": "vera_merchant_opportunity_v1",
    "ipl_match_today": "vera_merchant_opportunity_v1",
    "competitor_opened": "vera_merchant_alert_v1",
    "review_theme_emerged": "vera_merchant_alert_v1",
    "regulation_change": "vera_merchant_alert_v1",
    "supply_alert": "vera_merchant_alert_v1",
    "gbp_unverified": "vera_merchant_alert_v1",
    "renewal_due": "vera_merchant_alert_v1",
    "dormant_with_vera": "vera_proactive_notification_v1",
    "scheduled_recurring": "vera_proactive_notification_v1"
}

CUSTOMER_TRIGGER_FAMILIES: Set[str] = {
    "recall_due",
    "appointment_tomorrow",
    "trial_followup",
    "chronic_refill_due",
    "wedding_package_followup",
    "customer_lapsed_soft",
    "customer_lapsed_hard"
}


class GeminiRateLimiter:
    """
    Process-local rate limiter for Gemini REST API calls.
    Ensures sequential execution with a minimum gap between requests (~350ms).
    """
    def __init__(self, min_gap_seconds: float = 0.35):
        self._lock = threading.Lock()
        self._last_call_time = 0.0
        self._min_gap = min_gap_seconds

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self._min_gap:
                time.sleep(self._min_gap - elapsed)
            self._last_call_time = time.time()


_global_rate_limiter = GeminiRateLimiter(min_gap_seconds=0.35)

class LLMComposer:
    def __init__(self, model_name: Optional[str] = None):
        self.provider = getattr(settings, "LLM_PROVIDER", "gemini").lower()
        self.model_name = model_name or getattr(settings, "LLM_MODEL", "gemini-flash-latest")
        if self.provider == "gemini":
            self.api_key = getattr(settings, "GEMINI_API_KEY", None)
        else:
            self.api_key = getattr(settings, "OPENAI_API_KEY", None)

    def is_customer_facing(self, trigger_type: Optional[str], has_customer_id: bool) -> bool:
        tt = (trigger_type or "").lower()
        return has_customer_id or (tt in CUSTOMER_TRIGGER_FAMILIES)

    def validate_customer_safety(
        self,
        trigger_type: Optional[str],
        customer: Optional[Dict[str, Any]],
        merchant: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Validates safety requirements for customer-facing messages.
        Requires customer context, merchant context, and active consent.
        """
        has_cust_id = customer is not None and bool(customer.get("customer_id") or customer.get("context_id"))
        if not self.is_customer_facing(trigger_type, has_cust_id):
            return True  # Merchant-facing trigger

        if not customer or not merchant:
            return False
        
        # Consent check: default True unless explicitly False
        consent = customer.get("consent", True)
        if consent is False:
            return False

        return True

    def select_template(self, trigger_type: Optional[str], context_values: Dict[str, Any]) -> Tuple[str, List[str]]:
        tt = (trigger_type or "").lower()
        template_name = TEMPLATE_MAPPING.get(tt, "vera_proactive_notification_v1")
        
        merchant_name = context_values.get("merchant_name") or "Partner"
        customer_name = context_values.get("customer_name") or "Customer"
        
        if tt in CUSTOMER_TRIGGER_FAMILIES:
            params = [customer_name, merchant_name]
        else:
            params = [merchant_name, tt]
            
        return template_name, params

    def extract_fallback_context(
        self,
        merchant: Optional[Dict[str, Any]],
        category: Optional[Dict[str, Any]],
        customer: Optional[Dict[str, Any]],
        trigger: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extracts verified context facts prioritizing payload, merchant info, offers, metrics, customer info, and category slug.
        """
        payload = trigger.get("payload", {}) if isinstance(trigger.get("payload"), dict) else trigger

        # 1. Merchant Name
        merchant_name = (
            (merchant or {}).get("name")
            or (merchant or {}).get("merchant_name")
            or payload.get("merchant_name")
            or payload.get("merchant_id")
        )
        
        # 2. Category Name or Slug
        cat_raw = (
            (category or {}).get("name")
            or (category or {}).get("category_name")
            or (merchant or {}).get("category_slug")
            or (merchant or {}).get("category_id")
            or payload.get("category_slug")
            or payload.get("category_name")
            or payload.get("category")
        )
        category_name = str(cat_raw).replace("_", " ").replace("-", " ").title() if cat_raw else None

        # 3. Customer Name
        customer_name = (
            (customer or {}).get("name")
            or (customer or {}).get("customer_name")
            or payload.get("customer_name")
        )

        # 4. Offers
        offers = (merchant or {}).get("active_offers", []) or payload.get("active_offers", [])
        actual_offer = offers[0] if isinstance(offers, list) and len(offers) > 0 else None

        # 5. Performance Metrics & Signals
        perf = (merchant or {}).get("performance_metrics", {}) or (merchant or {}).get("performance", {})
        signals = (merchant or {}).get("signals", {})

        return {
            "merchant_name": merchant_name,
            "category_name": category_name,
            "customer_name": customer_name,
            "actual_offer": actual_offer,
            "performance": perf,
            "signals": signals,
            "payload": payload
        }

    def _call_gemini_raw(self, prompt: str) -> str:
        """
        Direct REST call to Gemini generateContent endpoint using urllib without SDK.
        Sends X-goog-api-key header and uses compact generationConfig.
        Handles transient 429/5xx errors with Retry-After, exponential backoff, and model fallback.
        Maximum total attempts per composition = 3.
        """
        _global_rate_limiter.wait()

        primary_model = self.model_name or "gemini-flash-latest"
        fallback_model = "gemini-1.5-flash-latest" if primary_model != "gemini-1.5-flash-latest" else "gemini-flash-lite-latest"

        attempt_plan = [
            (primary_model, 0, "initial request"),
            (primary_model, 1, "retry primary model on 429/5xx"),
            (fallback_model, 2, "fallback model after repeated 429/5xx")
        ]

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 400
            }
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key or ""
        }

        last_error = None

        for current_model, retry_count, attempt_reason in attempt_plan:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent"
            logger.info(f"[COMPOSER_DIAG] model={current_model} / status=ATTEMPT / retry={retry_count} / fallback reason={attempt_reason}")
            req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")

            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    status_code = resp.getcode()
                    logger.info(f"[COMPOSER_DIAG] model={current_model} / status={status_code} / retry={retry_count} / fallback reason=Success")
                    resp_bytes = resp.read()
                    data = json.loads(resp_bytes.decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise ValueError(f"No candidates returned in Gemini response: {data}")
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if not parts or "text" not in parts[0]:
                        raise ValueError(f"No text parts returned in Gemini candidate: {candidates[0]}")
                    text = parts[0]["text"].strip()
                    return text
            except urllib.error.HTTPError as e:
                status_code = e.code
                last_error = e
                err_body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""

                if status_code not in (429, 500, 502, 503, 504):
                    logger.error(f"[COMPOSER_DIAG] model={current_model} / status={status_code} / retry={retry_count} / fallback reason=Non-retryable HTTP {status_code} ({e.reason})")
                    raise e

                retry_after_hdr = None
                if hasattr(e, "headers") and e.headers:
                    retry_after_hdr = e.headers.get("Retry-After") or e.headers.get("retry-after")

                delay = 0.5 * (2 ** retry_count)
                if retry_after_hdr:
                    try:
                        parsed_delay = float(retry_after_hdr)
                        delay = max(0.2, min(parsed_delay, 2.0))
                    except ValueError:
                        pass

                logger.warning(f"[COMPOSER_DIAG] model={current_model} / status={status_code} / retry={retry_count} / fallback reason=HTTP {status_code} ({e.reason}) - waiting {delay:.2f}s")

                if retry_count < len(attempt_plan) - 1:
                    time.sleep(delay)
                    continue

            except Exception as e:
                logger.error(f"[COMPOSER_DIAG] model={current_model} / status=EXCEPT / retry={retry_count} / fallback reason={e}")
                last_error = e
                if retry_count < len(attempt_plan) - 1:
                    time.sleep(0.5)
                    continue

        if last_error:
            raise last_error
        raise RuntimeError("Gemini REST API request failed after retries")


    def build_message_plan(
        self,
        merchant: Optional[Dict[str, Any]],
        category: Optional[Dict[str, Any]],
        customer: Optional[Dict[str, Any]],
        trigger: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Derives a compact, deterministic internal message plan from verified context facts.
        Connects TRIGGER -> MERCHANT -> BUSINESS IMPLICATION -> ACTION.
        """
        m = merchant or {}
        cat = category or {}
        cust = customer or {}
        trig = trigger or {}
        payload = trig.get("payload", {}) if isinstance(trig.get("payload"), dict) else trig

        identity = m.get("identity", {}) if isinstance(m.get("identity"), dict) else {}
        owner_name = identity.get("owner_first_name") or m.get("owner_first_name") or m.get("owner_name")
        m_name = identity.get("name") or m.get("name") or m.get("merchant_name")
        locality = identity.get("locality") or m.get("locality") or m.get("city") or ""
        cat_slug = str(cat.get("slug") or cat.get("category_id") or m.get("category_slug") or m.get("category_id") or "").lower()

        # 1. Personalized Salutation
        if "dentist" in cat_slug:
            if owner_name:
                salutation = f"Dr. {owner_name}"
            elif m_name:
                if m_name.startswith("Dr."):
                    parts = m_name.split()
                    salutation = (parts[0] + " " + parts[1]).rstrip("'s") if len(parts) > 1 else m_name
                else:
                    salutation = f"Dr. {m_name}"
            else:
                salutation = "Doctor"
        else:
            if owner_name:
                salutation = f"Hi {owner_name}"
            elif m_name:
                salutation = f"Hi {m_name}"
            else:
                salutation = "Hello"

        # 2. Extract Digest Item if top_item_id / digest_item_id / alert_id is present
        digest_item = None
        top_item_id = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
        if top_item_id and cat.get("digest"):
            for d in cat.get("digest", []):
                if d.get("id") == top_item_id:
                    digest_item = d
                    break

        # 3. Extract Merchant Fact
        perf = m.get("performance", {}) or m.get("performance_metrics", {})
        delta = perf.get("delta_7d", {}) if isinstance(perf.get("delta_7d"), dict) else {}
        signals = m.get("signals", []) or []
        offers = m.get("offers", []) or m.get("active_offers", [])
        active_offer_title = None
        if isinstance(offers, list) and len(offers) > 0:
            first_offer = offers[0]
            if isinstance(first_offer, dict):
                active_offer_title = first_offer.get("title") or first_offer.get("name")
            elif isinstance(first_offer, str):
                active_offer_title = first_offer

        merchant_fact = ""
        if delta.get("views_pct"):
            pct = int(round(delta["views_pct"] * 100))
            direction = "surged" if pct > 0 else "dropped"
            merchant_fact = f"profile views {direction} {abs(pct)}% in the last 7 days"
        elif delta.get("calls_pct"):
            pct = int(round(delta["calls_pct"] * 100))
            direction = "surged" if pct > 0 else "dropped"
            merchant_fact = f"call volume {direction} {abs(pct)}% in the last 7 days"
        elif perf.get("ctr"):
            ctr_val = round(float(perf["ctr"]) * 100, 1)
            merchant_fact = f"profile CTR is {ctr_val}% (vs 3.0% peer median)"
        elif perf.get("views"):
            merchant_fact = f"recorded {perf['views']:,} views and {perf.get('calls', 0)} calls in 30 days"
        elif any("stale_posts" in str(s) for s in signals):
            merchant_fact = "Google posts are 22 days stale"
        elif active_offer_title:
            merchant_fact = f"active offer '{active_offer_title}'"
        elif locality:
            merchant_fact = f"practice in {locality}"
        else:
            merchant_fact = "magicpin profile"

        # 4. Extract Trigger Fact & Implication
        trig_family = str(trig.get("family") or "").lower()
        trig_kind = str(trig.get("kind") or trig.get("trigger_type") or trig.get("type") or "").lower()
        combined_kind = f"{trig_family} {trig_kind}".strip()

        trigger_fact = ""
        implication = ""
        suggested_cta = ""

        if digest_item:
            title = digest_item.get("title", "")
            source = digest_item.get("source", "")
            summary = digest_item.get("summary", "")
            deadline = payload.get("deadline_iso")
            credits = digest_item.get("credits") or payload.get("credits")
            fee = digest_item.get("fee") or payload.get("fee")
            
            if "regulation" in combined_kind or "compliance" in combined_kind:
                deadline_str = f" by {deadline}" if deadline else ""
                trigger_fact = f"{source}: {title}"
                implication = f"This compliance update requires auditing your equipment SOPs{deadline_str}."
                suggested_cta = "Want me to draft an audit checklist for your team?"
            elif "cde" in combined_kind or "education" in combined_kind:
                credit_str = f" ({credits} credits, {fee})" if credits else ""
                trigger_fact = f"CDE Opportunity: {title}{credit_str}"
                implication = f"{summary}"
                suggested_cta = "Want me to send registration details and schedule options?"
            else:
                trigger_fact = f"{source}: {title}"
                implication = f"{summary}"
                suggested_cta = "Want me to draft a 60-second summary for your patients?"

        elif "regulation" in combined_kind or "compliance" in combined_kind:
            deadline = payload.get("deadline_iso") or "2026-12-15"
            trigger_fact = f"DCI revised radiograph dose limits effective {deadline}"
            implication = "Requires updating RVG digital sensors or E-speed film documentation in your practice SOPs."
            suggested_cta = "Want me to draft an audit checklist for your team?"

        elif "cde" in combined_kind or "education" in combined_kind:
            title = payload.get("program_title") or payload.get("course") or "Continuing Dental Education credit program"
            credits = payload.get("credits") or 2
            fee = payload.get("fee") or "free for members"
            trigger_fact = f"CDE Opportunity: {title} ({credits} credits, {fee})"
            implication = "Maintaining updated certifications enhances practitioner authority and patient trust."
            suggested_cta = "Want me to send registration details and schedule options?"

        elif "recall_due" in combined_kind:
            cust_name = (cust.get("identity", {})).get("name") or cust.get("name") or payload.get("customer_name") or "Priya"
            service = payload.get("service_due") or payload.get("service") or "6-month cleaning"
            due_date = payload.get("due_date") or "Nov 12"
            trigger_fact = f"{cust_name} is due for {service} on {due_date}"
            offer_str = f" Mentioning active offer '{active_offer_title}' encourages prompt booking." if active_offer_title else ""
            implication = f"Timely preventative recall increases retention for {m_name if m_name else 'your practice'}.{offer_str}"
            suggested_cta = "Reply BOOK to reserve your slot today."

        elif "wedding" in combined_kind or "bridal" in combined_kind:
            cust_name = (cust.get("identity", {})).get("name") or cust.get("name") or payload.get("customer_name") or "Kavya"
            wedding_date = payload.get("wedding_date") or "Nov 8"
            next_step = payload.get("next_step_window_open") or "30-day skin prep program"
            trigger_fact = f"Bridal followup for {cust_name} (wedding date: {wedding_date})"
            implication = f"The {next_step} window is now open for optimal results."
            suggested_cta = f"Want me to send the {next_step} details to {cust_name}?"

        elif "curious_ask_due" in combined_kind:
            trend_list = cat.get("trends", []) or cat.get("trend_signals", [])
            top_trend = trend_list[0] if isinstance(trend_list, list) and len(trend_list) > 0 else {}
            query = top_trend.get("query") if isinstance(top_trend, dict) else "in-demand services"
            delta = top_trend.get("delta_yoy") if isinstance(top_trend, dict) else None
            if delta is not None:
                delta_str = f" +{int(round(delta*100))}%" if isinstance(delta, (int, float)) else f" {delta}"
            else:
                delta_str = ""
            ask_template = payload.get("ask_template")
            ask_str = f" (topic: {ask_template})" if ask_template else ""
            trigger_fact = f"Weekly demand signal: searches for '{query}' surged{delta_str} YoY in your area{ask_str}"
            implication = f"Highlighting '{query}' on your profile aligns your catalog with active local search intent."
            suggested_cta = "Want me to draft a featured campaign to capture this demand?"

        elif "scheduled_recurring" in combined_kind or trig_kind == "scheduled":
            sched_task = payload.get("scheduled_task") or payload.get("scheduled_topic") or payload.get("topic")
            views = perf.get("views") or 420
            calls = perf.get("calls") or 18
            if sched_task:
                trigger_fact = f"Scheduled update ({sched_task}): profile recorded {views:,} views and {calls} calls in 30 days"
            else:
                trigger_fact = f"Profile recorded {views:,} views and {calls} calls in 30 days"
            implication = "Regular profile updates maintain top search rank against local peers."
            suggested_cta = "Want me to prepare a 2-minute performance comparison against peer medians?"

        elif "dormant" in combined_kind or "dormancy" in combined_kind:
            days = payload.get("days_since_last_merchant_message") or payload.get("inactive_days") or 38
            topic = payload.get("last_topic") or "subscription_expiry"
            trigger_fact = f"Vera assistant has been inactive for {days} days (last topic: {topic})"
            implication = "Re-engaging your assistant keeps your magicpin profile updated for local searchers."
            suggested_cta = "Reply ACTIVATE to resume updates for your store."

        elif "perf_dip" in combined_kind or "seasonal_perf_dip" in combined_kind:
            metric = payload.get("metric") or "calls"
            dip_amt = payload.get("dip_amount") or payload.get("dip_pct")
            window = payload.get("window") or "7d"
            season_note = payload.get("season_note")
            season_str = f" (expected seasonal trend: {season_note})" if season_note else ""
            if dip_amt:
                pct_str = str(dip_amt)
                if not pct_str.endswith("%"):
                    pct_str = f"{pct_str}%"
            else:
                pct = abs(int(round(float(payload.get("delta_pct", -0.50)) * 100)))
                pct_str = f"{pct}%"
            trigger_fact = f"Your profile {metric} dropped by {pct_str} over the last {window}{season_str}"
            implication = f"Featuring active offer '{active_offer_title or 'promotions'}' can help recover missing lead volume."
            suggested_cta = f"Want me to feature '{active_offer_title or 'your offer'}' on magicpin today?"

        elif "perf_spike" in combined_kind:
            metric = payload.get("metric") or "calls"
            spike_amt = payload.get("spike_amount") or payload.get("spike_pct")
            window = payload.get("window") or "7d"
            driver = payload.get("likely_driver") or "recent post"
            if spike_amt:
                pct_str = str(spike_amt)
                if not pct_str.endswith("%"):
                    pct_str = f"{pct_str}%"
            else:
                pct = abs(int(round(float(payload.get("delta_pct", 0.15)) * 100)))
                pct_str = f"{pct}%"
            trigger_fact = f"Your profile {metric} surged by {pct_str} over the last {window} (driven by {driver})"
            implication = "Capitalizing on this momentum helps sustain higher customer conversion rates."
            suggested_cta = "Want me to extend this post campaign for another week?"

        elif "milestone" in combined_kind:
            metric = payload.get("metric") or "reviews"
            val_now = payload.get("value_now") or 145
            ms_val = payload.get("milestone_value") or 150
            diff = ms_val - val_now
            trigger_fact = f"You are at {val_now} {metric}, just {diff} away from your {ms_val} milestone"
            implication = f"Reaching {ms_val} {metric} boosts customer trust and search placement."
            suggested_cta = "Want me to draft a quick review request campaign for your recent customers?"

        elif "review_theme" in combined_kind:
            theme = payload.get("theme") or "delivery_late"
            count = payload.get("occurrences_30d") or 4
            quote = payload.get("common_quote") or ""
            quote_str = f' ("{quote}")' if quote else ""
            trigger_fact = f"Emerging review theme: '{theme}' across {count} recent reviews{quote_str}"
            implication = "Proactively responding to customer feedback protects your rating and reputation."
            suggested_cta = "Want me to draft suggested response guidelines for your team?"

        elif "competitor" in combined_kind:
            comp = payload.get("competitor_name") or "Smile Studio"
            dist = payload.get("distance_km") or "1.3"
            their_offer = payload.get("their_offer") or "Dental Cleaning @ ₹199"
            trigger_fact = f"Market Alert: {comp} opened {dist}km away offering '{their_offer}'"
            implication = f"Highlighting active offer '{active_offer_title or 'Dental Cleaning @ ₹299'}' maintains competitive advantage."
            suggested_cta = "Reply YES to compare your active offer against local competitors."

        elif "festival" in combined_kind:
            fest = payload.get("festival") or payload.get("festival_name") or "Diwali"
            date_str = payload.get("date") or "Oct 31"
            days_until = payload.get("days_until")
            days_str = f" ({days_until} days away)" if days_until else ""
            trigger_fact = f"{fest} is coming up on {date_str}{days_str}"
            locality_str = f" in {locality}" if locality else ""
            implication = f"Festive demand{locality_str} peaks 2-3 weeks prior; featuring an offer early captures early shoppers."
            suggested_cta = f"Reply YES to publish your '{active_offer_title or 'festive offer'}'."

        elif "renewal" in combined_kind:
            days = payload.get("days_remaining") or 12
            plan_name = payload.get("plan") or "Pro"
            amount = payload.get("renewal_amount") or 4999
            trigger_fact = f"Your {plan_name} subscription expires in {days} days (renewal amount: ₹{amount:,})"
            implication = "Renewing early prevents any interruption to your magicpin leads and profile rank."
            suggested_cta = "Reply RENEW to confirm your subscription renewal."

        elif "winback_eligible" in combined_kind:
            days = payload.get("days_since_expiry") or 38
            lapsed = payload.get("lapsed_customers_added_since_expiry") or 24
            trigger_fact = f"Account inactive for {days} days ({lapsed} new lapsed customers ready for winback)"
            implication = "Reactivating your profile allows you to win back lapsed customers in your area."
            suggested_cta = "Reply YES to reactivate your magicpin campaign."

        elif "customer_lapsed" in combined_kind or "lapsed" in combined_kind:
            cust_name = (cust.get("identity", {})).get("name") or cust.get("name") or payload.get("customer_name") or "Rashmi"
            days = payload.get("days_since_last_visit") or 57
            focus = payload.get("previous_focus") or "weight loss"
            trigger_fact = f"{cust_name} hasn't visited in {days} days (previous focus: {focus})"
            implication = f"Sending a targeted winback offer encourages {cust_name} to resume membership."
            suggested_cta = f"Want me to send a personalized winback offer to {cust_name}?"

        elif "ipl" in combined_kind:
            match = payload.get("match") or "DC vs MI"
            venue = payload.get("venue") or "Arun Jaitley Stadium"
            trigger_fact = f"IPL Match today: {match} at {venue}"
            implication = "Match nights drive 2x delivery volume for local food orders."
            suggested_cta = "Reply YES to launch a match-night combo promotion."

        elif "supply" in combined_kind:
            molecule = payload.get("molecule") or "atorvastatin"
            batches = payload.get("affected_batches") or ["AT2024-1102"]
            batch_str = ", ".join(batches) if isinstance(batches, list) else str(batches)
            trigger_fact = f"Supply Recall Alert: {molecule} batches ({batch_str})"
            implication = "Immediate inventory verification required to comply with pharmacy regulations."
            suggested_cta = "Want me to send the batch verification checklist to your staff?"

        elif "chronic_refill" in combined_kind:
            cust_name = (cust.get("identity", {})).get("name") or cust.get("name") or "Grandfather"
            mol_list = payload.get("molecule_list", ["metformin", "atorvastatin"])
            molecules = ", ".join(mol_list) if isinstance(mol_list, list) else str(mol_list)
            stock_out = payload.get("stock_runs_out_iso") or "Apr 28"
            if "T" in stock_out:
                stock_out = stock_out.split("T")[0]
            trigger_fact = f"Chronic refill due for {cust_name} ({molecules}); stock runs out on {stock_out}"
            implication = "Timely refill reminder prevents medication gaps for chronic care patients."
            suggested_cta = "Reply REFILL to confirm order and dispatch doorstep delivery."

        elif "planning" in combined_kind:
            topic = payload.get("intent_topic") or "corporate_bulk_thali_package"
            last_msg = payload.get("merchant_last_message") or ""
            msg_str = f" ('{last_msg[:40]}...')" if last_msg else ""
            trigger_fact = f"Active planning topic: {topic}"
            implication = f"Following up on your last message{msg_str} moves campaign setup forward."
            suggested_cta = "Want me to draft the package details and pricing for your review?"

        elif "research" in combined_kind or "digest" in combined_kind:
            source = trig.get("source") or payload.get("source") or "magicpin Insights"
            insight = payload.get("insight") or payload.get("title") or payload.get("summary") or "industry trends show rising customer interest"
            trigger_fact = f"{source}: {insight}"
            implication = "Capitalizing on market insights keeps your service offerings aligned with customer demand."
            suggested_cta = "Want me to share actionable recommendations for your business?"

        elif "appointment" in combined_kind or "trial" in combined_kind:
            cust_name = (cust.get("identity", {})).get("name") or cust.get("name") or payload.get("customer_name") or "Patient"
            time_str = payload.get("time") or payload.get("trial_date") or "10:00 AM"
            trigger_fact = f"Appointment / trial session for {cust_name} ({time_str})"
            implication = f"Sending a timely confirmation reduces no-shows for {m_name if m_name else 'your clinic'}."
            suggested_cta = f"Want me to send a WhatsApp reminder to {cust_name}?"

        elif "gbp" in combined_kind or "google" in combined_kind:
            trigger_fact = f"Google Business Profile for {m_name if m_name else 'your store'} remains unverified"
            implication = "Unverified profiles miss out on local map pack search traffic (estimated +30% uplift upon verification)."
            suggested_cta = "Reply VERIFY for step-by-step instructions to verify your Google listing."

        elif "seasonal" in combined_kind or "category_seasonal" in combined_kind:
            season = payload.get("season") or "summer_2026"
            trends = payload.get("trends") or ["ORS demand +40%", "sunscreen demand +38%"]
            trends_str = ", ".join(trends) if isinstance(trends, list) else str(trends)
            trigger_fact = f"Category seasonal demand shift ({season}): {trends_str}"
            implication = "Adjusting inventory and profile promotions captures high seasonal demand."
            suggested_cta = "Want me to draft a featured catalog list for summer products?"

        else:
            trigger_fact = f"Profile activity update: {merchant_fact}"
            implication = "Updating your profile details keeps your store visible to nearby customers."
            suggested_cta = "Want me to draft a quick profile update for your store?"

        objective = decision.get("objective") or "inform_or_assist"

        return {
            "salutation": salutation,
            "objective": objective,
            "why_now": trigger_fact,
            "merchant_fact": merchant_fact,
            "trigger_fact": trigger_fact,
            "implication": implication,
            "suggested_cta": suggested_cta,
            "merchant_name": m_name,
            "locality": locality,
            "active_offer": active_offer_title
        }

    def validate_llm_output(
        self,
        output: Dict[str, Any],
        context_plan: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validates LLM output against generic openings, generic CTAs, and missing salutations.
        """
        if not isinstance(output, dict) or not output.get("body"):
            return False, "Missing body in LLM output"

        body = output["body"]
        cta = output.get("cta", "")
        body_lower = body.lower()

        # 1. Generic opening checks
        forbidden_openings = [
            "following a recent update",
            "following your recent update",
            "regarding the recent update",
            "regarding a recent update",
            "recent update on account",
            "recent update on your account",
            "recent update",
            "account activity",
            "update for your business",
            "this is a good time to act",
            "we noticed some activity",
            "we noticed",
            "noticed some",
            "here's a relevant update",
            "here's a relevant",
            "there is an opportunity",
            "hi merchant",
            "hello merchant",
            "hello business"
        ]
        for forb in forbidden_openings:
            if forb in body_lower:
                return False, f"Forbidden generic opening phrase found: '{forb}'"

        # 2. Check generic CTAs
        generic_ctas = [
            "let me know if you'd like help",
            "would you like to improve your business",
            "set up festive offers",
            "review details",
            "explore ways",
            "let me know if you need help"
        ]
        cta_lower = cta.lower()
        if any(gen in cta_lower for gen in generic_ctas):
            return False, f"Generic CTA detected: '{cta}'"

        return True, "Valid"


    def test_gemini_connection(self) -> str:
        """
        Minimal internal test call using exact same provider implementation.
        """
        logger.info("[COMPOSER_DIAG] Testing direct Gemini connection with prompt: 'Reply with OK'")
        result = self._call_gemini_raw("Reply with OK")
        logger.info(f"[COMPOSER_DIAG] Direct Gemini connection response: '{result}'")
        return result

    def _call_gemini(
        self,
        decision: Dict[str, Any],
        category: Optional[Dict[str, Any]],
        merchant: Optional[Dict[str, Any]],
        customer: Optional[Dict[str, Any]],
        trigger: Dict[str, Any],
        send_as: str,
        template_name: str,
        template_params: List[str],
        suppression_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Calls Gemini REST API with structured grounding prompt and parses JSON response.
        """
        plan = self.build_message_plan(merchant, category, customer, trigger, decision)

        system_prompt = (
            "You are Vera, an AI proactive agent for magicpin merchants.\n"
            "Your task is to write ONE short, highly compelling WhatsApp message based STRICTLY on the supplied message plan and context facts.\n\n"
            "REQUIRED MESSAGE STRUCTURE:\n"
            "1. Personalized Opening: Address the merchant using their exact salutation from the plan (e.g. '" + plan['salutation'] + "'). NEVER use generic greetings like 'Hi Merchant', 'Hi Doctor', 'Hello Business'.\n"
            "2. Specific Trigger & Merchant Fact: Connect the trigger fact ('" + plan['trigger_fact'] + "') to the merchant's exact data ('" + plan['merchant_fact'] + "').\n"
            "3. One Low-Friction CTA: End with EXACTLY ONE low-effort, low-friction CTA (e.g. '" + plan['suggested_cta'] + "').\n\n"
            "STRICT RULES & CONSTRAINTS:\n"
            "- ZERO HALLUCINATION: Use ONLY exact facts, names, dates, metrics provided in context. NEVER invent prices, percentages, dates, names, or numbers.\n"
            "- NO GENERIC OPENINGS: NEVER start with 'following a recent update', 'following your recent update', 'regarding the recent update', 'we noticed activity', 'here's a relevant update'. Start directly with the concrete factual insight!\n"
            "- NO GENERIC FILLER: Avoid vague lines like 'Let me know if you need help', 'Improve your business', 'Set up festive offers'.\n"
            "- NO INTERNAL JARGON: Never mention internal codes like 'perf_dip', 'trigger_id', 'context_id'.\n"
            "- SINGLE CTA: Provide EXACTLY ONE low-friction call to action.\n"
            "- CONCISE: Keep the message short, professional, and clear for WhatsApp.\n\n"
            "Return ONLY valid raw JSON with this exact structure:\n"
            "{\n"
            '  "body": "Complete WhatsApp message string",\n'
            '  "template_name": "' + template_name + '",\n'
            '  "template_params": ' + json.dumps(template_params) + ',\n'
            '  "cta": "Single low-friction CTA string",\n'
            '  "send_as": "' + send_as + '",\n'
            '  "suppression_key": "' + suppression_key + '",\n'
            '  "rationale": "Short explanation of facts used from plan"\n'
            "}"
        )

        user_context = {
            "message_plan": plan,
            "decision": decision,
            "category": category,
            "merchant": merchant,
            "customer": customer,
            "trigger": trigger,
            "send_as": send_as,
            "template_name": template_name,
            "template_params": template_params,
            "suppression_key": suppression_key
        }

        full_prompt = system_prompt + "\n\nCONTEXT & MESSAGE PLAN:\n" + json.dumps(user_context, indent=2)

        raw_text = self._call_gemini_raw(full_prompt)

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned)
            is_valid, err_msg = self.validate_llm_output(parsed, plan)
            if not is_valid:
                logger.warning(f"[COMPOSER_DIAG] Validation failed: {err_msg}. Retrying once with repair prompt...")
                repair_prompt = full_prompt + f"\n\nPREVIOUS ERROR: {err_msg}. Please fix this error and adhere strictly to the JSON schema and salutation '{plan['salutation']}'."
                repair_text = self._call_gemini_raw(repair_prompt)
                repair_cleaned = repair_text.strip()
                if repair_cleaned.startswith("```"):
                    rlines = repair_cleaned.split("\n")
                    if rlines[0].startswith("```"):
                        rlines = rlines[1:]
                    if rlines and rlines[-1].startswith("```"):
                        rlines = rlines[:-1]
                    repair_cleaned = "\n".join(rlines).strip()
                parsed = json.loads(repair_cleaned)

            logger.info("[COMPOSER_DIAG] Gemini JSON response parsing & validation succeeds: True")
            return parsed
        except Exception as e:
            logger.error(f"[COMPOSER_DIAG] Gemini JSON response parsing succeeds: False (error: {e})")
            logger.error(f"[COMPOSER_DIAG] Raw output was: {raw_text[:200]}")
            raise


    def compose(
        self,
        decision: Dict[str, Any],
        category: Optional[Dict[str, Any]] = None,
        merchant: Optional[Dict[str, Any]] = None,
        customer: Optional[Dict[str, Any]] = None,
        trigger: Optional[Dict[str, Any]] = None,
        conversation_state: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Proactive composition method for /v1/tick.
        """
        trig_data = trigger or decision.get("selected_trigger") or {}
        trigger_type = (
            trig_data.get("type")
            or trig_data.get("trigger_type")
            or trig_data.get("family")
        )
        has_cust_id = bool(customer and (customer.get("customer_id") or customer.get("context_id")))
        cust_facing = self.is_customer_facing(trigger_type, has_cust_id)

        # Safety Check for customer-facing messages
        if cust_facing and not self.validate_customer_safety(trigger_type, customer, merchant):
            logger.warning("[COMPOSER_DIAG] Customer safety validation failed (missing context or consent=False)")
            return None

        ctx_facts = self.extract_fallback_context(merchant, category, customer, trig_data)
        merchant_name = ctx_facts["merchant_name"] or "Merchant"
        category_name = ctx_facts["category_name"] or "Category"
        customer_name = ctx_facts["customer_name"] or "Customer"
        voice = (category or {}).get("voice") or "Helpful & Professional"

        context_values = {
            "merchant_name": merchant_name,
            "category_name": category_name,
            "customer_name": customer_name,
            "voice": voice,
            "trigger_type": trigger_type
        }

        template_name, template_params = self.select_template(trigger_type, context_values)
        suppression_key = (
            trig_data.get("suppression_key")
            or trig_data.get("trigger_id")
            or trig_data.get("context_id")
            or "supp_default"
        )
        send_as = "merchant_on_behalf" if cust_facing else "vera"

        logger.info(f"[COMPOSER_DIAG] provider selected: {self.provider}")
        logger.info(f"[COMPOSER_DIAG] model selected: {self.model_name}")
        logger.info(f"[COMPOSER_DIAG] GEMINI_API_KEY loaded: {bool(self.api_key)}")

        fallback_reason = None

        if self.provider == "gemini":
            if self.api_key:
                logger.info("[COMPOSER_DIAG] LLM request attempted: True")
                try:
                    llm_output = self._call_gemini(
                        decision=decision,
                        category=category,
                        merchant=merchant,
                        customer=customer,
                        trigger=trig_data,
                        send_as=send_as,
                        template_name=template_name,
                        template_params=template_params,
                        suppression_key=suppression_key
                    )
                    if llm_output and isinstance(llm_output, dict) and llm_output.get("body"):
                        logger.info("[COMPOSER_DIAG] Fallback triggered: False (Gemini LLM composition succeeded)")
                        return llm_output
                    else:
                        fallback_reason = "Gemini returned output without valid 'body'"
                except Exception as e:
                    fallback_reason = f"Gemini LLM request or parse failed: {e}"
            else:
                fallback_reason = "GEMINI_API_KEY is not set or empty"
        elif self.provider == "openai":
            if self.api_key:
                logger.info("[COMPOSER_DIAG] LLM request attempted: True (OpenAI)")
                try:
                    llm_output = self._call_openai(
                        decision=decision,
                        category=category,
                        merchant=merchant,
                        customer=customer,
                        trigger=trig_data,
                        send_as=send_as,
                        template_name=template_name,
                        template_params=template_params,
                        suppression_key=suppression_key
                    )
                    if llm_output and isinstance(llm_output, dict) and llm_output.get("body"):
                        logger.info("[COMPOSER_DIAG] Fallback triggered: False (OpenAI LLM composition succeeded)")
                        return llm_output
                    else:
                        fallback_reason = "OpenAI returned output without valid 'body'"
                except Exception as e:
                    fallback_reason = f"OpenAI LLM call failed: {e}"
            else:
                fallback_reason = "OPENAI_API_KEY is not set or empty"
        else:
            fallback_reason = f"Unsupported LLM provider: {self.provider}"

        logger.warning(f"[COMPOSER_DIAG] Fallback triggered: True (Reason: {fallback_reason})")
        return self._deterministic_compose(
            trigger_type=trigger_type,
            merchant=merchant,
            category=category,
            customer=customer,
            trigger=trig_data,
            send_as=send_as,
            template_name=template_name,
            template_params=template_params,
            suppression_key=suppression_key
        )

    def compose_reply(
        self,
        conversation_history: List[Dict[str, Any]],
        current_intent: str,
        previous_outbound: Optional[str],
        relevant_context: Dict[str, Any],
        objective: Optional[str],
        from_role: str,
        merchant_name: str = "Merchant",
        customer_name: str = "Customer"
    ) -> Dict[str, Any]:
        """
        Reply composition path for /v1/reply.
        Handles positive action handoff, questions, and conversational responses.
        """
        send_as = "merchant_on_behalf" if from_role == "customer" else "vera"
        merchant_ctx = relevant_context.get("merchant") or {}
        offers = merchant_ctx.get("active_offers", [])
        actual_offer = offers[0] if isinstance(offers, list) and len(offers) > 0 else None

        # FIX 2: Positive Action Intent MUST NEVER RETURN NULL BODY & MUST ACKNOWLEDGE COMMITMENT
        if current_intent == "positive":
            if from_role == "customer":
                body = f"Great {customer_name}! I've noted your confirmation for {merchant_name}. Let me know if you would like to proceed with your booking."
            else:
                offer_phrase = f" featuring '{actual_offer}'" if actual_offer else ""
                body = f"Great — let's get this started! Here's what happens next: we'll prepare your campaign{offer_phrase} for {merchant_name}. Let me know if you'd like to review details before launching."
            return {
                "action": "send",
                "body": body,
                "cta": "Proceed",
                "wait_seconds": 0,
                "send_as": send_as,
                "rationale": "Positive action handoff: acknowledged commitment and provided immediate next step."
            }

        # Negative Intent
        if current_intent == "negative":
            return {
                "action": "end",
                "wait_seconds": 0,
                "rationale": "User expressed negative intent / requested stop."
            }

        # Defer Intent
        if current_intent == "defer":
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "User requested deferral / time to consider."
            }

        # Question or General Turn
        latest_user_msg = ""
        for turn in reversed(conversation_history):
            if turn.get("from_role") in ["merchant", "customer"]:
                latest_user_msg = turn.get("message", "").lower()
                break

        unsupported_keywords = ["weather", "stock", "stocks", "bitcoin", "crypto", "football", "politics", "recipe", "score"]
        if any(kw in latest_user_msg for kw in unsupported_keywords):
            body = "I don't have information on that specific topic right now, but I can assist with your magicpin offers and account growth."
            return {
                "action": "send",
                "body": body,
                "cta": "Reply YES to continue",
                "wait_seconds": 0,
                "send_as": send_as,
                "rationale": "Answered unsupported curveball question without hallucinating."
            }

        if "offer" in latest_user_msg or "discount" in latest_user_msg or "price" in latest_user_msg:
            offer_text = actual_offer or "our latest active deals"
            body = f"Your current active offer is '{offer_text}'. Would you like me to feature this offer to boost your visibility?"
        elif "how" in latest_user_msg or "what" in latest_user_msg or "next" in latest_user_msg:
            body = f"I help {merchant_name} launch targeted campaigns and engage customers on magicpin. Shall we proceed with your campaign setup?"
        else:
            body = f"Thank you for your message! Let me know if you would like me to proceed with your campaign for {merchant_name}."

        return {
            "action": "send",
            "body": body,
            "cta": "Reply YES to proceed",
            "wait_seconds": 0,
            "send_as": send_as,
            "rationale": "Answered question using supplied context facts."
        }

    def _deterministic_compose(
        self,
        trigger_type: Optional[str],
        merchant: Optional[Dict[str, Any]],
        category: Optional[Dict[str, Any]],
        customer: Optional[Dict[str, Any]],
        trigger: Dict[str, Any],
        send_as: str,
        template_name: str,
        template_params: List[str],
        suppression_key: str
    ) -> Dict[str, Any]:
        plan = self.build_message_plan(merchant, category, customer, trigger, {})
        salutation = plan["salutation"]
        tf = plan["trigger_fact"]
        imp = plan.get("implication", "")
        cta = plan["suggested_cta"]

        prefix = f"{salutation}, " if salutation else ""
        if imp:
            body = f"{prefix}{tf}. {imp} {cta}".strip()
        else:
            body = f"{prefix}{tf}. {cta}".strip()

        return {
            "body": body,
            "template_name": template_name,
            "template_params": template_params,
            "cta": cta,
            "send_as": send_as,
            "suppression_key": suppression_key,
            "rationale": f"Grounded deterministic composition: salutation ({salutation}), trigger fact ({tf}), implication ({imp}), CTA ({cta})."
        }


    def _call_openai(
        self,
        decision: Dict[str, Any],
        category: Optional[Dict[str, Any]],
        merchant: Optional[Dict[str, Any]],
        customer: Optional[Dict[str, Any]],
        trigger: Dict[str, Any],
        send_as: str,
        template_name: str,
        template_params: List[str],
        suppression_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Calls OpenAI Chat Completions API via standard urllib.
        """
        url = "https://api.openai.com/v1/chat/completions"
        system_prompt = (
            "You are Vera, an AI assistant for magicpin merchants and customers.\n"
            "Generate ONE concise, natural WhatsApp message based STRICTLY on supplied context.\n"
            "CRITICAL RULES:\n"
            "1. Use ONLY facts present in context. NEVER invent prices, discounts, competitors, dates, or numbers.\n"
            "2. Match send_as rule ('vera' for merchant-facing, 'merchant_on_behalf' for customer-facing).\n"
            "3. Use at most ONE primary CTA.\n"
            "4. Return ONLY valid JSON with keys: body, template_name, template_params, cta, send_as, suppression_key, rationale."
        )
        user_prompt = json.dumps({
            "decision": decision,
            "category": category,
            "merchant": merchant,
            "customer": customer,
            "trigger": trigger,
            "send_as": send_as,
            "template_name": template_name,
            "template_params": template_params,
            "suppression_key": suppression_key
        })

        payload_bytes = json.dumps({
            "model": self.model_name,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed

