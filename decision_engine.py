import threading
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from pydantic import BaseModel, Field

from context_store import ContextStore
from conversation import ConversationStore
from composer import LLMComposer

# Objective mapping for known trigger families
OBJECTIVE_MAPPING: Dict[str, str] = {
    "recall_due": "drive_booking_or_action",
    "appointment_tomorrow": "drive_booking_or_action",
    "customer_lapsed_soft": "reengage_lapsed_customer",
    "customer_lapsed_hard": "winback_lapsed_customer",
    "perf_dip": "diagnose_and_improve_performance",
    "perf_spike": "capitalize_on_momentum",
    "milestone_reached": "reinforce_achievement_and_next_action",
    "review_theme_emerged": "address_review_theme",
    "research_digest": "share_relevant_insight",
    "research_digest_release": "share_relevant_insight",
    "competitor_opened": "surface_competitive_signal_and_action",
    "festival": "create_timely_merchant_opportunity",
    "festival_upcoming": "create_timely_merchant_opportunity",
    "category_trend_movement": "connect_demand_signal_to_merchant",
    "curious_ask_due": "generate_useful_curiosity_conversation",
    "scheduled_recurring": "inform_or_assist"
}

# Action-oriented trigger families (prioritized over informational)
ACTION_TRIGGER_FAMILIES: Set[str] = {
    "recall_due",
    "appointment_tomorrow",
    "customer_lapsed_soft",
    "customer_lapsed_hard",
    "perf_dip",
    "perf_spike",
    "milestone_reached",
    "review_theme_emerged",
    "competitor_opened",
    "festival",
    "festival_upcoming",
    "curious_ask_due"
}

class SuppressionRegistry:
    """
    In-process suppression registry. Marks (suppression_key, version) tuples as used.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._suppressed: Set[Tuple[str, int]] = set()

    def suppress(self, suppression_key: str, version: int) -> None:
        with self._lock:
            self._suppressed.add((suppression_key, version))

    def is_suppressed(self, suppression_key: str, version: int) -> bool:
        with self._lock:
            return (suppression_key, version) in self._suppressed

class Decision(BaseModel):
    selected_trigger: Optional[Dict[str, Any]] = None
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    objective: str = "inform_or_assist"
    priority: int = 0
    relevant_context: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

class DecisionEngine:
    def __init__(self, context_store: ContextStore, conversation_store: ConversationStore, composer: Optional[LLMComposer] = None):
        self.context_store = context_store
        self.conversation_store = conversation_store
        self.composer = composer or LLMComposer()
        self.suppression_registry = SuppressionRegistry()

    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        if not dt_str:
            return None
        try:
            clean_str = dt_str.replace("Z", "+00:00")
            return datetime.fromisoformat(clean_str)
        except Exception:
            return None

    def _get_urgency_score(self, urgency: Any) -> int:
        if isinstance(urgency, (int, float)):
            return int(urgency)
        if isinstance(urgency, str):
            mapping = {
                "critical": 50,
                "high": 40,
                "medium": 30,
                "low": 20,
                "info": 10
            }
            return mapping.get(urgency.lower(), 0)
        return 0

    def map_objective(self, trigger_type: Optional[str]) -> str:
        if not trigger_type:
            return "inform_or_assist"
        return OBJECTIVE_MAPPING.get(trigger_type.lower(), "inform_or_assist")

    def build_relevant_context(self, all_ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds a compact dictionary containing only prioritized, useful context fields.
        """
        rel: Dict[str, Any] = {}

        merchant = all_ctx.get("merchant")
        if merchant:
            identity = merchant.get("identity", {}) if isinstance(merchant.get("identity"), dict) else {}
            rel["merchant"] = {
                "merchant_id": merchant.get("merchant_id") or merchant.get("context_id"),
                "name": identity.get("name") or merchant.get("name") or merchant.get("merchant_name"),
                "owner_first_name": identity.get("owner_first_name") or merchant.get("owner_first_name"),
                "locality": identity.get("locality") or merchant.get("locality"),
                "city": identity.get("city") or merchant.get("city"),
                "category_slug": merchant.get("category_slug") or merchant.get("category_id"),
                "identity": identity,
                "offers": merchant.get("offers", merchant.get("active_offers", [])),
                "active_offers": merchant.get("active_offers", merchant.get("offers", [])),
                "performance": merchant.get("performance", merchant.get("performance_metrics", {})),
                "performance_metrics": merchant.get("performance_metrics", merchant.get("performance", {})),
                "signals": merchant.get("signals", []),
                "review_themes": merchant.get("review_themes", []),
                "customer_aggregate": merchant.get("customer_aggregate", {})
            }

        category = all_ctx.get("category")
        if category:
            rel["category"] = {
                "category_id": category.get("category_id") or category.get("context_id") or category.get("slug"),
                "slug": category.get("slug") or category.get("category_id"),
                "name": category.get("display_name") or category.get("name") or category.get("category_name"),
                "voice": category.get("voice") or category.get("brand_voice"),
                "peer_stats": category.get("peer_stats", {}),
                "digest": category.get("digest", []),
                "trends": category.get("trends", category.get("trend_signals", []))
            }

        customer = all_ctx.get("customer")
        if customer:
            cust_identity = customer.get("identity", {}) if isinstance(customer.get("identity"), dict) else {}
            rel["customer"] = {
                "customer_id": customer.get("customer_id") or customer.get("context_id"),
                "name": cust_identity.get("name") or customer.get("name") or customer.get("customer_name"),
                "identity": cust_identity,
                "preferences": customer.get("preferences", {}),
                "consent": customer.get("consent", True),
                "state": customer.get("state")
            }


        trigger = all_ctx.get("trigger")
        if trigger:
            rel["trigger"] = {
                "trigger_id": trigger.get("trigger_id") or trigger.get("context_id"),
                "type": trigger.get("type") or trigger.get("trigger_type") or trigger.get("family"),
                "urgency": trigger.get("urgency"),
                "source": trigger.get("source"),
                "payload": trigger.get("payload", {})
            }

        return rel

    def rank_triggers(
        self,
        now: Optional[str],
        available_triggers: List[Any]
    ) -> List[Tuple[Tuple[int, int, int, str], Dict[str, Any]]]:
        now_dt = self._parse_datetime(now)
        candidates = []

        for item in available_triggers:
            trigger_data = None
            if isinstance(item, str):
                trigger_data = self.context_store.get_trigger(item)
            elif isinstance(item, dict):
                tid = item.get("context_id") or item.get("trigger_id") or item.get("id")
                if tid:
                    trigger_data = self.context_store.get_trigger(tid)
                if not trigger_data:
                    trigger_data = item

            if not trigger_data:
                continue

            trigger_id = (
                trigger_data.get("trigger_id")
                or trigger_data.get("context_id")
                or trigger_data.get("id")
            )
            if not trigger_id:
                continue

            expires_at_str = trigger_data.get("expires_at")
            if expires_at_str and now_dt:
                expires_at_dt = self._parse_datetime(expires_at_str)
                if expires_at_dt and now_dt >= expires_at_dt:
                    continue

            suppression_key = trigger_data.get("suppression_key") or trigger_id
            version = int(trigger_data.get("version", 1))
            if self.suppression_registry.is_suppressed(suppression_key, version):
                continue

            urgency_score = self._get_urgency_score(trigger_data.get("urgency"))

            trigger_type = (
                trigger_data.get("type")
                or trigger_data.get("trigger_type")
                or trigger_data.get("family")
                or ""
            )
            
            is_action_flag = (
                trigger_data.get("is_action_required", False)
                or trigger_data.get("is_customer_important", False)
                or (trigger_type.lower() in ACTION_TRIGGER_FAMILIES)
            )
            action_score = 1 if is_action_flag else 0

            priority_tuple = (urgency_score, action_score, version, str(trigger_id))
            candidates.append((priority_tuple, trigger_data))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates

    def make_decision(
        self,
        now: Optional[str],
        available_triggers: List[Any]
    ) -> Decision:
        ranked = self.rank_triggers(now, available_triggers)
        if not ranked:
            return Decision(reason="No valid, unexpired, unsuppressed triggers available")

        best_priority, best_trigger = ranked[0]
        trigger_id = (
            best_trigger.get("trigger_id")
            or best_trigger.get("context_id")
            or best_trigger.get("id")
        )

        suppression_key = best_trigger.get("suppression_key") or trigger_id
        version = int(best_trigger.get("version", 1))

        self.suppression_registry.suppress(suppression_key, version)

        all_ctx = self.context_store.get_all_context_for_trigger(trigger_id)
        if not all_ctx.get("trigger"):
            all_ctx["trigger"] = best_trigger

        rel_ctx = self.build_relevant_context(all_ctx)

        trigger_type = (
            best_trigger.get("type")
            or best_trigger.get("trigger_type")
            or best_trigger.get("family")
        )
        objective = self.map_objective(trigger_type)

        merchant_id = (
            best_trigger.get("merchant_id")
            or (rel_ctx.get("merchant") or {}).get("merchant_id")
        )
        customer_id = (
            best_trigger.get("customer_id")
            or (rel_ctx.get("customer") or {}).get("customer_id")
        )

        priority_value = best_priority[0]

        return Decision(
            selected_trigger=best_trigger,
            merchant_id=merchant_id,
            customer_id=customer_id,
            objective=objective,
            priority=priority_value,
            relevant_context=rel_ctx,
            reason=f"Selected trigger {trigger_id} (urgency: {priority_value}, objective: {objective})"
        )

    def process_tick(self, now: Optional[str], available_triggers: List[Any]) -> Dict[str, Any]:
        decision = self.make_decision(now, available_triggers)
        if not decision.selected_trigger:
            return {"actions": []}

        action_entry = {
            "action_type": "decision",
            "trigger_id": (
                decision.selected_trigger.get("trigger_id")
                or decision.selected_trigger.get("context_id")
            ),
            "merchant_id": decision.merchant_id,
            "customer_id": decision.customer_id,
            "objective": decision.objective,
            "priority": decision.priority,
            "relevant_context": decision.relevant_context,
            "reason": decision.reason
        }
        return {"actions": [action_entry]}

    def detect_intent(self, message: str) -> str:
        """
        Deterministic intent detection prioritizing positive commitment over general question checks.
        """
        msg = message.strip().lower()

        # 1. Negative / Stop Intent
        if re.search(r'\bno\b', msg) or any(phrase in msg for phrase in [
            "not interested", "don't want", "dont want", "stop",
            "don't send", "dont send", "unsubscribe", "never", "cancel", "dont publish"
        ]):
            return "negative"

        # 2. Positive / Action Handoff Intent (Priority check before question)
        positive_phrases = [
            "i want to join", "lets do it", "let's do it", "go ahead", "do it",
            "publish it", "send it", "count me in", "yes", "yep", "yeah", "sure", "affirmative"
        ]
        for phrase in positive_phrases:
            if re.search(r'\b' + re.escape(phrase) + r'\b', msg) or phrase in msg:
                return "positive"

        # 3. Question / Info Request
        if "?" in msg or any(w in msg for w in ["what", "how", "why", "when", "where", "which", "price", "cost", "details", "forecast"]):
            return "question"

        # 4. Defer / Wait Intent
        defer_phrases = [
            "later", "give me time", "not now", "busy",
            "remind me later", "hold on", "wait", "remind me tomorrow", "call me tomorrow"
        ]
        for phrase in defer_phrases:
            if phrase in msg:
                return "defer"

        if msg == "tomorrow":
            return "defer"

        # 5. Generic Acknowledgement
        if msg in ["thanks", "thank you", "got it", "noted", "thx", "k"]:
            return "acknowledgement"

        return "general"

    def detect_auto_reply(self, message: str, turns: List[Dict[str, Any]]) -> bool:
        """
        Detects canned auto-responses or repeated messages across turns.
        """
        msg = message.strip().lower()
        canned_patterns = [
            "automated message",
            "thank you for reaching out",
            "currently away",
            "out of office",
            "auto response",
            "will get back to you",
            "automatic reply"
        ]
        if any(pat in msg for pat in canned_patterns):
            return True

        user_msgs = [t.get("message", "").strip().lower() for t in turns if t.get("from_role") in ["merchant", "customer"]]
        if len(user_msgs) >= 2 and user_msgs[-1] == user_msgs[-2]:
            return True

        return False

    def process_reply(
        self,
        conversation_id: str,
        merchant_id: str,
        customer_id: Optional[str],
        from_role: str,
        message: str,
        received_at: Optional[str],
        turn_number: int
    ) -> Dict[str, Any]:
        """
        Full conversation reply handler implementing intent detection, auto-reply handling,
        negative/defer routing, positive action handoff, and LLM reply composition.
        """
        conv = self.conversation_store.get_conversation(conversation_id)
        if not conv:
            conv = self.conversation_store.init_conversation(
                conversation_id=conversation_id,
                merchant_id=merchant_id,
                customer_id=customer_id
            )

        conv = self.conversation_store.add_turn(
            conversation_id=conversation_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            from_role=from_role,
            message=message,
            received_at=received_at,
            turn_number=turn_number
        )

        if conv.get("ended"):
            return {
                "action": "end",
                "wait_seconds": 0,
                "rationale": "Conversation already ended"
            }

        turns = conv.get("turns", [])

        if self.detect_auto_reply(message, turns):
            conv["auto_reply_count"] += 1
            conv["auto_reply_detected"] = True

            if conv["auto_reply_count"] >= 2:
                conv["ended"] = True
                self.conversation_store.update_state(conversation_id, ended=True)
                return {
                    "action": "end",
                    "wait_seconds": 0,
                    "rationale": "Repeated auto-reply detected: ending conversation gracefully"
                }
            else:
                clarification_msg = "Hi! I am Vera, magicpin's AI assistant. Let me know if you would like me to help with your magicpin offers."
                self.conversation_store.add_turn(
                    conversation_id=conversation_id,
                    merchant_id=merchant_id,
                    customer_id=customer_id,
                    from_role="system",
                    message=clarification_msg,
                    received_at=received_at,
                    turn_number=turn_number + 1
                )
                return {
                    "action": "send",
                    "body": clarification_msg,
                    "cta": "Reply YES to continue",
                    "wait_seconds": 0,
                    "send_as": "vera",
                    "rationale": "First suspected auto-reply: sent short clarification attempt"
                }

        intent = self.detect_intent(message)
        conv["merchant_intent"] = intent
        self.conversation_store.update_state(conversation_id, merchant_intent=intent)

        if intent == "negative":
            conv["ended"] = True
            self.conversation_store.update_state(conversation_id, ended=True)
            return {
                "action": "end",
                "wait_seconds": 0,
                "rationale": "User expressed negative intent / requested stop"
            }

        if intent == "defer":
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "User requested deferral / time to consider"
            }

        merchant_ctx = self.context_store.get_merchant(merchant_id) or {}
        customer_ctx = self.context_store.get_customer(customer_id) if customer_id else None
        
        last_trig_id = conv.get("last_trigger_id")
        trig_ctx = self.context_store.get_trigger(last_trig_id) if last_trig_id else None

        all_ctx = {
            "merchant": merchant_ctx,
            "customer": customer_ctx,
            "trigger": trig_ctx
        }
        rel_ctx = self.build_relevant_context(all_ctx)

        previous_outbound = None
        for t in reversed(turns[:-1]):
            if t.get("from_role") == "system":
                previous_outbound = t.get("message")
                break

        merchant_name = merchant_ctx.get("name") or merchant_ctx.get("merchant_name") or "Merchant"
        customer_name = (customer_ctx or {}).get("name") or "Customer"

        composed_reply = self.composer.compose_reply(
            conversation_history=turns,
            current_intent=intent,
            previous_outbound=previous_outbound,
            relevant_context=rel_ctx,
            objective=conv.get("objective"),
            from_role=from_role,
            merchant_name=merchant_name,
            customer_name=customer_name
        )

        reply_action = composed_reply.get("action", "send")
        if reply_action == "end":
            conv["ended"] = True
            self.conversation_store.update_state(conversation_id, ended=True)
            return {
                "action": "end",
                "wait_seconds": 0,
                "rationale": composed_reply.get("rationale", "Conversation ended")
            }

        if reply_action == "wait":
            return {
                "action": "wait",
                "wait_seconds": composed_reply.get("wait_seconds", 86400),
                "rationale": composed_reply.get("rationale", "Waiting for user action")
            }

        outbound_body = composed_reply.get("body", "")
        if outbound_body:
            self.conversation_store.add_turn(
                conversation_id=conversation_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                from_role="system",
                message=outbound_body,
                received_at=received_at,
                turn_number=turn_number + 1
            )

        return {
            "action": "send",
            "body": outbound_body,
            "cta": composed_reply.get("cta", ""),
            "send_as": composed_reply.get("send_as", "vera"),
            "wait_seconds": 0,
            "rationale": composed_reply.get("rationale", "Sent conversational reply")
        }
