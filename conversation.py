import threading
from typing import Dict, Any, List, Optional
from datetime import datetime

class ConversationStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._conversations: Dict[str, Dict[str, Any]] = {}

    def init_conversation(
        self,
        conversation_id: str,
        merchant_id: str,
        customer_id: Optional[str] = None,
        last_trigger_id: Optional[str] = None,
        objective: Optional[str] = None,
        suppression_key: Optional[str] = None,
        initial_message: Optional[str] = None,
        received_at: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Initializes a new proactive conversation state or updates existing metadata.
        """
        with self._lock:
            if conversation_id not in self._conversations:
                self._conversations[conversation_id] = {
                    "conversation_id": conversation_id,
                    "merchant_id": merchant_id,
                    "customer_id": customer_id,
                    "last_trigger_id": last_trigger_id,
                    "objective": objective,
                    "merchant_intent": None,
                    "auto_reply_detected": False,
                    "auto_reply_count": 0,
                    "ended": False,
                    "suppression_key": suppression_key,
                    "turn_number": 0,
                    "turns": [],
                    "created_at": datetime.utcnow().isoformat()
                }

            conv = self._conversations[conversation_id]
            if merchant_id:
                conv["merchant_id"] = merchant_id
            if customer_id is not None:
                conv["customer_id"] = customer_id
            if last_trigger_id:
                conv["last_trigger_id"] = last_trigger_id
            if objective:
                conv["objective"] = objective
            if suppression_key:
                conv["suppression_key"] = suppression_key

            if initial_message and len(conv["turns"]) == 0:
                conv["turn_number"] = 1
                conv["turns"].append({
                    "from_role": "system",
                    "message": initial_message,
                    "received_at": received_at,
                    "turn_number": 1,
                    "recorded_at": datetime.utcnow().isoformat()
                })

            return conv

    def add_turn(
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
        Stores conversation state and appends turn history.
        """
        with self._lock:
            if conversation_id not in self._conversations:
                self.init_conversation(
                    conversation_id=conversation_id,
                    merchant_id=merchant_id,
                    customer_id=customer_id
                )

            conv = self._conversations[conversation_id]
            conv["merchant_id"] = merchant_id
            if customer_id is not None:
                conv["customer_id"] = customer_id

            turn_entry = {
                "from_role": from_role,
                "message": message,
                "received_at": received_at,
                "turn_number": turn_number,
                "recorded_at": datetime.utcnow().isoformat()
            }
            conv["turns"].append(turn_entry)
            conv["turn_number"] = max(conv["turn_number"], turn_number)

            return conv

    def update_state(self, conversation_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if not conv:
                return None
            for key, val in kwargs.items():
                if key in conv:
                    conv[key] = val
            return conv

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._conversations.get(conversation_id)

    def get_total_conversations(self) -> int:
        with self._lock:
            return len(self._conversations)
