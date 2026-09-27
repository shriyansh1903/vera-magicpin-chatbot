import threading
from typing import Dict, Any, Tuple, Optional, List

class ContextStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._store: Dict[str, Dict[str, Dict[str, Any]]] = {
            "category": {},
            "merchant": {},
            "customer": {},
            "trigger": {}
        }

    def save_context(
        self,
        scope: str,
        context_id: str,
        version: int,
        payload: Dict[str, Any],
        delivered_at: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Thread-safe context saver with version handling.
        Returns a tuple of (status, active_version):
        - ("accepted", version) when inserted or updated with higher version
        - ("noop", existing_version) when exact (context_id, version) already exists
        - ("stale", existing_version) when provided version is lower than existing version
        """
        with self._lock:
            if scope not in self._store:
                self._store[scope] = {}

            existing = self._store[scope].get(context_id)
            if existing is not None:
                existing_version = existing["version"]
                if version == existing_version:
                    return "noop", existing_version
                elif version < existing_version:
                    return "stale", existing_version

            # Store or replace with newer version
            self._store[scope][context_id] = {
                "context_id": context_id,
                "version": version,
                "payload": payload,
                "delivered_at": delivered_at
            }
            return "accepted", version

    def _format_context(self, item: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not item:
            return None
        res = dict(item.get("payload", {}))
        res["context_id"] = item["context_id"]
        res["version"] = item["version"]
        if item.get("delivered_at"):
            res["delivered_at"] = item["delivered_at"]
        return res

    def get_context(self, scope: str, context_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._store.get(scope, {}).get(context_id)
            return self._format_context(item)

    def get_category(self, category_id: str) -> Optional[Dict[str, Any]]:
        return self.get_context("category", category_id)

    def get_merchant(self, merchant_id: str) -> Optional[Dict[str, Any]]:
        return self.get_context("merchant", merchant_id)

    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        return self.get_context("customer", customer_id)

    def get_trigger(self, trigger_id: str) -> Optional[Dict[str, Any]]:
        return self.get_context("trigger", trigger_id)

    def get_active_triggers(self, trigger_ids: List[str]) -> List[Dict[str, Any]]:
        with self._lock:
            active = []
            for tid in trigger_ids:
                trig = self.get_trigger(tid)
                if trig:
                    active.append(trig)
            return active

    def get_all_context_for_trigger(self, trigger_id: str) -> Dict[str, Any]:
        trig = self.get_trigger(trigger_id)
        if not trig:
            return {}

        merchant_id = trig.get("merchant_id") or (trig.get("merchant", {}).get("merchant_id") if isinstance(trig.get("merchant"), dict) else None)
        customer_id = trig.get("customer_id") or (trig.get("customer", {}).get("customer_id") if isinstance(trig.get("customer"), dict) else None)
        category_id = trig.get("category_id") or (trig.get("category", {}).get("category_id") if isinstance(trig.get("category"), dict) else None)

        merchant = self.get_merchant(merchant_id) if merchant_id else None

        if not category_id and merchant:
            category_id = merchant.get("category_id") or merchant.get("category_slug")

        category = self.get_category(category_id) if category_id else None
        if not category and category_id:
            with self._lock:
                for cid, citem in self._store.get("category", {}).items():
                    cpayload = citem.get("payload", {})
                    if cid == category_id or cpayload.get("slug") == category_id or cpayload.get("category_id") == category_id:
                        category = self._format_context(citem)
                        break

        customer = self.get_customer(customer_id) if customer_id else None


        return {
            "category": category,
            "merchant": merchant,
            "customer": customer,
            "trigger": trig
        }

    def get_counts(self) -> Dict[str, int]:
        with self._lock:
            return {
                "category": len(self._store.get("category", {})),
                "merchant": len(self._store.get("merchant", {})),
                "customer": len(self._store.get("customer", {})),
                "trigger": len(self._store.get("trigger", {}))
            }
