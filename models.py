from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

class ContextRequest(BaseModel):
    scope: Literal["category", "merchant", "customer", "trigger"]
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str] = None

class ContextResponse(BaseModel):
    accepted: bool = True
    status: str
    scope: str
    context_id: str
    version: int

class TickRequest(BaseModel):
    now: Optional[str] = None
    available_triggers: List[Any] = Field(default_factory=list)

class TickResponse(BaseModel):
    actions: List[Dict[str, Any]] = Field(default_factory=list)

class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: int

class ReplyResponse(BaseModel):
    action: str = "wait"
    wait_seconds: int = 0
    rationale: str = "Conversation handling completed"
    body: Optional[str] = None
    cta: Optional[str] = None
    send_as: Optional[str] = None

class HealthzResponse(BaseModel):
    status: str = "ok"
    uptime_seconds: float
    contexts: Dict[str, int]

class MetadataResponse(BaseModel):
    team_name: str
    team_members: List[str]
    model: str
    approach: str
    contact_email: str
    version: str
    submitted_at: str
