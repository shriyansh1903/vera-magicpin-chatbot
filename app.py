import time
import uuid
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import settings
from models import (
    ContextRequest,
    ContextResponse,
    TickRequest,
    TickResponse,
    ReplyRequest,
    ReplyResponse,
    HealthzResponse,
    MetadataResponse,
)
from context_store import ContextStore
from conversation import ConversationStore
from decision_engine import DecisionEngine
from composer import LLMComposer
from validator import OutputValidator

app = FastAPI(
    title="Vera AI Challenge Bot",
    version=settings.VERSION,
    description="FastAPI service for Magicpin Vera AI challenge bot"
)

# Record application start time for healthz uptime calculation
START_TIME = time.time()

# Instantiate core architecture components
context_store = ContextStore()
conversation_store = ConversationStore()
composer = LLMComposer(model_name=settings.MODEL)
decision_engine = DecisionEngine(context_store, conversation_store, composer)
validator = OutputValidator()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Override default FastAPI 422 error response to return 400 for invalid inputs as required.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "invalid_input",
            "detail": exc.errors(),
            "message": "Invalid request payload format"
        }
    )

@app.post("/v1/context", response_model=ContextResponse)
async def ingest_context(req: ContextRequest):
    """
    Ingest category, merchant, customer, or trigger contexts.
    - Idempotent no-op for matching (context_id, version): returns accepted=True.
    - Higher version replaces older version: returns accepted=True.
    - Lower version returns 409 conflict with stale_version.
    - Invalid input returns 400 bad request.
    """
    store_status, current_version = context_store.save_context(
        scope=req.scope,
        context_id=req.context_id,
        version=req.version,
        payload=req.payload,
        delivered_at=req.delivered_at
    )

    if store_status == "stale":
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": "stale_version",
                "detail": "stale_version",
                "message": f"Provided version {req.version} is lower than current stored version {current_version}",
                "current_version": current_version,
                "provided_version": req.version
            }
        )

    return ContextResponse(
        accepted=True,
        status="accepted",
        scope=req.scope,
        context_id=req.context_id,
        version=req.version
    )

@app.post("/v1/tick", response_model=TickResponse)
async def tick(req: TickRequest):
    """
    Periodic evaluation endpoint for proactive actions.
    Flow: available_triggers -> Decision Engine -> Composer -> Action -> Record Conversation State -> Trigger Suppression
    """
    # 1. Ask Decision Engine for best decision
    decision_obj = decision_engine.make_decision(now=req.now, available_triggers=req.available_triggers)
    if not decision_obj or not decision_obj.selected_trigger:
        return TickResponse(actions=[])

    selected_trigger = decision_obj.selected_trigger
    trigger_id = (
        selected_trigger.get("trigger_id")
        or selected_trigger.get("context_id")
        or selected_trigger.get("id")
    )
    trigger_type = (
        selected_trigger.get("type")
        or selected_trigger.get("trigger_type")
        or selected_trigger.get("family")
    )

    # 2. Extract context items
    rel_ctx = decision_obj.relevant_context or {}
    merchant_ctx = rel_ctx.get("merchant")
    category_ctx = rel_ctx.get("category")
    customer_ctx = rel_ctx.get("customer")
    trigger_ctx = rel_ctx.get("trigger") or selected_trigger

    # 3. Safety validation for customer-facing messages
    has_cust = bool(customer_ctx and (customer_ctx.get("customer_id") or customer_ctx.get("context_id")))
    cust_facing = composer.is_customer_facing(trigger_type, has_cust)
    if cust_facing and not composer.validate_customer_safety(trigger_type, customer_ctx, merchant_ctx):
        return TickResponse(actions=[])

    # 4. Compose WhatsApp message via Composer
    decision_dict = decision_obj.model_dump()
    composed_msg = composer.compose(
        decision=decision_dict,
        category=category_ctx,
        merchant=merchant_ctx,
        customer=customer_ctx,
        trigger=trigger_ctx
    )

    if not composed_msg or not isinstance(composed_msg, dict) or not composed_msg.get("body"):
        return TickResponse(actions=[])

    # 5. Generate unique conversation_id for new proactive conversation
    conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
    merchant_id = (
        decision_obj.merchant_id
        or (merchant_ctx or {}).get("merchant_id")
        or (merchant_ctx or {}).get("context_id")
    )
    
    # Merchant-facing action MUST have customer_id = None; Customer-facing action MUST have actual customer_id
    if cust_facing:
        customer_id = (
            decision_obj.customer_id
            or (customer_ctx or {}).get("customer_id")
            or (customer_ctx or {}).get("context_id")
        )
    else:
        customer_id = None

    suppression_key = (
        composed_msg.get("suppression_key")
        or selected_trigger.get("suppression_key")
        or trigger_id
    )

    # 6. Initialize and record conversation state
    conversation_store.init_conversation(
        conversation_id=conversation_id,
        merchant_id=merchant_id,
        customer_id=customer_id,
        last_trigger_id=trigger_id,
        objective=decision_obj.objective,
        suppression_key=suppression_key,
        initial_message=composed_msg["body"],
        received_at=req.now
    )

    # 7. Construct challenge-compatible action object
    action_entry = {
        "conversation_id": conversation_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": composed_msg.get("send_as", "vera"),
        "trigger_id": trigger_id,
        "template_name": composed_msg.get("template_name", "vera_proactive_notification_v1"),
        "template_params": composed_msg.get("template_params", []),
        "body": composed_msg.get("body", ""),
        "cta": composed_msg.get("cta", ""),
        "suppression_key": suppression_key,
        "rationale": composed_msg.get("rationale", "")
    }

    return TickResponse(actions=[action_entry])

@app.post("/v1/reply", response_model=ReplyResponse)
async def reply(req: ReplyRequest):
    """
    Receive customer or merchant messages and process replies via Decision Engine & Conversation Store.
    """
    res = decision_engine.process_reply(
        conversation_id=req.conversation_id,
        merchant_id=req.merchant_id,
        customer_id=req.customer_id,
        from_role=req.from_role,
        message=req.message,
        received_at=req.received_at,
        turn_number=req.turn_number
    )

    return ReplyResponse(
        action=res.get("action", "send"),
        wait_seconds=res.get("wait_seconds", 0),
        rationale=res.get("rationale", "Conversation handling completed"),
        body=res.get("body"),
        cta=res.get("cta"),
        send_as=res.get("send_as")
    )

@app.get("/v1/healthz", response_model=HealthzResponse)
async def healthz():
    """
    Health check endpoint returning system status, uptime, and context counts.
    """
    uptime = round(time.time() - START_TIME, 2)
    counts = context_store.get_counts()
    return HealthzResponse(
        status="ok",
        uptime_seconds=uptime,
        contexts=counts
    )

@app.get("/v1/metadata", response_model=MetadataResponse)
async def metadata():
    """
    Metadata endpoint returning team information, model, and version details.
    """
    return MetadataResponse(
        team_name=settings.TEAM_NAME,
        team_members=settings.TEAM_MEMBERS,
        model=settings.MODEL,
        approach=settings.APPROACH,
        contact_email=settings.CONTACT_EMAIL,
        version=settings.VERSION,
        submitted_at=settings.SUBMITTED_AT
    )
