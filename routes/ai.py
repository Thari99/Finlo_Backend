import asyncio
import json
import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import CLAUDE_TIMEOUT_S
from services import agent, cost_guard, rate_limit
from services.auth import verify_access
from services.logging_setup import logger

router = APIRouter(prefix="/api/ai", tags=["AI"])

KEEPALIVE_INTERVAL_S = 15


class Message(BaseModel):
    role: str
    content: str


# ── Structured financial snapshot ────────────────────────────────────────────

class Account(BaseModel):
    name: str
    type: str
    balance: float
    currency: str = ""


class Transaction(BaseModel):
    account_id: str = ""
    type: str  # "income" | "expense"
    amount: float
    category: str = ""
    merchant: str = ""
    date: str  # ISO8601


class Bill(BaseModel):
    name: str
    amount: float
    due_date: str  # ISO8601
    recurring: str = ""


class Debt(BaseModel):
    lender_name: str
    total_amount: float
    paid_amount: float
    due_date: Optional[str] = None
    status: str = "active"


class Lending(BaseModel):
    person_name: str
    amount: float
    returned_amount: float
    due_date: Optional[str] = None
    status: str = "active"


class Budget(BaseModel):
    category: str
    limit_amount: float


class FinancialSnapshot(BaseModel):
    default_currency: str = "USD"
    accounts: list[Account] = Field(default_factory=list)
    transactions: list[Transaction] = Field(default_factory=list)
    bills: list[Bill] = Field(default_factory=list)
    debts: list[Debt] = Field(default_factory=list)
    lendings: list[Lending] = Field(default_factory=list)
    budgets: list[Budget] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list)
    snapshot: FinancialSnapshot = Field(default_factory=FinancialSnapshot)


# ── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(
    body: ChatRequest,
    user: dict = Depends(verify_access),
):
    user_id = user["user_id"]
    is_premium = user["is_premium"]

    if not rate_limit.can_send(user_id, is_premium):
        remaining = rate_limit.get_remaining(user_id, is_premium)
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit reached. {remaining} messages left today.",
        )

    history = [{"role": m.role, "content": m.content} for m in body.history]
    snapshot_dict = body.snapshot.model_dump()

    # Cost guard is char-based; estimate from message + history + snapshot size.
    input_chars = (
        sum(len(m["content"]) for m in history)
        + len(body.message)
        + len(json.dumps(snapshot_dict))
    )
    cost_guard.preflight(user_id, input_chars)
    rate_limit.increment(user_id)

    started = time.monotonic()
    logger.info(
        "chat.start",
        user_id=user_id,
        input_chars=input_chars,
        history_turns=len(history),
        accounts=len(snapshot_dict.get("accounts", [])),
        transactions=len(snapshot_dict.get("transactions", [])),
        is_premium=is_premium,
    )

    async def event_stream():
        output_chars = 0
        last_yield = time.monotonic()

        async def chunk_iterator():
            nonlocal output_chars
            try:
                gen = agent.stream_chat(snapshot_dict, history, body.message)
                async for chunk in _with_timeout(gen, CLAUDE_TIMEOUT_S + 30):
                    output_chars += len(chunk)
                    yield f"data: {json.dumps({'text': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            except asyncio.TimeoutError:
                logger.warning("chat.timeout", user_id=user_id)
                yield f"data: {json.dumps({'error': 'Response timed out — please try again.'})}\n\n"
            except Exception as e:  # noqa: BLE001
                logger.exception("chat.error", user_id=user_id, error=str(e))
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        chunks = chunk_iterator()
        try:
            while True:
                try:
                    next_chunk = await asyncio.wait_for(
                        chunks.__anext__(),
                        timeout=KEEPALIVE_INTERVAL_S,
                    )
                    last_yield = time.monotonic()
                    yield next_chunk
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    if time.monotonic() - last_yield > CLAUDE_TIMEOUT_S + 30:
                        logger.warning("chat.stalled", user_id=user_id)
                        break
        finally:
            try:
                running_total = cost_guard.record_usage(
                    user_id, input_chars, output_chars
                )
                logger.info(
                    "chat.end",
                    user_id=user_id,
                    output_chars=output_chars,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    monthly_usd=round(running_total, 4),
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("chat.cost_record_failed", user_id=user_id, error=str(e))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _with_timeout(gen, total_timeout_s: int):
    async def _step():
        return await gen.__anext__()

    deadline = asyncio.get_event_loop().time() + total_timeout_s
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        try:
            yield await asyncio.wait_for(_step(), timeout=remaining)
        except StopAsyncIteration:
            return


@router.get("/usage")
async def usage(user: dict = Depends(verify_access)):
    return {
        "user_id": user["user_id"],
        "is_premium": user["is_premium"],
        "remaining_today": rate_limit.get_remaining(user["user_id"], user["is_premium"]),
        "monthly_usd_spent": round(cost_guard.get_spent_usd(user["user_id"]), 4),
    }
