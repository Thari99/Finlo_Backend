"""
Per-user monthly $ budget for AI calls.

We charge users (in Redis) based on a char→token estimate after each request.
Before every request we check whether the *worst-case* charge (input chars +
CLAUDE_MAX_TOKENS output) would push them past MONTHLY_USD_CAP. If yes, we
reject with 402.

Redis key: finlo:cost:YYYY-MM:<user_id>  → float USD spent this month
Auto-expires 90 days after first write (long enough to handle billing
reconciliation, short enough not to leak storage).
"""
from datetime import datetime, timezone
from fastapi import HTTPException

from config import (
    CLAUDE_MAX_TOKENS,
    INPUT_USD_PER_MTOK,
    MAX_INPUT_CHARS,
    MONTHLY_USD_CAP,
    OUTPUT_USD_PER_MTOK,
    REDIS_URL,
)

_CHARS_PER_TOKEN = 4  # Anthropic rough average — good enough for budgeting

_redis = None
if REDIS_URL:
    import redis  # type: ignore

    _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)

_memory: dict[str, float] = {}


def _month_key(user_id: str) -> str:
    return f"finlo:cost:{datetime.now(timezone.utc).strftime('%Y-%m')}:{user_id}"


def _estimate_usd(input_chars: int, output_chars: int) -> float:
    in_tokens = input_chars / _CHARS_PER_TOKEN
    out_tokens = output_chars / _CHARS_PER_TOKEN
    return (
        in_tokens / 1_000_000 * INPUT_USD_PER_MTOK
        + out_tokens / 1_000_000 * OUTPUT_USD_PER_MTOK
    )


def get_spent_usd(user_id: str) -> float:
    if _redis is not None:
        val = _redis.get(_month_key(user_id))
        return float(val) if val else 0.0
    return _memory.get(_month_key(user_id), 0.0)


def preflight(user_id: str, input_chars: int) -> None:
    """Raises 413 / 402 if the request can't be served."""
    if input_chars > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Input too large ({input_chars} chars). "
                f"Limit is {MAX_INPUT_CHARS}."
            ),
        )
    spent = get_spent_usd(user_id)
    worst_case = _estimate_usd(input_chars, CLAUDE_MAX_TOKENS * _CHARS_PER_TOKEN)
    if spent + worst_case > MONTHLY_USD_CAP:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Monthly AI budget exhausted "
                f"(spent ${spent:.2f} of ${MONTHLY_USD_CAP:.2f}). "
                f"Resets at the start of next month."
            ),
        )


def record_usage(user_id: str, input_chars: int, output_chars: int) -> float:
    """Charges the user for an actual request. Returns new running total."""
    cost = _estimate_usd(input_chars, output_chars)
    key = _month_key(user_id)
    if _redis is not None:
        pipe = _redis.pipeline()
        pipe.incrbyfloat(key, cost)
        pipe.expire(key, 60 * 60 * 24 * 90, nx=True)
        new_total, _ = pipe.execute()
        return float(new_total)
    _memory[key] = _memory.get(key, 0.0) + cost
    return _memory[key]
