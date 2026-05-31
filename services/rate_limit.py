"""
Per-user daily message rate limiter.

Backed by Redis when REDIS_URL is set (durable + multi-worker safe).
Falls back to an in-memory dict for local development.

The two backends share the same interface:
    can_send(user_id, is_premium) -> bool
    get_remaining(user_id, is_premium) -> int
    increment(user_id) -> None
"""
from collections import defaultdict
from datetime import date
from typing import Optional

from config import FREE_DAILY_LIMIT, REDIS_URL

# ── Backend selection (eager so we fail loudly on bad Redis URL) ──────────────

_redis_client = None
if REDIS_URL:
    try:
        import redis  # type: ignore

        _redis_client = redis.Redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=2,
            socket_connect_timeout=2,
        )
        # Verify connectivity at import. Re-raises on auth/network failure.
        _redis_client.ping()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"REDIS_URL is set but Redis is unreachable: {e}. "
            f"Unset REDIS_URL to fall back to in-memory rate limiting."
        ) from e


# ── In-memory fallback state ─────────────────────────────────────────────────

_usage: dict[str, dict] = defaultdict(lambda: {"date": "", "count": 0})


def _today() -> str:
    return date.today().isoformat()


def _key(user_id: str) -> str:
    return f"finlo:ratelimit:{_today()}:{user_id}"


# ── Public API ───────────────────────────────────────────────────────────────

def get_remaining(user_id: str, is_premium: bool) -> int:
    if is_premium:
        return 999_999

    if _redis_client is not None:
        count = _get_redis_count(user_id)
    else:
        entry = _usage[user_id]
        count = entry["count"] if entry["date"] == _today() else 0

    return max(0, FREE_DAILY_LIMIT - count)


def can_send(user_id: str, is_premium: bool) -> bool:
    return get_remaining(user_id, is_premium) > 0


def increment(user_id: str) -> None:
    if _redis_client is not None:
        _incr_redis(user_id)
    else:
        entry = _usage[user_id]
        if entry["date"] != _today():
            entry["date"] = _today()
            entry["count"] = 0
        entry["count"] += 1


# ── Redis helpers ────────────────────────────────────────────────────────────

def _get_redis_count(user_id: str) -> int:
    val: Optional[str] = _redis_client.get(_key(user_id))  # type: ignore[union-attr]
    return int(val) if val else 0


def _incr_redis(user_id: str) -> None:
    pipe = _redis_client.pipeline()  # type: ignore[union-attr]
    k = _key(user_id)
    pipe.incr(k, 1)
    # Auto-expire 48h after first hit of the day so old keys don't linger.
    pipe.expire(k, 60 * 60 * 48, nx=True)
    pipe.execute()
