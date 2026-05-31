"""
Authoritative store for per-user metadata (premium status, etc.).

Redis-backed when REDIS_URL is set (durable + multi-worker safe). Falls back
to an in-memory dict for dev. The Redis schema:

    finlo:user:<sub>  → Hash {
        email, name, picture,
        is_premium, premium_until,
        created_at, updated_at,
    }

Premium state should be the source of truth for AI quotas + ad-free state.
The Flutter client's `PremiumProvider` is a hint only — every request that
cares about premium re-checks via the JWT (minted from this store).

TODO: when Firestore admin SDK is wired, migrate this to Firestore so the
mobile sync layer + backend share one source of truth.
"""
from datetime import datetime, timezone
from typing import Optional

from config import REDIS_URL

_redis = None
if REDIS_URL:
    import redis  # type: ignore

    _redis = redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_timeout=2,
        socket_connect_timeout=2,
    )

_memory: dict[str, dict] = {}


def _key(sub: str) -> str:
    return f"finlo:user:{sub}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_premium_active(record: dict) -> bool:
    if record.get("is_premium") != "1":
        return False
    until = record.get("premium_until")
    if not until:
        return True  # premium with no expiry
    try:
        return datetime.fromisoformat(until) > datetime.now(timezone.utc)
    except ValueError:
        return False


def upsert(
    sub: str,
    email: str = "",
    name: str = "",
    picture: str = "",
) -> dict:
    """Create or refresh a user record. Called on every successful sign-in."""
    existing = get(sub) or {}
    record = {
        "email": email or existing.get("email", ""),
        "name": name or existing.get("name", ""),
        "picture": picture or existing.get("picture", ""),
        "is_premium": existing.get("is_premium", "0"),
        "premium_until": existing.get("premium_until", ""),
        "created_at": existing.get("created_at", _now()),
        "updated_at": _now(),
    }
    _write(sub, record)
    return record


def get(sub: str) -> Optional[dict]:
    if _redis is not None:
        data = _redis.hgetall(_key(sub))
        return data or None
    return _memory.get(sub)


def is_premium(sub: str) -> bool:
    record = get(sub)
    return bool(record) and _is_premium_active(record)


def set_premium(sub: str, until_iso: str = "") -> None:
    record = get(sub) or {}
    record["is_premium"] = "1"
    record["premium_until"] = until_iso
    record["updated_at"] = _now()
    _write(sub, record)


def clear_premium(sub: str) -> None:
    record = get(sub) or {}
    record["is_premium"] = "0"
    record["premium_until"] = ""
    record["updated_at"] = _now()
    _write(sub, record)


def _write(sub: str, record: dict) -> None:
    if _redis is not None:
        _redis.hset(_key(sub), mapping=record)
    else:
        _memory[sub] = record
