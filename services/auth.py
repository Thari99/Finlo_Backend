"""
Short-lived access JWT + long-lived opaque refresh token.

Access tokens are HS256 JWTs (stateless, fast verify, can't revoke individually).
Refresh tokens are random opaque strings stored in Redis (or in-memory for
dev), letting us revoke individual sessions by deleting the key.

Flow:
  /api/auth/google     → mint_session_pair → returns (access, refresh)
  /api/auth/refresh    → verify_refresh + mint_session_pair (rotate)
  /api/auth/sign-out   → revoke_refresh
  /api/ai/chat etc.    → Depends(verify_access)
"""
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Header, HTTPException

from config import (
    ACCESS_TOKEN_TTL_MIN,
    JWT_SECRET,
    REDIS_URL,
    REFRESH_TOKEN_TTL_DAYS,
)
from services import user_store

_ALGO = "HS256"

_redis = None
if REDIS_URL:
    import redis  # type: ignore

    _redis = redis.Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_timeout=2,
        socket_connect_timeout=2,
    )

_refresh_memory: dict[str, dict] = {}


# ── Access (JWT) ─────────────────────────────────────────────────────────────

def _mint_access(sub: str, is_premium: bool, email: str, name: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "email": email,
        "name": name,
        "is_premium": is_premium,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_TTL_MIN)).timestamp()),
        "typ": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=_ALGO)


def verify_access(authorization: str = Header(default="")) -> dict:
    """FastAPI dependency for protected endpoints."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[_ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Access token expired — refresh required",
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e

    if payload.get("typ") != "access":
        raise HTTPException(status_code=401, detail="Wrong token type")

    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token missing subject")

    return {
        "user_id": sub,
        "email": payload.get("email", ""),
        "name": payload.get("name", ""),
        "is_premium": bool(payload.get("is_premium", False)),
    }


# Back-compat alias for the older dependency name in tests/routes.
verify_token = verify_access


# ── Refresh (opaque, Redis-stored) ───────────────────────────────────────────

def _refresh_key(token: str) -> str:
    return f"finlo:refresh:{token}"


def _mint_refresh(sub: str) -> str:
    raw = secrets.token_urlsafe(48)
    ttl_seconds = REFRESH_TOKEN_TTL_DAYS * 24 * 3600
    payload = {
        "sub": sub,
        "iat": datetime.now(timezone.utc).isoformat(),
    }
    if _redis is not None:
        _redis.setex(_refresh_key(raw), ttl_seconds, json.dumps(payload))
    else:
        # In-memory fallback: stash expiry alongside payload.
        _refresh_memory[raw] = {
            **payload,
            "exp_ts": datetime.now(timezone.utc).timestamp() + ttl_seconds,
        }
    return raw


def verify_refresh(token: str) -> Optional[str]:
    """Returns the user_id (sub) if valid, else None."""
    if not token:
        return None
    if _redis is not None:
        raw = _redis.get(_refresh_key(token))
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        return payload.get("sub")
    entry = _refresh_memory.get(token)
    if entry is None:
        return None
    if entry["exp_ts"] < datetime.now(timezone.utc).timestamp():
        _refresh_memory.pop(token, None)
        return None
    return entry.get("sub")


def revoke_refresh(token: str) -> None:
    if not token:
        return
    if _redis is not None:
        _redis.delete(_refresh_key(token))
    else:
        _refresh_memory.pop(token, None)


# ── High-level: pair mint ────────────────────────────────────────────────────

def mint_session_pair(
    sub: str,
    email: str = "",
    name: str = "",
) -> dict:
    """Returns {access_token, refresh_token, expires_in}."""
    is_premium = user_store.is_premium(sub)
    access = _mint_access(sub, is_premium, email, name)
    refresh = _mint_refresh(sub)
    return {
        "access_token": access,
        "refresh_token": refresh,
        "expires_in": ACCESS_TOKEN_TTL_MIN * 60,
        "is_premium": is_premium,
    }
