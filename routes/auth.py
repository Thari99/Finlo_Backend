from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from services import user_store
from services.auth import (
    mint_session_pair,
    revoke_refresh,
    verify_access,
    verify_refresh,
)
from services.google_auth import verify_google_id_token
from services.logging_setup import logger

router = APIRouter(prefix="/api/auth", tags=["Auth"])


class GoogleSignInRequest(BaseModel):
    id_token: str


class RefreshRequest(BaseModel):
    refresh_token: str


class SignOutRequest(BaseModel):
    refresh_token: str = ""


class SessionResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    user_id: str
    email: str
    name: str
    picture: str = ""
    is_premium: bool = False


@router.post("/google", response_model=SessionResponse)
async def google_sign_in(body: GoogleSignInRequest):
    info = verify_google_id_token(body.id_token)
    sub = info["sub"]

    # Source of truth for user metadata + premium status lives server-side.
    user_store.upsert(
        sub=sub,
        email=info["email"],
        name=info["name"],
        picture=info["picture"],
    )

    pair = mint_session_pair(sub=sub, email=info["email"], name=info["name"])
    logger.info("auth.sign_in", user_id=sub, email=info["email"])

    return SessionResponse(
        access_token=pair["access_token"],
        refresh_token=pair["refresh_token"],
        expires_in=pair["expires_in"],
        user_id=sub,
        email=info["email"],
        name=info["name"],
        picture=info["picture"],
        is_premium=pair["is_premium"],
    )


@router.post("/refresh", response_model=SessionResponse)
async def refresh(body: RefreshRequest):
    sub = verify_refresh(body.refresh_token)
    if not sub:
        raise HTTPException(status_code=401, detail="Refresh token invalid or expired")

    # Rotate: invalidate the old refresh token before issuing the new pair.
    revoke_refresh(body.refresh_token)

    record = user_store.get(sub) or {}
    pair = mint_session_pair(
        sub=sub,
        email=record.get("email", ""),
        name=record.get("name", ""),
    )
    logger.info("auth.refresh", user_id=sub)

    return SessionResponse(
        access_token=pair["access_token"],
        refresh_token=pair["refresh_token"],
        expires_in=pair["expires_in"],
        user_id=sub,
        email=record.get("email", ""),
        name=record.get("name", ""),
        picture=record.get("picture", ""),
        is_premium=pair["is_premium"],
    )


@router.post("/sign-out")
async def sign_out(body: SignOutRequest):
    """Revokes the supplied refresh token. Idempotent."""
    if body.refresh_token:
        revoke_refresh(body.refresh_token)
    return {"ok": True}


@router.get("/me")
async def me(user: dict = Depends(verify_access)):
    return user
