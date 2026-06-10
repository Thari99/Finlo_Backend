"""
POST /api/ocr/scan-bill

Authenticated multipart upload. Forwards the image to Gemini, returns the
structured receipt JSON. Image is held in memory only — never written to
disk by this service.

Quota: shares the daily rate limiter with chat but uses a separate
OCR_FREE_DAILY_LIMIT so a heavy chat user can still scan, and vice-versa.
Premium users get unlimited (modulo the monthly USD cost guard).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config import OCR_FREE_DAILY_LIMIT, OCR_MAX_IMAGE_MB
from services import rate_limit
from services.auth import verify_access
from services.gemini_ocr import OcrError, scan_receipt
from services.logging_setup import logger

router = APIRouter(prefix="/api/ocr", tags=["OCR"])

_ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


def _sniff_mime(body: bytes) -> str:
    """Return a MIME type from the file's magic header, or empty string if
    unknown. Covers the formats our allowlist accepts.

    Mobile multipart uploads frequently arrive with content-type
    `application/octet-stream` (image_picker on Android does this), so we
    sniff the actual bytes as a fallback before rejecting."""
    if len(body) < 12:
        return ""
    if body.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if body[:4] == b"RIFF" and body[8:12] == b"WEBP":
        return "image/webp"
    return ""


@router.post("/scan-bill")
async def scan_bill(
    file: UploadFile = File(...),
    user: dict = Depends(verify_access),
) -> JSONResponse:
    user_id = user["user_id"]
    is_premium = user["is_premium"]

    # Per-user daily quota — separate counter from /chat so the two features
    # don't starve each other.
    if not _can_scan(user_id, is_premium):
        remaining = _remaining_scans(user_id, is_premium)
        logger.info("ocr.rate_limited", user_id=user_id, remaining=remaining)
        raise HTTPException(
            status_code=429,
            detail=f"Daily scan limit reached. {remaining} scans left today.",
        )

    # Reject non-image uploads early. Gemini would handle most of these but
    # we want a fast, predictable error for the client.
    raw_content_type = (file.content_type or "").lower().split(";", 1)[0].strip()
    # Sniff the file header when the client didn't send a usable MIME — most
    # often the case with multipart uploads from mobile picker libraries that
    # default to application/octet-stream.
    image_bytes = await file.read()
    sniffed = _sniff_mime(image_bytes)
    content_type = raw_content_type if raw_content_type in _ALLOWED_MIME else sniffed

    logger.info(
        "ocr.request",
        user_id=user_id,
        bytes=len(image_bytes),
        raw_mime=raw_content_type,
        sniffed_mime=sniffed,
        resolved_mime=content_type,
    )

    if content_type not in _ALLOWED_MIME:
        logger.warning(
            "ocr.bad_mime",
            raw=raw_content_type,
            sniffed=sniffed,
        )
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use JPEG, PNG, or WEBP.",
        )

    # Read into memory with a hard cap so a giant upload can't OOM the worker.
    max_bytes = OCR_MAX_IMAGE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        logger.warning("ocr.too_large", bytes=len(image_bytes))
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Max {OCR_MAX_IMAGE_MB} MB.",
        )
    if len(image_bytes) < 1024:
        logger.warning("ocr.too_small", bytes=len(image_bytes))
        raise HTTPException(
            status_code=400,
            detail="Image is too small to be a receipt.",
        )

    logger.info(
        "ocr.start",
        user_id=user_id,
        bytes=len(image_bytes),
        mime=content_type,
        is_premium=is_premium,
    )

    try:
        result = await scan_receipt(image_bytes, content_type)
    except OcrError as e:
        # Provider failures bubble up as 502 with a stable error code the
        # client can branch on.
        raise HTTPException(
            status_code=502,
            detail={"code": e.code, "message": str(e)},
        ) from None

    # Only increment the counter on a successful scan — failed attempts
    # (bad image, provider error) shouldn't burn the user's daily quota.
    _increment_scan(user_id)

    return JSONResponse({"success": True, "data": result})


# ── Quota helpers (thin wrappers around services.rate_limit) ─────────────────

def _can_scan(user_id: str, is_premium: bool) -> bool:
    if is_premium:
        return True
    return _scan_count_today(user_id) < OCR_FREE_DAILY_LIMIT


def _remaining_scans(user_id: str, is_premium: bool) -> int:
    if is_premium:
        return 999_999
    return max(0, OCR_FREE_DAILY_LIMIT - _scan_count_today(user_id))


# Use a parallel counter namespace ("scan") so it doesn't clobber chat usage.
# Implemented on the existing rate_limit backend via a per-day suffix.

def _scan_count_today(user_id: str) -> int:
    return _read(_scan_key(user_id))


def _increment_scan(user_id: str) -> None:
    _incr(_scan_key(user_id))


def _scan_key(user_id: str) -> str:
    return f"scan:{user_id}"


# Reach into the rate_limit module for raw read/incr — both backends already
# expose Redis + in-memory paths, we just want a separate counter name.
def _read(key: str) -> int:
    client = getattr(rate_limit, "_redis_client", None)
    if client is not None:
        full_key = f"finlo:ratelimit:{rate_limit._today()}:{key}"  # type: ignore[attr-defined]
        v = client.get(full_key)
        return int(v) if v else 0
    # In-memory fallback. Same shape as _usage in rate_limit.py.
    entry = rate_limit._usage[key]  # type: ignore[attr-defined]
    return entry["count"] if entry["date"] == rate_limit._today() else 0  # type: ignore[attr-defined]


def _incr(key: str) -> None:
    client = getattr(rate_limit, "_redis_client", None)
    if client is not None:
        full_key = f"finlo:ratelimit:{rate_limit._today()}:{key}"  # type: ignore[attr-defined]
        pipe = client.pipeline()
        pipe.incr(full_key, 1)
        pipe.expire(full_key, 60 * 60 * 48, nx=True)
        pipe.execute()
        return
    entry = rate_limit._usage[key]  # type: ignore[attr-defined]
    today = rate_limit._today()  # type: ignore[attr-defined]
    if entry["date"] != today:
        entry["date"] = today
        entry["count"] = 0
    entry["count"] += 1
