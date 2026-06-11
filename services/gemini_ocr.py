"""
Gemini-powered bill / receipt OCR.

Image bytes go in, structured JSON comes out — no on-disk storage. The
caller (routes/ocr.py) is responsible for auth, rate limiting, and
multipart file handling; this module is pure "give me bytes, get JSON".

Gemini occasionally wraps JSON in ```json fences or prepends an apology
line. `_parse_json` strips the noise before json.loads().
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TIMEOUT_S
from services.logging_setup import logger

# Configure once at import; the client is stateless.
genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel(GEMINI_MODEL)

# Tight prompt — `response_mime_type: application/json` already forces a
# single well-formed JSON object, so we don't need to repeat "no prose" verbosely.
_PROMPT = """Parse this receipt image. Return JSON:
{
  "merchant": str|null,
  "date": "YYYY-MM-DD"|null,
  "currency": "LKR"|"INR"|"USD"|"GBP"|...|null,
  "subtotal": num|null,
  "tax": num|null,
  "discount": num|null,
  "total": num,
  "items": [{"description": str, "quantity": num|null, "unit_price": num|null, "total": num}],
  "confidence": 0.0-1.0
}

Rules:
- "total" = final amount paid (after tax/discount).
- "items" = product/service lines only. Exclude subtotal/tax/discount/total/change rows.
- Quantity default = 1.
- Use null when unreadable. Plain numbers, no symbols.
"""

# Tighter generation config: lower max_output_tokens (most receipts produce
# <800 tokens of JSON) and temperature 0 for deterministic + fast output.
_GENERATION_CONFIG = {
    "response_mime_type": "application/json",
    "temperature": 0.0,
    "max_output_tokens": 1024,
}


class OcrError(Exception):
    """Raised when Gemini call fails or returns unusable output. The route
    handler converts these to 502s for the client."""

    def __init__(self, message: str, *, code: str):
        super().__init__(message)
        self.code = code


async def scan_receipt(image_bytes: bytes, mime_type: str) -> dict[str, Any]:
    """Send `image_bytes` (already loaded into memory by FastAPI's UploadFile)
    to Gemini and return the parsed receipt dict. Image is NEVER written to
    disk by this service.

    Raises [OcrError] on Gemini failure / malformed output.
    """
    started = time.monotonic()
    try:
        # generate_content is sync in the SDK; the route runs it inside
        # FastAPI's threadpool by virtue of being an async def with await.
        # For now we accept the blocking call — Gemini 2.0 Flash is fast
        # enough (~1-2s) that a threadpool worker is fine.
        response = _model.generate_content(
            [
                _PROMPT,
                {"mime_type": mime_type, "data": image_bytes},
            ],
            generation_config=_GENERATION_CONFIG,
            request_options={"timeout": GEMINI_TIMEOUT_S},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("ocr.gemini_failed", err=str(e)[:200])
        raise OcrError(
            "Couldn't reach the OCR provider. Try again in a moment.",
            code="provider_unreachable",
        ) from e

    raw = (response.text or "").strip()
    if not raw:
        logger.warning("ocr.empty_response")
        raise OcrError(
            "OCR returned an empty result — try a clearer photo.",
            code="empty_result",
        )

    data = _parse_json(raw)
    if data is None:
        logger.warning("ocr.bad_json", raw_preview=raw[:200])
        raise OcrError(
            "Couldn't read the receipt — try a clearer, well-lit photo.",
            code="bad_json",
        )

    normalized = _normalize(data)
    elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.info(
        "ocr.success",
        elapsed_ms=elapsed_ms,
        has_total=normalized.get("total") is not None,
        item_count=len(normalized.get("items") or []),
        confidence=normalized.get("confidence"),
    )
    return normalized


# ── Helpers ──────────────────────────────────────────────────────────────────

# Strip ```json ... ``` fences Gemini sometimes adds even in JSON mode.
_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json(raw: str) -> Optional[dict[str, Any]]:
    candidate = _FENCE_PATTERN.sub("", raw).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Defensive: try to find the first { ... } block in the response.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _normalize(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce Gemini's output to predictable types so the Flutter client can
    deserialize without defensive parsing on every field. Anything missing
    or wrong-typed becomes null (or empty list for items)."""

    def _f(key: str) -> Optional[float]:
        v = data.get(key)
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v.replace(",", "").strip())
            except ValueError:
                return None
        return None

    def _s(key: str) -> Optional[str]:
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        return None

    raw_items = data.get("items")
    items: list[dict[str, Any]] = []
    if isinstance(raw_items, list):
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            desc = it.get("description")
            total_v = it.get("total")
            if not isinstance(desc, str) or not desc.strip():
                continue
            try:
                total_n = float(total_v) if total_v is not None else None
            except (TypeError, ValueError):
                continue
            if total_n is None:
                continue
            qty_v = it.get("quantity")
            try:
                qty_n = (
                    float(qty_v)
                    if qty_v is not None and isinstance(qty_v, (int, float, str))
                    else None
                )
            except (TypeError, ValueError):
                qty_n = None
            unit_v = it.get("unit_price")
            try:
                unit_n = (
                    float(unit_v)
                    if unit_v is not None and isinstance(unit_v, (int, float, str))
                    else None
                )
            except (TypeError, ValueError):
                unit_n = None
            items.append(
                {
                    "description": desc.strip()[:80],
                    "quantity": qty_n,
                    "unit_price": unit_n,
                    "total": total_n,
                }
            )

    confidence = _f("confidence")
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))

    return {
        "merchant": _s("merchant"),
        "date": _s("date"),
        "currency": _s("currency"),
        "subtotal": _f("subtotal"),
        "tax": _f("tax"),
        "discount": _f("discount"),
        "total": _f("total"),
        "items": items,
        "confidence": confidence if confidence is not None else 0.7,
    }
