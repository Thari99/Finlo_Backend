from dotenv import load_dotenv
import os

load_dotenv()

# ── Anthropic ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
CLAUDE_TIMEOUT_S  = int(os.getenv("CLAUDE_TIMEOUT_S", "60"))
CLAUDE_MAX_TOKENS = int(os.getenv("CLAUDE_MAX_TOKENS", "1024"))

# ── Gemini (bill scanning OCR) ───────────────────────────────────────────────
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT_S  = int(os.getenv("GEMINI_TIMEOUT_S", "30"))
# Reject uploads larger than this so a malicious client can't OOM the worker.
OCR_MAX_IMAGE_MB  = int(os.getenv("OCR_MAX_IMAGE_MB", "8"))
# Separate quota for OCR so a heavy chat user doesn't lock themselves out of
# scanning (and vice-versa). Premium users get unlimited.
OCR_FREE_DAILY_LIMIT = int(os.getenv("OCR_FREE_DAILY_LIMIT", "20"))

# ── Google Sign-In ───────────────────────────────────────────────────────────
GOOGLE_CLIENT_ID  = os.getenv("GOOGLE_CLIENT_ID", "")

# ── JWT (short-lived access + long-lived refresh) ────────────────────────────
JWT_SECRET            = os.getenv("JWT_SECRET", "")
ACCESS_TOKEN_TTL_MIN  = int(os.getenv("ACCESS_TOKEN_TTL_MIN", "15"))
REFRESH_TOKEN_TTL_DAYS = int(os.getenv("REFRESH_TOKEN_TTL_DAYS", "30"))

# ── Quotas + cost guardrails ─────────────────────────────────────────────────
FREE_DAILY_LIMIT       = int(os.getenv("FREE_DAILY_LIMIT", "10"))
# Hard pre-check: refuse any single request whose total input (system +
# history + user message) exceeds this many characters. ~4 chars/token.
MAX_INPUT_CHARS        = int(os.getenv("MAX_INPUT_CHARS", "80000"))
# Monthly USD cap per user. Free users hit FREE_DAILY_LIMIT first; this is a
# defense-in-depth ceiling for premium / abusive accounts.
MONTHLY_USD_CAP        = float(os.getenv("MONTHLY_USD_CAP", "5.0"))
# Claude Haiku 4.5 list price. Override if Anthropic changes pricing.
INPUT_USD_PER_MTOK     = float(os.getenv("INPUT_USD_PER_MTOK", "1.0"))
OUTPUT_USD_PER_MTOK    = float(os.getenv("OUTPUT_USD_PER_MTOK", "5.0"))

# ── Environment ──────────────────────────────────────────────────────────────
ENV               = os.getenv("ENV", "development")

# ── Redis ────────────────────────────────────────────────────────────────────
# Required in production (rate limit, refresh tokens, monthly cost, user store
# all live here). Optional in development — services fall back to in-memory.
REDIS_URL         = os.getenv("REDIS_URL", "")

# ── CORS ─────────────────────────────────────────────────────────────────────
CORS_ORIGINS_RAW  = os.getenv("CORS_ORIGINS", "")

# ── Observability ────────────────────────────────────────────────────────────
SENTRY_DSN        = os.getenv("SENTRY_DSN", "")
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")

# ── Validation ───────────────────────────────────────────────────────────────
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set in .env. Get one at "
        "https://aistudio.google.com/apikey"
    )
if not GOOGLE_CLIENT_ID:
    raise RuntimeError("GOOGLE_CLIENT_ID is not set in .env")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET is not set in .env")
if ENV == "production" and not REDIS_URL:
    raise RuntimeError(
        "REDIS_URL is required in production. Refresh tokens, monthly cost "
        "tracking, and rate limits must be durable + cross-worker."
    )


def cors_origins() -> list[str]:
    if CORS_ORIGINS_RAW:
        return [o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()]
    return ["*"] if ENV == "development" else []
