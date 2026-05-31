import uuid

import sentry_sdk
import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentry_sdk.integrations.fastapi import FastApiIntegration

from config import ENV, SENTRY_DSN, cors_origins
from routes import ai, auth
from services.logging_setup import configure_logging, logger

# ── Logging + Sentry must come before app creation ──────────────────────────
configure_logging()

if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=ENV,
        integrations=[FastApiIntegration()],
        traces_sample_rate=0.1 if ENV == "production" else 0.0,
        send_default_pii=False,
    )
    logger.info("sentry.enabled", environment=ENV)


app = FastAPI(
    title="Finlo AI Backend",
    version="1.0.0",
    docs_url="/docs" if ENV == "development" else None,
    redoc_url="/redoc" if ENV == "development" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Bind a per-request UUID into structlog so every log line carries it."""
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:16]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        path=request.url.path,
        method=request.method,
    )
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


# ── Routes ───────────────────────────────────────────────────────────────────

app.include_router(auth.router)
app.include_router(ai.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "finlo-ai-backend"}


logger.info("app.started", environment=ENV)
