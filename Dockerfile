# syntax=docker/dockerfile:1.7

# Production image for the Finlo FastAPI backend.
#
# Build:   docker build -t finlo-backend .
# Run:     docker run -p 8000:8000 --env-file .env finlo-backend
# Render:  push to GitHub + connect repo; Render uses this file automatically.

FROM python:3.11-slim AS base

# Don't write .pyc files; stream stdout/stderr so logs aren't buffered.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install runtime OS deps (kept minimal — langchain-anthropic + google-auth need
# only their pure-Python deps; if you later add a wheel that requires gcc, add
# build-essential here and use a multi-stage build).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user.
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# Install Python deps FIRST so the Docker layer cache survives unrelated code edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Now copy source.
COPY --chown=app:app . .

USER app

# Render injects $PORT at runtime; fall back to 8000 for local `docker run`.
ENV PORT=8000 \
    WEB_CONCURRENCY=2 \
    TIMEOUT=120

EXPOSE 8000

# Healthcheck so Render / Docker can verify the service is live.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:${PORT}/health || exit 1

# Use shell form so env-var substitution works ($PORT, $WEB_CONCURRENCY, $TIMEOUT).
CMD exec gunicorn main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers ${WEB_CONCURRENCY} \
    --bind 0.0.0.0:${PORT} \
    --timeout ${TIMEOUT} \
    --access-logfile - \
    --error-logfile -
