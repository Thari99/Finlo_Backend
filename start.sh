#!/usr/bin/env bash
# Production entrypoint. Run from the backend/ directory.
#
# Local dev still uses:  uvicorn main:app --reload --host 0.0.0.0 --port 8000
# This script is for the deployed host (Linux container, Render, Fly, etc.).
#
# Behind a TLS-terminating proxy (nginx/Caddy/Cloudflare/ALB), bind to
# 127.0.0.1 instead of 0.0.0.0 and let the proxy forward.

set -euo pipefail

WORKERS="${WEB_CONCURRENCY:-2}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TIMEOUT="${TIMEOUT:-120}"

exec gunicorn main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers "$WORKERS" \
  --bind "${HOST}:${PORT}" \
  --timeout "$TIMEOUT" \
  --access-logfile - \
  --error-logfile -
