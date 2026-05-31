"""
Structured JSON logging via structlog + request-scoped context binding.

Usage in route/service code:
    from services.logging_setup import logger
    logger.info("chat.start", user_id=user_id, prompt_chars=len(msg))

`request_id` and `user_id` are bound per-request by the middleware in
`main.py` and appear automatically on every subsequent log line for that
request.
"""
import logging
import sys

import structlog

from config import ENV, LOG_LEVEL


def configure_logging() -> None:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    # stdlib logging — sink for libraries that use it (uvicorn, anthropic, etc.)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if ENV == "development":
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


logger = structlog.get_logger("finlo")
