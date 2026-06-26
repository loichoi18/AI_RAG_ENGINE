"""Structured logging via ``structlog``.

Strategy
--------
We emit **structured** log events (key/value pairs) rather than free-text lines.
Structured logs are machine-parseable, so in production every event carries
context (request id, latency, retriever, token counts) that can be queried in a
log aggregator without brittle regexes.

* In production (``json_logs=True``) we render newline-delimited JSON — the
  format expected by Docker log drivers and aggregators (Loki, ELK, etc.).
* In local development (``json_logs=False``) we render a colorized console
  format that is easier for a human to scan.

``configure_logging`` is idempotent and is called once at application startup.
Modules obtain a logger via ``get_logger(__name__)`` and never configure
logging themselves.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.types import Processor

from configs.settings import LoggingSettings


def configure_logging(settings: LoggingSettings) -> None:
    """Configure ``structlog`` and the stdlib logging bridge.

    Parameters
    ----------
    settings:
        Logging configuration controlling level and JSON vs. console rendering.
    """
    level = logging.getLevelNamesMapping()[settings.log_level.value]

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if settings.json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Route the stdlib root logger (used by uvicorn, qdrant-client, etc.)
    # through the same handler so output is uniform.
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger.

    Parameters
    ----------
    name:
        Logger name, conventionally ``__name__`` of the calling module.
    """
    return structlog.get_logger(name)
