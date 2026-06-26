"""Request middleware: correlation id + latency.

Assigns a ``request_id`` to every request, binds it into the structlog context
(so every log line emitted while handling the request carries it), measures
wall-clock latency, and returns the id in the ``X-Request-ID`` response header.
This is the backbone of request tracing.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from utils.logging import get_logger

logger = get_logger("api.request")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a request id and log request completion with latency."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "request",
                method=request.method,
                path=request.url.path,
                latency_ms=latency_ms,
            )
            structlog.contextvars.clear_contextvars()

        response.headers["X-Request-ID"] = request_id
        return response
