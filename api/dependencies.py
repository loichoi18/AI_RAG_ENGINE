"""FastAPI dependencies.

Thin accessors that pull the wired :class:`AppServices` (and its parts) off
``app.state``. Endpoints depend on these, and tests override them with fakes via
``app.dependency_overrides`` — no global singletons, fully injectable.
"""

from __future__ import annotations

from starlette.requests import Request

from api.services import AppServices
from utils.metrics import MetricsRegistry


def get_services(request: Request) -> AppServices:
    """Return the application's wired service graph."""
    return request.app.state.services  # type: ignore[no-any-return]


def get_metrics(request: Request) -> MetricsRegistry:
    """Return the metrics registry."""
    return get_services(request).metrics
