"""Metrics endpoint: GET /v1/metrics.

Returns an in-process snapshot of request counts, success/error rates, and
per-operation latency summaries (retrieval/generation/query) as JSON.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_services
from api.schemas import MetricsResponse
from api.services import AppServices

router = APIRouter(prefix="/v1", tags=["metrics"])


@router.get("/metrics", response_model=MetricsResponse)
async def metrics(services: AppServices = Depends(get_services)) -> MetricsResponse:
    """Return the current metrics snapshot."""
    snapshot = services.metrics.snapshot()
    return MetricsResponse(**snapshot)
