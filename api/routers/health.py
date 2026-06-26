"""Health-check routers.

Liveness probes used by Docker / orchestrators. Intentionally dependency-free so
they succeed even if downstream services (Qdrant, Ollama) are unavailable —
liveness ("is the process up?") is distinct from readiness, which arrives later.
Both the legacy ``/health`` and the versioned ``/v1/health`` are exposed.
"""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service liveness status."""
    return HealthResponse(status="ok")


@router.get("/v1/health", response_model=HealthResponse)
async def health_v1() -> HealthResponse:
    """Versioned liveness endpoint."""
    return HealthResponse(status="ok")
