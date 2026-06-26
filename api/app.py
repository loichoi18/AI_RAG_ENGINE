"""FastAPI application factory.

The application-factory pattern (``create_app``) builds a fresh, fully configured
app instance on demand — the seam that makes the API testable. The production
service graph is constructed in the ``lifespan`` startup (so module import and
tests stay cheap) and is wrapped in a guard so the app still boots if models or
backends are temporarily unavailable. Tests pass a pre-built ``services`` object
and the startup skips real construction entirely.

Wired here: request-context middleware (correlation id + latency), versioned
routers, and structured exception handlers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.errors import register_exception_handlers
from api.middleware import RequestContextMiddleware
from api.routers import documents, health, metrics, query
from api.services import AppServices, build_services
from configs.settings import Settings, get_settings
from utils.logging import configure_logging, get_logger


def create_app(settings: Settings | None = None, services: AppServices | None = None) -> FastAPI:
    """Build and return a configured FastAPI application.

    Parameters
    ----------
    settings:
        Optional settings override (primarily for tests).
    services:
        Optional pre-built service graph. When provided (tests), the lifespan
        startup skips real construction.
    """
    settings = settings or get_settings()
    configure_logging(settings.logging)
    logger = get_logger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "services", None) is None:
            try:
                app.state.services = build_services(settings)
                logger.info(
                    "application.startup",
                    app_name=settings.app_name,
                    environment=settings.environment.value,
                    llm_provider=settings.llm.provider.value,
                )
            except Exception as exc:  # noqa: BLE001 - boot even if models/backends absent
                logger.warning("application.startup_degraded", error=str(exc))
        yield
        logger.info("application.shutdown", app_name=settings.app_name)

    app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)
    app.state.settings = settings
    app.state.services = services

    app.add_middleware(RequestContextMiddleware)

    # CORS: allow the browser frontend (served from another origin, e.g. Vercel)
    # to call this API. Origins are configurable via RAG_CORS_ORIGINS.
    cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(documents.router)
    app.include_router(metrics.router)
    return app


# Module-level ASGI app for `uvicorn api.app:app`.
app = create_app()
