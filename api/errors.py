"""Structured exception handling.

All handled errors return a consistent :class:`~api.schemas.ErrorResponse`
``{error, detail, request_id}`` so clients get machine-readable failures and the
request id ties the response back to the logs.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import ErrorResponse
from utils.logging import get_logger

logger = get_logger("api.error")


class APIError(Exception):
    """Application error with an HTTP status and client-safe message."""

    def __init__(self, status_code: int, error: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.error = error
        self.detail = detail


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach structured handlers for app, validation, and unexpected errors."""

    @app.exception_handler(APIError)
    async def _handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        body = ErrorResponse(error=exc.error, detail=exc.detail, request_id=_request_id(request))
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        body = ErrorResponse(
            error="validation_error", detail=str(exc.errors()), request_id=_request_id(request)
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.error("api.unhandled", error=str(exc), path=request.url.path)
        body = ErrorResponse(
            error="internal_error", detail="An unexpected error occurred.",
            request_id=_request_id(request),
        )
        return JSONResponse(status_code=500, content=body.model_dump())
