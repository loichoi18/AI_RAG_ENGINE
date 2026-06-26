"""Unit test for the FastAPI bootstrap.

Verifies the application-factory pattern: ``create_app`` builds a working app
whose health endpoint responds, without any backend services running.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import create_app


def test_health_endpoint_returns_ok() -> None:
    """GET /health returns 200 with the documented body."""
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
