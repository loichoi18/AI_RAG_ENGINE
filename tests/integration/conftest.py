from __future__ import annotations
import uuid
from collections.abc import Iterator
import pytest
from configs.settings import get_settings


@pytest.fixture
def qdrant_client() -> Iterator[object]:
    mod = pytest.importorskip("qdrant_client")
    settings = get_settings()
    try:
        client = mod.QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port, timeout=2.0)
        client.get_collections()
    except Exception:
        pytest.skip("No live Qdrant reachable; skipping integration test")
    yield client


@pytest.fixture
def temp_collection_name() -> str:
    return f"test_{uuid.uuid4().hex[:12]}"
