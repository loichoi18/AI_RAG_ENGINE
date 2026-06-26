"""Integration tests for :class:`QdrantVectorStore` against a live Qdrant.

Verifies the three guarantees that unit tests with fakes cannot: real
collection creation, idempotent upserts (re-upserting a chunk does not
duplicate it), and ACL filtering enforced inside the engine. Vectors are passed
explicitly so no embedding model is needed.

Run the stack first:
    docker compose -f docker/docker-compose.yml up -d qdrant
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from models.domain import Chunk, ChunkStrategy
from storage.qdrant_store import QdrantVectorStore

pytestmark = pytest.mark.integration

_DIM = 4


@pytest.fixture
def store(qdrant_client: object, temp_collection_name: str) -> Iterator[QdrantVectorStore]:
    s = QdrantVectorStore(
        collection_name=temp_collection_name,
        vector_size=_DIM,
        client=qdrant_client,  # type: ignore[arg-type]
    )
    s.ensure_collection()
    yield s
    qdrant_client.delete_collection(temp_collection_name)  # type: ignore[attr-defined]


def _chunk(chunk_id: str, text: str, acl: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text=text,
        chunk_strategy=ChunkStrategy.RECURSIVE,
        token_count=len(text.split()),
        acl=acl or [],
    )


def test_ensure_collection_is_idempotent(store: QdrantVectorStore) -> None:
    store.ensure_collection()  # second call must not raise
    assert store.count() == 0


def test_upsert_is_idempotent_on_chunk_id(store: QdrantVectorStore) -> None:
    chunk = _chunk("c1", "alpha beta")
    store.upsert([chunk], [[1.0, 0.0, 0.0, 0.0]])
    store.upsert([chunk], [[1.0, 0.0, 0.0, 0.0]])  # same id again
    assert store.count() == 1


def test_search_returns_indexed_chunk(store: QdrantVectorStore) -> None:
    store.upsert([_chunk("c1", "alpha beta")], [[1.0, 0.0, 0.0, 0.0]])
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].chunk.text == "alpha beta"


def test_acl_filter_excludes_unauthorized(store: QdrantVectorStore) -> None:
    store.upsert(
        [
            _chunk("public", "public text", acl=[]),
            _chunk("eng", "eng only", acl=["group:eng"]),
            _chunk("sec", "sec only", acl=["group:sec"]),
        ],
        [[1.0, 0.0, 0.0, 0.0]] * 3,
    )

    # Caller in group:eng sees the public chunk and the eng chunk, not sec.
    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=10, acl=["group:eng"])
    ids = {r.chunk.chunk_id for r in results}
    assert ids == {"public", "eng"}


def test_delete_removes_chunk(store: QdrantVectorStore) -> None:
    store.upsert([_chunk("c1", "alpha")], [[1.0, 0.0, 0.0, 0.0]])
    store.delete(["c1"])
    assert store.count() == 0
