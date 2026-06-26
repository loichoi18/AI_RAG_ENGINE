"""Integration tests for retrieval against a live Qdrant.

Exercises the dense and hybrid retrievers end-to-end over a real collection,
including ACL filtering and building the BM25 index from a Qdrant scroll. A
deterministic fake embedder is used so no model download is required; the focus
is the retrieval + storage integration, not embedding quality.

Run the stack first:
    docker compose -f docker/docker-compose.yml up -d qdrant
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from models.domain import Chunk
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import BM25Retriever
from storage.qdrant_store import QdrantVectorStore
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


@pytest.fixture
def populated_store(
    qdrant_client: object, temp_collection_name: str
) -> Iterator[tuple[QdrantVectorStore, FakeEmbedder]]:
    embedder = FakeEmbedder()
    store = QdrantVectorStore(
        collection_name=temp_collection_name,
        vector_size=embedder.dimension,
        client=qdrant_client,  # type: ignore[arg-type]
    )
    store.ensure_collection()
    chunks = [
        Chunk(chunk_id="a", document_id="doc1", text="deploy service with docker", acl=[]),
        Chunk(chunk_id="b", document_id="doc2", text="hr vacation leave policy", acl=["group:hr"]),
        Chunk(chunk_id="c", document_id="doc1", text="deployment rollout release", acl=[]),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    yield store, embedder
    qdrant_client.delete_collection(temp_collection_name)  # type: ignore[attr-defined]


def test_dense_retrieval_end_to_end(
    populated_store: tuple[QdrantVectorStore, FakeEmbedder],
) -> None:
    store, embedder = populated_store
    results = DenseRetriever(embedder, store).retrieve("docker deploy", top_k=5)
    assert results
    assert all(r.retriever_name == "dense" for r in results)


def test_dense_acl_filter_end_to_end(
    populated_store: tuple[QdrantVectorStore, FakeEmbedder],
) -> None:
    store, embedder = populated_store
    results = DenseRetriever(embedder, store).retrieve(
        "vacation policy", top_k=5, acl=["group:engineering"]
    )
    assert all(r.chunk.chunk_id != "b" for r in results)


def test_hybrid_with_bm25_from_scroll(
    populated_store: tuple[QdrantVectorStore, FakeEmbedder],
) -> None:
    store, embedder = populated_store
    sparse = BM25Retriever(store.scroll_all())  # BM25 built from the live corpus
    hybrid = HybridRetriever(DenseRetriever(embedder, store), sparse, candidate_k=10)

    results = hybrid.retrieve("docker deploy", top_k=3)
    assert results
    assert all(r.retriever_name == "hybrid" for r in results)
    assert results[0].chunk.chunk_id in {"a", "c"}
