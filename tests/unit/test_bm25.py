"""Unit tests for the BM25 sparse retriever."""

from __future__ import annotations

from models.domain import Chunk
from retrieval.sparse import BM25Retriever


def _corpus() -> list[Chunk]:
    return [
        Chunk(chunk_id="a", document_id="doc1", text="deploy the service with docker", acl=[]),
        Chunk(chunk_id="b", document_id="doc2", text="hr vacation and leave policy", acl=["group:hr"]),
        Chunk(chunk_id="c", document_id="doc1", text="deployment rollout and release steps", acl=[]),
    ]


def test_bm25_ranks_lexical_overlap() -> None:
    retriever = BM25Retriever(_corpus())
    results = retriever.retrieve("docker deploy", top_k=3)
    assert results
    assert results[0].chunk.chunk_id == "a"
    assert all(r.retriever_name == "sparse" for r in results)


def test_bm25_acl_filter_excludes_unauthorized() -> None:
    retriever = BM25Retriever(_corpus())
    results = retriever.retrieve("vacation policy", top_k=5, acl=["group:engineering"])
    assert all(r.chunk.chunk_id != "b" for r in results)


def test_bm25_document_filter() -> None:
    retriever = BM25Retriever(_corpus())
    results = retriever.retrieve("deploy", top_k=5, document_ids=["doc1"])
    assert {r.chunk.document_id for r in results} == {"doc1"}


def test_bm25_empty_index_returns_empty() -> None:
    assert BM25Retriever([]).retrieve("anything", top_k=3) == []
