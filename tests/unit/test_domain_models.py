"""Unit tests for the domain models.

Testing strategy
----------------
Domain models are the system's data contracts, so the tests assert both the
**happy path** (defaults and field assignment) and the **validation
guarantees** that later layers rely on: non-empty text, sane numeric bounds,
and normalized confidence. If these invariants hold, downstream code can trust
its inputs without re-checking them.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.domain import (
    Chunk,
    ChunkStrategy,
    Document,
    GenerationResult,
    RetrievalResult,
)


# --- Document --------------------------------------------------------------


def test_document_defaults() -> None:
    """A document auto-generates an id and an empty metadata dict."""
    doc = Document(source_path="/docs/a.pdf", content="hello world")
    assert doc.document_id  # non-empty generated id
    assert doc.metadata == {}
    assert doc.source_updated_at is None


def test_document_rejects_blank_content() -> None:
    """Whitespace-only content is rejected to avoid empty downstream chunks."""
    with pytest.raises(ValidationError):
        Document(source_path="/docs/a.pdf", content="   \n  ")


def test_document_forbids_unknown_fields() -> None:
    """Unknown fields surface typos early (extra='forbid')."""
    with pytest.raises(ValidationError):
        Document(source_path="/d", content="x", typo_field=1)  # type: ignore[call-arg]


# --- Chunk -----------------------------------------------------------------


def test_chunk_defaults() -> None:
    """Defaults: recursive strategy, empty ACL, zero tokens, generated id."""
    chunk = Chunk(document_id="doc1", text="a passage")
    assert chunk.chunk_id
    assert chunk.chunk_strategy is ChunkStrategy.RECURSIVE
    assert chunk.acl == []
    assert chunk.token_count == 0
    assert chunk.page_number is None


def test_chunk_rejects_blank_text() -> None:
    """A chunk with no usable text is invalid."""
    with pytest.raises(ValidationError):
        Chunk(document_id="doc1", text="  ")


def test_chunk_page_number_must_be_positive() -> None:
    """Page numbers are 1-indexed; zero/negative is invalid."""
    with pytest.raises(ValidationError):
        Chunk(document_id="doc1", text="x", page_number=0)


def test_chunk_token_count_non_negative() -> None:
    """Token counts cannot be negative."""
    with pytest.raises(ValidationError):
        Chunk(document_id="doc1", text="x", token_count=-1)


def test_chunk_accepts_acl_and_provenance() -> None:
    """Provenance and ACL fields are stored as provided."""
    chunk = Chunk(
        document_id="doc1",
        text="restricted passage",
        page_number=4,
        section_title="Security",
        chunk_strategy=ChunkStrategy.SEMANTIC,
        token_count=12,
        acl=["group:eng", "group:sec"],
    )
    assert chunk.page_number == 4
    assert chunk.section_title == "Security"
    assert chunk.chunk_strategy is ChunkStrategy.SEMANTIC
    assert chunk.acl == ["group:eng", "group:sec"]


# --- RetrievalResult -------------------------------------------------------


def test_retrieval_result_wraps_chunk() -> None:
    """A retrieval result carries its chunk, score, and originating retriever."""
    chunk = Chunk(document_id="doc1", text="passage")
    result = RetrievalResult(chunk=chunk, score=0.87, retriever_name="hybrid")
    assert result.chunk is chunk
    assert result.score == pytest.approx(0.87)
    assert result.retriever_name == "hybrid"


# --- GenerationResult ------------------------------------------------------


def test_generation_result_defaults() -> None:
    """Defaults: empty citations, zero confidence, empty usage."""
    gen = GenerationResult(answer="The API is async [1].")
    assert gen.citations == []
    assert gen.confidence == 0.0
    assert gen.token_usage == {}


def test_generation_confidence_bounds() -> None:
    """Confidence is normalized to [0, 1]."""
    with pytest.raises(ValidationError):
        GenerationResult(answer="x", confidence=1.5)
    with pytest.raises(ValidationError):
        GenerationResult(answer="x", confidence=-0.1)


def test_generation_result_full() -> None:
    """All fields round-trip as provided."""
    gen = GenerationResult(
        answer="Async requests are supported [1].",
        citations=["chunk-123"],
        confidence=0.92,
        token_usage={"prompt": 350, "completion": 40},
    )
    assert gen.citations == ["chunk-123"]
    assert gen.confidence == pytest.approx(0.92)
    assert gen.token_usage["prompt"] == 350
