"""Unit tests for chunkers.

A deterministic word tokenizer and a lexical fake embedder let us assert the
core guarantees of each strategy without model downloads: token budgets are
respected, provenance and ACL propagate, and the semantic chunker splits at a
genuine topic boundary.
"""

from __future__ import annotations

from ingestion.chunkers import FixedSizeChunker, RecursiveChunker, SemanticChunker
from ingestion.segment import SEGMENTS_KEY, Segment
from models.domain import ChunkStrategy, Document
from tests.fakes import FakeEmbedder, FakeWordTokenizer


def _doc(text: str, acl: list[str] | None = None, segments: list[Segment] | None = None) -> Document:
    metadata: dict[str, object] = {}
    if acl is not None:
        metadata["acl"] = acl
    if segments is not None:
        metadata[SEGMENTS_KEY] = segments
    return Document(source_path="/m", content=text, metadata=metadata)


# --- FixedSizeChunker ------------------------------------------------------


def test_fixed_chunker_respects_token_budget_and_overlap() -> None:
    words = " ".join(f"w{i}" for i in range(25))
    chunker = FixedSizeChunker(FakeWordTokenizer(), chunk_size=10, chunk_overlap=2)

    chunks = chunker.chunk(_doc(words, acl=["group:eng"]))

    assert len(chunks) > 1
    assert all(c.token_count <= 10 for c in chunks)
    assert all(c.chunk_strategy is ChunkStrategy.FIXED for c in chunks)
    assert all(c.acl == ["group:eng"] for c in chunks)


def test_fixed_chunker_rejects_bad_overlap() -> None:
    try:
        FixedSizeChunker(FakeWordTokenizer(), chunk_size=10, chunk_overlap=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlap >= size")


# --- RecursiveChunker ------------------------------------------------------


def test_recursive_chunker_packs_within_budget() -> None:
    text = "\n\n".join(" ".join(f"p{p}w{i}" for i in range(8)) for p in range(6))
    chunker = RecursiveChunker(FakeWordTokenizer(), chunk_size=20, chunk_overlap=4)

    chunks = chunker.chunk(_doc(text))

    assert len(chunks) >= 2
    assert all(c.token_count <= 20 for c in chunks)
    assert all(c.chunk_strategy is ChunkStrategy.RECURSIVE for c in chunks)


def test_recursive_chunker_preserves_segment_provenance() -> None:
    segments = [Segment(text="alpha beta gamma", section_title="A", page_number=3)]
    chunker = RecursiveChunker(FakeWordTokenizer(), chunk_size=50, chunk_overlap=0)

    chunks = chunker.chunk(_doc("alpha beta gamma", segments=segments))

    assert chunks[0].section_title == "A"
    assert chunks[0].page_number == 3


# --- SemanticChunker -------------------------------------------------------


def test_semantic_chunker_splits_on_topic_shift() -> None:
    # Two lexically distinct topics; the chunker should break between them.
    text = (
        "alpha alpha beta. alpha beta beta. "
        "gamma gamma delta. gamma delta delta."
    )
    chunker = SemanticChunker(
        FakeEmbedder(), FakeWordTokenizer(), max_tokens=100, threshold_percentile=60.0
    )

    chunks = chunker.chunk(_doc(text, acl=["group:all"]))

    assert len(chunks) >= 2
    assert all(c.chunk_strategy is ChunkStrategy.SEMANTIC for c in chunks)
    assert all(c.acl == ["group:all"] for c in chunks)
    # The first chunk is about the 'alpha/beta' topic, a later one about 'gamma'.
    assert "alpha" in chunks[0].text
    assert any("gamma" in c.text for c in chunks[1:])


def test_semantic_chunker_single_sentence_is_one_chunk() -> None:
    chunker = SemanticChunker(FakeEmbedder(), FakeWordTokenizer())
    chunks = chunker.chunk(_doc("only one sentence here"))
    assert len(chunks) == 1
