"""Unit tests for the ingestion pipeline orchestration.

Uses fakes for every layer so the wiring (load -> chunk -> embed -> upsert) is
verified in isolation, including that unsupported files are skipped.
"""

from __future__ import annotations

from pathlib import Path

from ingestion.chunkers import RecursiveChunker
from ingestion.pipeline import IngestionPipeline
from tests.fakes import FakeEmbedder, FakeVectorStore, FakeWordTokenizer


def _pipeline(store: FakeVectorStore) -> IngestionPipeline:
    chunker = RecursiveChunker(FakeWordTokenizer(), chunk_size=20, chunk_overlap=4)
    return IngestionPipeline(chunker, FakeEmbedder(), store)


def test_ingest_document_indexes_chunks(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text(" ".join(f"word{i}" for i in range(60)), encoding="utf-8")

    store = FakeVectorStore()
    chunks = _pipeline(store).ingest_document(f)

    assert chunks
    assert len(store.points) == len(chunks)
    # Every stored chunk has a vector of the embedder's dimension.
    for chunk, vector in store.points.values():
        assert len(vector) == FakeEmbedder().dimension
        assert chunk.token_count > 0


def test_ingest_paths_skips_unsupported(tmp_path: Path) -> None:
    good = tmp_path / "a.txt"
    good.write_text("hello world from the corpus", encoding="utf-8")
    bad = tmp_path / "b.zip"
    bad.write_bytes(b"PK\x03\x04")

    store = FakeVectorStore()
    report = _pipeline(store).ingest_paths([good, bad])

    assert report.documents == 1
    assert str(bad) in report.skipped
    assert report.chunks == len(store.points)
