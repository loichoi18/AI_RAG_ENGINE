"""Ingestion pipeline orchestrator.

Wires the ingestion layers together — load -> chunk -> embed -> index — behind a
single entry point. It depends only on the layer *interfaces* (Loader registry,
Chunker, Embedder, VectorStore), so any implementation can be swapped without
touching this orchestration. Embedding is batched in one call per document to
amortize model overhead.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ingestion.base import Chunker, Embedder
from ingestion.loaders.registry import LoaderRegistry
from models.domain import Chunk
from storage.base import VectorStore
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestionReport:
    """Summary of an ingestion run."""

    documents: int
    chunks: int
    skipped: list[str]


class IngestionPipeline:
    """Coordinates loading, chunking, embedding, and indexing of documents."""

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        registry: LoaderRegistry | None = None,
    ) -> None:
        self._registry = registry or LoaderRegistry()
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    def ingest_document(self, path: str | Path) -> list[Chunk]:
        """Load, chunk, embed, and index a single document.

        Returns
        -------
        list[Chunk]
            The chunks produced and indexed for this document.
        """
        document = self._registry.load(path)
        chunks = self._chunker.chunk(document)
        if not chunks:
            logger.warning("ingestion.no_chunks", source=str(path))
            return []

        vectors = self._embedder.embed([chunk.text for chunk in chunks])
        self._store.upsert(chunks, vectors)
        logger.info(
            "ingestion.document_indexed",
            source=str(path),
            document_id=document.document_id,
            chunks=len(chunks),
        )
        return chunks

    def ingest_paths(self, paths: Iterable[str | Path]) -> IngestionReport:
        """Ingest many documents, skipping unsupported/unreadable ones.

        Collection setup is a deployment concern and is the caller's
        responsibility (see ``scripts/ingest.py``); the pipeline assumes the
        store is ready.
        """
        total_chunks = 0
        documents = 0
        skipped: list[str] = []

        for path in paths:
            try:
                chunks = self.ingest_document(path)
            except (ValueError, FileNotFoundError) as exc:
                logger.warning("ingestion.skipped", source=str(path), reason=str(exc))
                skipped.append(str(path))
                continue
            documents += 1
            total_chunks += len(chunks)

        return IngestionReport(documents=documents, chunks=total_chunks, skipped=skipped)

    def ingest_directory(self, directory: str | Path, recursive: bool = True) -> IngestionReport:
        """Ingest every supported file under ``directory``."""
        root = Path(directory)
        pattern = "**/*" if recursive else "*"
        candidates: Sequence[Path] = [
            p
            for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in self._registry.supported_extensions
        ]
        return self.ingest_paths(candidates)
