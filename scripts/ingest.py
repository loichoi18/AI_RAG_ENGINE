"""CLI to ingest documents into the vector store.

Usage
-----
    python -m scripts.ingest path/to/file.pdf path/to/docs/
    python -m scripts.ingest --strategy semantic ./corpus

Wires concrete implementations (bge embedder + tokenizer, configured chunker,
Qdrant store) from application settings and runs the ingestion pipeline. This is
the production entry point; tests exercise the pipeline with injected fakes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.settings import Settings, get_settings
from ingestion.base import Chunker
from ingestion.chunkers import FixedSizeChunker, RecursiveChunker, SemanticChunker
from ingestion.embedding import BGEEmbedder
from ingestion.pipeline import IngestionPipeline
from models.domain import ChunkStrategy
from storage.qdrant_store import QdrantVectorStore
from utils.logging import configure_logging, get_logger
from utils.tokenizer import BGETokenizer


def _build_chunker(settings: Settings, embedder: BGEEmbedder) -> Chunker:
    """Construct the configured chunker, sharing the embedding tokenizer."""
    tokenizer = BGETokenizer(settings.embeddings.embedding_model)
    c = settings.chunking
    match c.strategy:
        case ChunkStrategy.FIXED:
            return FixedSizeChunker(tokenizer, c.chunk_size, c.chunk_overlap)
        case ChunkStrategy.RECURSIVE:
            return RecursiveChunker(tokenizer, c.chunk_size, c.chunk_overlap)
        case ChunkStrategy.SEMANTIC:
            return SemanticChunker(
                embedder, tokenizer, c.chunk_size, c.semantic_threshold_percentile
            )


def main() -> None:
    """Parse arguments and run ingestion."""
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest.")
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in ChunkStrategy],
        default=None,
        help="Override the configured chunking strategy.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.strategy:
        settings.chunking.strategy = ChunkStrategy(args.strategy)
    configure_logging(settings.logging)
    logger = get_logger("scripts.ingest")

    embedder = BGEEmbedder(
        settings.embeddings.embedding_model,
        settings.embeddings.device,
        settings.embeddings.batch_size,
        settings.embeddings.normalize,
    )
    store = QdrantVectorStore(
        collection_name=settings.qdrant.collection_name,
        vector_size=embedder.dimension,
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        api_key=settings.qdrant.api_key,
        https=settings.qdrant.https,
    )
    store.ensure_collection()
    pipeline = IngestionPipeline(_build_chunker(settings, embedder), embedder, store)

    expanded: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            report = pipeline.ingest_directory(p)
            logger.info(
                "ingestion.directory_done",
                directory=str(p),
                documents=report.documents,
                chunks=report.chunks,
                skipped=len(report.skipped),
            )
        else:
            expanded.append(p)

    if expanded:
        report = pipeline.ingest_paths(expanded)
        logger.info(
            "ingestion.files_done",
            documents=report.documents,
            chunks=report.chunks,
            skipped=len(report.skipped),
        )


if __name__ == "__main__":
    main()
