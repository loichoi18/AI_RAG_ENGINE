"""Seed the vector store with demo data.

Ingests the bundled evaluation corpus (``evaluation/datasets/corpus.json``) into
Qdrant so a reviewer can query the system immediately after
``docker compose up``. Reuses the application's composition root, so it indexes
with the exact components the API uses.

Usage:
    python -m scripts.bootstrap_demo
"""

from __future__ import annotations

from api.services import build_services
from configs.settings import get_settings
from evaluation.dataset import load_corpus_documents
from utils.logging import configure_logging, get_logger


def main() -> None:
    """Ingest the demo corpus into the configured vector store."""
    settings = get_settings()
    configure_logging(settings.logging)
    logger = get_logger("scripts.bootstrap_demo")

    services = build_services(settings)
    services.store.ensure_collection()

    documents = load_corpus_documents()
    total_chunks = 0
    for document in documents:
        chunks = services.chunker.chunk(document)
        if not chunks:
            continue
        vectors = services.embedder.embed([c.text for c in chunks])
        services.store.upsert(chunks, vectors)
        total_chunks += len(chunks)

    services.reindex_sparse()
    logger.info("bootstrap.done", documents=len(documents), chunks=total_chunks)
    print(f"Seeded {len(documents)} documents ({total_chunks} chunks). Try the dashboard at :8501")


if __name__ == "__main__":
    main()
