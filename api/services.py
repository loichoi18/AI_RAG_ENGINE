"""Application composition root.

Builds the concrete service graph once at startup (the production wiring) and
holds it on a single :class:`AppServices` object that endpoints reach via
dependency injection. Heavy models load lazily on first use, so startup is cheap;
the Qdrant collection is ensured best-effort so the API still boots if Qdrant is
not yet reachable.

Tests do NOT call :func:`build_services`; they inject a fake ``AppServices`` via
dependency overrides, which is exactly why all wiring lives behind this seam.
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.settings import Settings
from ingestion.base import Chunker
from generation.gate import ScoreThresholdGate
from generation.generator import GroundedGenerator
from generation.llm.factory import build_llm_client
from ingestion.chunkers import RecursiveChunker
from ingestion.embedding import BGEEmbedder
from ingestion.pipeline import IngestionPipeline
from reranking.cross_encoder import CrossEncoderReranker
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.sparse import BM25Retriever
from services.answer_service import AnswerService
from storage.qdrant_store import QdrantVectorStore
from utils.logging import get_logger
from utils.metrics import MetricsRegistry
from utils.tokenizer import BGETokenizer
from utils.tracing import Tracer, build_tracer

logger = get_logger(__name__)


@dataclass
class AppServices:
    """The wired service graph shared across requests."""

    settings: Settings
    store: QdrantVectorStore
    embedder: BGEEmbedder
    chunker: Chunker
    sparse: BM25Retriever
    retrievers: dict[str, object]
    ingestion: IngestionPipeline
    answer_services: dict[str, AnswerService]
    metrics: MetricsRegistry
    tracer: Tracer

    def reindex_sparse(self) -> None:
        """Rebuild the in-memory BM25 index from the current corpus.

        Called after ingestion so hybrid retrieval reflects new documents. The
        hybrid retriever shares this BM25 object, so updating it in place
        updates hybrid retrieval too.
        """
        self.sparse.index(self.store.scroll_all())


def build_services(settings: Settings) -> AppServices:
    """Construct the production service graph from ``settings``."""
    embedder = BGEEmbedder(
        settings.embeddings.embedding_model,
        settings.embeddings.device,
        settings.embeddings.batch_size,
        settings.embeddings.normalize,
    )
    store = QdrantVectorStore(
        collection_name=settings.qdrant.collection_name,
        vector_size=settings.embeddings.dimension,
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        api_key=settings.qdrant.api_key,
        https=settings.qdrant.https,
    )
    try:
        store.ensure_collection()
        corpus = store.scroll_all()
    except Exception as exc:  # noqa: BLE001 - boot even if Qdrant is down
        logger.warning("startup.qdrant_unavailable", error=str(exc))
        corpus = []

    dense = DenseRetriever(embedder, store)
    sparse = BM25Retriever(corpus)
    hybrid = HybridRetriever(
        dense, sparse, rrf_k=settings.retrieval.rrf_k, candidate_k=settings.retrieval.top_k_dense
    )

    reranker = CrossEncoderReranker(settings.reranker.model_name, settings.reranker.device)
    gate = ScoreThresholdGate(settings.generation.gate_threshold)
    generator = GroundedGenerator(build_llm_client(settings.llm))

    def _answer_service(retriever: object) -> AnswerService:
        return AnswerService(
            retriever=retriever,  # type: ignore[arg-type]
            reranker=reranker,
            gate=gate,
            generator=generator,
            candidate_k=settings.retrieval.top_k_dense,
            top_k=settings.reranker.top_k,
            refusal_message=settings.generation.refusal_message,
        )

    chunker = RecursiveChunker(
        BGETokenizer(settings.embeddings.embedding_model),
        settings.chunking.chunk_size,
        settings.chunking.chunk_overlap,
    )
    ingestion = IngestionPipeline(chunker, embedder, store)

    return AppServices(
        settings=settings,
        store=store,
        embedder=embedder,
        chunker=chunker,
        sparse=sparse,
        retrievers={"hybrid": hybrid, "dense": dense},
        ingestion=ingestion,
        answer_services={"hybrid": _answer_service(hybrid), "dense": _answer_service(dense)},
        metrics=MetricsRegistry(),
        tracer=build_tracer("structlog"),
    )
