"""Query endpoint: POST /v1/query.

Runs the full RAG pipeline (retrieve -> rerank -> gate -> generate) via the
selected mode's :class:`AnswerService`, and returns the answer with citations,
confidence, sources, and the retrieved chunks. Emits a per-query trace/log with
``query_id, latency, retrieval_count, rerank_count, token_usage, confidence``.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends

from api.dependencies import get_services
from api.errors import APIError
from api.schemas import QueryRequest, QueryResponse, RetrievedChunk, SourceItem
from api.services import AppServices
from retrieval.access import get_user_acl
from utils.logging import get_logger

router = APIRouter(prefix="/v1", tags=["query"])
logger = get_logger("api.query")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, services: AppServices = Depends(get_services)) -> QueryResponse:
    """Answer a question over the knowledge base."""
    query_id = uuid.uuid4().hex
    answer_service = services.answer_services.get(request.mode.value)
    if answer_service is None:
        raise APIError(400, "invalid_mode", f"Unknown retrieval mode: {request.mode}")

    try:
        acl = get_user_acl(request.user) if request.user else None
    except KeyError as exc:
        raise APIError(400, "unknown_user", str(exc)) from exc

    started = time.perf_counter()
    with services.tracer.start_span("query", query_id=query_id, mode=request.mode.value):
        try:
            response = answer_service.answer(request.query, acl)
        except Exception as exc:  # noqa: BLE001
            services.metrics.record_request(success=False)
            raise APIError(502, "pipeline_error", str(exc)) from exc
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    services.metrics.record_request(success=True)
    services.metrics.observe_latency("query", latency_ms)
    logger.info(
        "query.completed",
        query_id=query_id,
        latency_ms=latency_ms,
        retrieval_count=len(response.sources),
        rerank_count=len(response.sources),
        token_usage=response.token_usage,
        confidence_score=response.confidence,
        refused=response.refused,
    )

    return QueryResponse(
        answer=response.answer,
        citations=response.citations,
        confidence=response.confidence,
        refused=response.refused,
        sources=[
            SourceItem(
                document_id=r.chunk.document_id,
                section_title=r.chunk.section_title,
                page_number=r.chunk.page_number,
                score=round(r.score, 4),
            )
            for r in response.sources
        ],
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id=r.chunk.chunk_id,
                document_id=r.chunk.document_id,
                text=r.chunk.text,
                score=round(r.score, 4),
            )
            for r in response.sources
        ],
        latency_ms=latency_ms,
    )
