"""API request/response schemas.

Kept separate from :mod:`models.domain`: domain models are internal data
contracts, while these are the external HTTP wire format. Separating them lets
the API evolve (versioning, field renaming) without disturbing core types.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response body for the health endpoints."""

    status: str = "ok"


class RetrievalMode(StrEnum):
    """Which retriever the query endpoint should use."""

    HYBRID = "hybrid"
    DENSE = "dense"


class ErrorResponse(BaseModel):
    """Structured error body returned for all handled exceptions."""

    error: str
    detail: str
    request_id: str | None = None


# --- /v1/query -------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request body for ``POST /v1/query``."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: RetrievalMode = RetrievalMode.HYBRID
    user: str | None = Field(
        default=None,
        description="Mock user for ACL scoping: admin / engineering / hr. None = admin.",
    )


class SourceItem(BaseModel):
    """A source document surfaced for an answer."""

    document_id: str
    section_title: str | None = None
    page_number: int | None = None
    score: float


class RetrievedChunk(BaseModel):
    """A retrieved chunk returned for transparency/debugging."""

    chunk_id: str
    document_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    """Response body for ``POST /v1/query``."""

    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence: float
    refused: bool = False
    sources: list[SourceItem] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float


# --- /v1/ingest ------------------------------------------------------------


class IngestDocument(BaseModel):
    """A single document to ingest."""

    document_id: str
    text: str = Field(min_length=1)
    title: str | None = None
    acl: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    """Request body for ``POST /v1/ingest``."""

    documents: list[IngestDocument] = Field(min_length=1)


class IngestResponse(BaseModel):
    """Response body for ``POST /v1/ingest``."""

    ingested_documents: int
    indexed_chunks: int


# --- /v1/documents ---------------------------------------------------------


class DocumentInfo(BaseModel):
    """Summary of an indexed document."""

    document_id: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    """Response body for ``GET /v1/documents``."""

    documents: list[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    """Response body for ``DELETE /v1/documents/{id}``."""

    document_id: str
    deleted_chunks: int


# --- /v1/metrics -----------------------------------------------------------


class MetricsResponse(BaseModel):
    """Response body for ``GET /v1/metrics`` (free-form metrics snapshot)."""

    counters: dict[str, int] = Field(default_factory=dict)
    success_rate: float = 0.0
    error_rate: float = 0.0
    latency_ms: dict[str, dict[str, float]] = Field(default_factory=dict)
