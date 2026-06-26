"""Document endpoints: ingest, list, delete.

* ``POST /v1/ingest`` — chunk, embed, and index inline documents, then rebuild
  the BM25 index so hybrid retrieval sees the new content immediately.
* ``GET /v1/documents`` — list indexed documents with chunk counts.
* ``DELETE /v1/documents/{id}`` — remove all chunks for a document.
"""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends

from api.dependencies import get_services
from api.errors import APIError
from api.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentsResponse,
    IngestRequest,
    IngestResponse,
)
from api.services import AppServices
from ingestion.segment import ACL_KEY
from models.domain import Document
from utils.logging import get_logger

router = APIRouter(prefix="/v1", tags=["documents"])
logger = get_logger("api.documents")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest, services: AppServices = Depends(get_services)
) -> IngestResponse:
    """Ingest inline documents into the vector store."""
    try:
        services.store.ensure_collection()
        total_chunks = 0
        for doc in request.documents:
            document = Document(
                document_id=doc.document_id,
                source_path=f"api://{doc.document_id}",
                content=doc.text,
                metadata={ACL_KEY: doc.acl, "title": doc.title},
            )
            chunks = services.chunker.chunk(document)
            if not chunks:
                continue
            vectors = services.embedder.embed([c.text for c in chunks])
            services.store.upsert(chunks, vectors)
            total_chunks += len(chunks)
        services.reindex_sparse()
    except APIError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise APIError(502, "ingest_error", str(exc)) from exc

    logger.info("ingest.completed", documents=len(request.documents), chunks=total_chunks)
    return IngestResponse(ingested_documents=len(request.documents), indexed_chunks=total_chunks)


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(services: AppServices = Depends(get_services)) -> DocumentsResponse:
    """List indexed documents and their chunk counts."""
    try:
        chunks = services.store.scroll_all()
    except Exception as exc:  # noqa: BLE001
        raise APIError(502, "store_error", str(exc)) from exc
    counts = Counter(c.document_id for c in chunks)
    docs = [DocumentInfo(document_id=d, chunk_count=n) for d, n in sorted(counts.items())]
    return DocumentsResponse(documents=docs, total=len(docs))


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str, services: AppServices = Depends(get_services)
) -> DeleteResponse:
    """Delete all chunks belonging to a document."""
    try:
        chunk_ids = [c.chunk_id for c in services.store.scroll_all() if c.document_id == document_id]
        services.store.delete(chunk_ids)
        services.reindex_sparse()
    except Exception as exc:  # noqa: BLE001
        raise APIError(502, "delete_error", str(exc)) from exc
    return DeleteResponse(document_id=document_id, deleted_chunks=len(chunk_ids))
