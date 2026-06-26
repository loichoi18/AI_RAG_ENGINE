"""Qdrant-backed implementation of the VectorStore contract."""
from __future__ import annotations
import uuid
from collections.abc import Mapping, Sequence
from typing import Any
from qdrant_client import QdrantClient, models
from models.domain import Chunk, ChunkStrategy, RetrievalResult
from storage.base import VectorStore
from utils.logging import get_logger

logger = get_logger(__name__)
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


class QdrantVectorStore(VectorStore):
    def __init__(self, collection_name: str, vector_size: int, *, client: QdrantClient | None = None,
                 host: str = "localhost", port: int = 6333,
                 api_key: str | None = None, https: bool = False,
                 distance: models.Distance = models.Distance.COSINE) -> None:
        # api_key + https enable managed Qdrant Cloud clusters (TLS + auth).
        self._client = client or QdrantClient(host=host, port=port, api_key=api_key, https=https)
        self._collection = collection_name
        self._vector_size = vector_size
        self._distance = distance

    @property
    def collection_name(self) -> str:
        return self._collection

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(collection_name=self._collection,
            vectors_config=models.VectorParams(size=self._vector_size, distance=self._distance))
        self._client.create_payload_index(collection_name=self._collection, field_name="acl",
            field_schema=models.PayloadSchemaType.KEYWORD)
        logger.info("qdrant.collection_created", collection=self._collection)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have equal length")
        if not chunks:
            return
        points = [models.PointStruct(id=_point_id(c.chunk_id), vector=list(v), payload=self._to_payload(c))
                  for c, v in zip(chunks, vectors, strict=True)]
        self._client.upsert(collection_name=self._collection, points=points)
        logger.info("qdrant.upsert", collection=self._collection, count=len(points))

    def search(self, query_vector: Sequence[float], top_k: int, acl: Sequence[str] | None = None,
               metadata_filter: Mapping[str, Any] | None = None,
               score_threshold: float | None = None) -> list[RetrievalResult]:
        hits = self._client.query_points(collection_name=self._collection, query=list(query_vector),
            limit=top_k, query_filter=self._build_filter(acl, metadata_filter),
            score_threshold=score_threshold, with_payload=True).points
        return [RetrievalResult(chunk=self._from_payload(h.payload or {}), score=float(h.score),
                                retriever_name="vector_store") for h in hits]

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        self._client.delete(collection_name=self._collection,
            points_selector=models.PointIdsList(points=[_point_id(c) for c in chunk_ids]))
        logger.info("qdrant.delete", collection=self._collection, count=len(chunk_ids))

    def count(self) -> int:
        return int(self._client.count(collection_name=self._collection).count)

    def scroll_all(self, batch_size: int = 256) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset: object | None = None
        while True:
            points, offset = self._client.scroll(collection_name=self._collection, limit=batch_size,
                offset=offset, with_payload=True, with_vectors=False)
            chunks.extend(self._from_payload(p.payload or {}) for p in points)
            if offset is None:
                break
        return chunks

    @staticmethod
    def _to_payload(chunk: Chunk) -> dict[str, object]:
        return {"chunk_id": chunk.chunk_id, "document_id": chunk.document_id, "text": chunk.text,
                "page_number": chunk.page_number, "section_title": chunk.section_title,
                "chunk_strategy": chunk.chunk_strategy.value, "token_count": chunk.token_count,
                "acl": chunk.acl, "metadata": chunk.metadata}

    @staticmethod
    def _from_payload(payload: dict[str, object]) -> Chunk:
        return Chunk(chunk_id=str(payload["chunk_id"]), document_id=str(payload["document_id"]),
                     text=str(payload["text"]), page_number=payload.get("page_number"),
                     section_title=payload.get("section_title"),
                     chunk_strategy=ChunkStrategy(payload.get("chunk_strategy", "recursive")),
                     token_count=int(payload.get("token_count", 0)),
                     acl=list(payload.get("acl", []) or []), metadata=dict(payload.get("metadata", {}) or {}))

    @staticmethod
    def _acl_subfilter(acl: Sequence[str]) -> models.Filter:
        return models.Filter(should=[models.IsEmptyCondition(is_empty=models.PayloadField(key="acl")),
                                     models.FieldCondition(key="acl", match=models.MatchAny(any=list(acl)))])

    @classmethod
    def _build_filter(cls, acl: Sequence[str] | None, metadata_filter: Mapping[str, Any] | None) -> models.Filter | None:
        must: list[models.Condition] = []
        if metadata_filter:
            must.extend(models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in metadata_filter.items())
        if acl is not None:
            must.append(cls._acl_subfilter(acl))
        if not must:
            return None
        return models.Filter(must=must)
