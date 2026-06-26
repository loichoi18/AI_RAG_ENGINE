"""Golden dataset loading and the relevance predicate."""
from __future__ import annotations
import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from ingestion.segment import ACL_KEY
from models.domain import Chunk, Document

_DATA_DIR = Path(__file__).parent / "datasets"
CORPUS_PATH = _DATA_DIR / "corpus.json"
GOLDEN_PATH = _DATA_DIR / "golden.json"


class QuestionType(StrEnum):
    SIMPLE_LOOKUP = "simple_lookup"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"
    UNANSWERABLE = "unanswerable"


class CorpusDocument(BaseModel):
    document_id: str
    title: str
    acl: list[str] = Field(default_factory=list)
    text: str

    def to_document(self) -> Document:
        return Document(document_id=self.document_id, source_path=f"corpus://{self.document_id}",
                        content=self.text, metadata={ACL_KEY: self.acl, "title": self.title})


class GoldenExample(BaseModel):
    id: str
    query: str
    document_id: str
    answer_spans: list[str] = Field(default_factory=list)
    answer: str = ""
    acl: list[str] | None = None
    question_type: QuestionType = QuestionType.SIMPLE_LOOKUP
    expected_documents: list[str] = Field(default_factory=list)
    expected_chunks: list[str] = Field(default_factory=list)

    @property
    def question(self) -> str:
        return self.query

    @property
    def ground_truth(self) -> str:
        return self.answer

    @property
    def relevant_documents(self) -> list[str]:
        return self.expected_documents or [self.document_id]

    @property
    def is_answerable(self) -> bool:
        return self.question_type is not QuestionType.UNANSWERABLE


def load_corpus(path: str | Path = CORPUS_PATH) -> list[CorpusDocument]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CorpusDocument(**row) for row in data]


def load_corpus_documents(path: str | Path = CORPUS_PATH) -> list[Document]:
    return [doc.to_document() for doc in load_corpus(path)]


def load_golden(path: str | Path = GOLDEN_PATH) -> list[GoldenExample]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    records = data["examples"] if isinstance(data, dict) else data
    return [GoldenExample.model_validate(row) for row in records]


def is_relevant(chunk: Chunk, example: GoldenExample) -> bool:
    text = chunk.text.lower()
    if any(span.lower() in text for span in example.answer_spans):
        return True
    if chunk.document_id in example.relevant_documents:
        return True
    return chunk.chunk_id in example.expected_chunks


def relevance_flags(chunks: Sequence[Chunk], example: GoldenExample) -> list[bool]:
    return [is_relevant(chunk, example) for chunk in chunks]


def ranked_document_ids(chunks: Sequence[Chunk]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in chunks:
        if chunk.document_id not in seen:
            seen.add(chunk.document_id); ordered.append(chunk.document_id)
    return ordered
