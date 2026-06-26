"""Tests for golden dataset loading and the relevance predicate."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.dataset import (
    GoldenExample,
    QuestionType,
    is_relevant,
    load_golden,
    ranked_document_ids,
)
from models.domain import Chunk


def test_golden_example_aliases_and_defaults() -> None:
    ex = GoldenExample(id="q1", query="What is X?", document_id="doc1", answer="X is Y")
    assert ex.question == "What is X?"  # alias for query
    assert ex.ground_truth == "X is Y"  # alias for answer
    assert ex.question_type is QuestionType.SIMPLE_LOOKUP
    assert ex.relevant_documents == ["doc1"]  # falls back to document_id
    assert ex.is_answerable is True


def test_unanswerable_flag() -> None:
    ex = GoldenExample(
        id="q2", query="?", document_id="", question_type=QuestionType.UNANSWERABLE
    )
    assert ex.is_answerable is False


def test_load_golden_json(tmp_path: Path) -> None:
    data = [
        {"id": "q1", "query": "a", "document_id": "d1", "answer_spans": ["x"], "answer": "x"},
        {"id": "q2", "query": "b", "document_id": "d2", "answer_spans": ["y"], "answer": "y"},
    ]
    f = tmp_path / "golden.json"
    f.write_text(json.dumps(data), encoding="utf-8")

    examples = load_golden(f)
    assert [e.id for e in examples] == ["q1", "q2"]


def test_load_golden_yaml(tmp_path: Path) -> None:
    f = tmp_path / "golden.yaml"
    f.write_text(
        "examples:\n"
        "  - id: q1\n    query: a\n    document_id: d1\n    answer_spans: [x]\n    answer: x\n",
        encoding="utf-8",
    )
    examples = load_golden(f)
    assert examples[0].id == "q1"


def test_is_relevant_span_and_document() -> None:
    ex = GoldenExample(
        id="q", query="q", document_id="doc1", answer_spans=["async"], answer="a"
    )
    by_span = Chunk(chunk_id="a", document_id="other", text="it is async here")
    by_doc = Chunk(chunk_id="b", document_id="doc1", text="unrelated text")
    irrelevant = Chunk(chunk_id="c", document_id="other", text="unrelated text")

    assert is_relevant(by_span, ex) is True
    assert is_relevant(by_doc, ex) is True
    assert is_relevant(irrelevant, ex) is False


def test_ranked_document_ids_dedup() -> None:
    chunks = [
        Chunk(chunk_id="1", document_id="d1", text="t"),
        Chunk(chunk_id="2", document_id="d1", text="t"),
        Chunk(chunk_id="3", document_id="d2", text="t"),
    ]
    assert ranked_document_ids(chunks) == ["d1", "d2"]
