"""Tests for citation evaluation (supported / unsupported / missing)."""

from __future__ import annotations

from evaluation.citation_eval import evaluate_citations
from evaluation.judges.lexical_judge import LexicalJudge
from models.domain import Chunk


def _context() -> list[Chunk]:
    return [
        Chunk(chunk_id="a", document_id="d1", text="the api supports async requests"),
        Chunk(chunk_id="b", document_id="d2", text="rate limits are 100 per minute"),
    ]


def test_supported_citation() -> None:
    answer = "The API supports async requests [1]."
    result = evaluate_citations(answer, _context(), LexicalJudge())
    assert result.supported == 1
    assert result.unsupported == 0
    assert result.missing == 0
    assert result.citation_accuracy_score == 1.0


def test_hallucinated_citation_is_unsupported() -> None:
    answer = "Totally unrelated claim [9]."
    result = evaluate_citations(answer, _context(), LexicalJudge())
    # [9] is out of range -> unsupported, and the sentence has no valid cite -> missing.
    assert result.unsupported >= 1
    assert result.citation_accuracy_score < 1.0


def test_missing_citation_counts_uncited_claim() -> None:
    answer = "The API supports async requests [1]. This claim has no citation."
    result = evaluate_citations(answer, _context(), LexicalJudge())
    assert result.supported == 1
    assert result.missing == 1
    assert result.citation_accuracy_score == 0.5  # 1 supported / (1 + 0 + 1)


def test_no_citations_no_claims_is_vacuously_one() -> None:
    result = evaluate_citations("", _context(), LexicalJudge())
    assert result.citation_accuracy_score == 1.0
