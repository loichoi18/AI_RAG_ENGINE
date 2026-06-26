"""Unit tests for citation parsing, validation, and coverage."""

from __future__ import annotations

from generation.citations import (
    citation_coverage,
    extract_citation_indices,
    parse_citations,
)
from models.domain import Chunk


def _context() -> list[Chunk]:
    return [
        Chunk(chunk_id="a", document_id="doc1", text="alpha", section_title="Intro"),
        Chunk(chunk_id="b", document_id="doc2", text="beta", page_number=2),
    ]


def test_extract_indices_in_order() -> None:
    assert extract_citation_indices("foo [2] bar [1] baz [2]") == [2, 1, 2]


def test_parse_maps_valid_and_drops_hallucinated() -> None:
    answer = "Alpha is true [1]. Also beta [2]. Bogus [9]."
    citations = parse_citations(answer, _context())

    assert [c.index for c in citations] == [1, 2]  # [9] dropped, distinct
    assert citations[0].chunk_id == "a"
    assert citations[1].chunk_id == "b"
    assert citations[1].page_number == 2


def test_parse_deduplicates() -> None:
    citations = parse_citations("x [1] y [1] z [1]", _context())
    assert len(citations) == 1


def test_coverage_all_sentences_cited() -> None:
    answer = "Alpha is true [1]. Beta holds [2]."
    assert citation_coverage(answer, context_size=2) == 1.0


def test_coverage_partial() -> None:
    answer = "Alpha is true [1]. This sentence is ungrounded."
    assert citation_coverage(answer, context_size=2) == 0.5


def test_coverage_ignores_hallucinated_citation() -> None:
    # [9] is out of range, so the sentence is not counted as covered.
    assert citation_coverage("Claim [9].", context_size=2) == 0.0


def test_coverage_empty_answer() -> None:
    assert citation_coverage("", context_size=2) == 0.0
