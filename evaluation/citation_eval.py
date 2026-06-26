"""Citation evaluation: classify citations and derive an accuracy score."""
from __future__ import annotations
import re
from collections.abc import Sequence
from pydantic import BaseModel
from evaluation.judges.base import Judge
from generation.citations import extract_citation_indices, parse_citations
from models.domain import Chunk

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class CitationEvalResult(BaseModel):
    supported: int
    unsupported: int
    missing: int
    citation_accuracy_score: float


def evaluate_citations(answer: str, context: Sequence[Chunk], judge: Judge,
                       support_threshold: float = 0.5) -> CitationEvalResult:
    distinct_indices = list(dict.fromkeys(extract_citation_indices(answer)))
    valid = parse_citations(answer, context)
    valid_indices = {c.index for c in valid}
    unsupported = sum(1 for i in distinct_indices if i not in valid_indices)
    supported = 0
    for citation in valid:
        claim = _citing_sentence(answer, citation.index)
        verdict = judge.citation_support(claim, context[citation.index - 1].text)
        if verdict.score >= support_threshold:
            supported += 1
        else:
            unsupported += 1
    missing = _missing_sentences(answer, len(context))
    denom = supported + unsupported + missing
    score = 1.0 if denom == 0 else supported / denom
    return CitationEvalResult(supported=supported, unsupported=unsupported,
                              missing=missing, citation_accuracy_score=round(score, 4))


def _missing_sentences(answer: str, context_size: int) -> int:
    sentences = [s for s in _SENTENCE_RE.split(answer.strip()) if s.strip()]
    missing = 0
    for sentence in sentences:
        indices = extract_citation_indices(sentence)
        if not any(1 <= i <= context_size for i in indices):
            missing += 1
    return missing


def _citing_sentence(answer: str, index: int) -> str:
    marker = f"[{index}]"
    sentences = [s for s in _SENTENCE_RE.split(answer.strip()) if s.strip()]
    citing = [s for s in sentences if marker in s]
    return " ".join(citing) if citing else answer
