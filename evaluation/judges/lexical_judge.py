"""Deterministic lexical judge.

Implements the :class:`Judge` interface with the reproducible lexical metrics
from :mod:`evaluation.metrics.generation`. Requires no model or API key, so it
is the CI default and the fallback when an LLM judge is unavailable.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from evaluation.judges.base import Judge, JudgeVerdict
from evaluation.metrics.generation import (
    answer_completeness,
    faithfulness as lexical_faithfulness,
    token_f1,
)
from models.domain import Chunk

_WORD_RE = re.compile(r"[a-z0-9]+")


def _overlap(claim: str, cited_text: str) -> float:
    claim_words = set(_WORD_RE.findall(claim.lower()))
    cited_words = set(_WORD_RE.findall(cited_text.lower()))
    if not claim_words:
        return 0.0
    return len(claim_words & cited_words) / len(claim_words)


class LexicalJudge(Judge):
    """Judge backed by deterministic lexical overlap metrics."""

    def __init__(self, support_threshold: float = 0.5) -> None:
        self._support_threshold = support_threshold

    def faithfulness(self, answer: str, context: Sequence[Chunk]) -> JudgeVerdict:
        return JudgeVerdict(score=lexical_faithfulness(answer, context), reasoning="lexical")

    def correctness(self, question: str, answer: str, ground_truth: str) -> JudgeVerdict:
        return JudgeVerdict(score=token_f1(answer, ground_truth), reasoning="lexical token-F1")

    def completeness(self, question: str, answer: str, ground_truth: str) -> JudgeVerdict:
        return JudgeVerdict(
            score=answer_completeness(answer, ground_truth), reasoning="lexical coverage"
        )

    def citation_support(self, claim: str, cited_text: str) -> JudgeVerdict:
        overlap = _overlap(claim, cited_text)
        return JudgeVerdict(score=1.0 if overlap >= self._support_threshold else overlap)
