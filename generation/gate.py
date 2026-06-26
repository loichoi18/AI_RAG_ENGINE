"""Confidence gate (hallucination guardrail).

Inspects the reranked candidates and decides whether generation may proceed. If
the top candidate's normalized score is below the threshold, the gate fails and
the caller returns a refusal *without ever invoking the LLM*. This turns "never
hallucinate on weak context" into an architectural guarantee and avoids spending
tokens on unanswerable queries.
"""

from __future__ import annotations

from collections.abc import Sequence

from generation.base import ConfidenceGate
from models.domain import RetrievalResult


class ScoreThresholdGate(ConfidenceGate):
    """Passes only when the best reranked score meets a threshold."""

    def __init__(self, threshold: float = 0.3) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """The minimum top score required to pass."""
        return self._threshold

    def passes(self, results: Sequence[RetrievalResult]) -> bool:
        """Return ``True`` if the top result's score meets the threshold."""
        if not results:
            return False
        return max(r.score for r in results) >= self._threshold
