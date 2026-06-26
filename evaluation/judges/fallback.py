"""Fallback judge wrapper.

Wraps a primary judge (typically :class:`LLMJudge`) and a fallback (typically
:class:`LexicalJudge`). If a primary call raises — model unreachable, network
error, parse failure surfaced as an exception — the call is retried on the
fallback, so an evaluation run never crashes because the LLM was unavailable.
This realizes the "LLM judge primary, lexical fallback" policy.
"""

from __future__ import annotations

from collections.abc import Sequence

from evaluation.judges.base import Judge, JudgeVerdict
from models.domain import Chunk
from utils.logging import get_logger

logger = get_logger(__name__)


class FallbackJudge(Judge):
    """Try the primary judge; fall back to the secondary on any error."""

    def __init__(self, primary: Judge, fallback: Judge) -> None:
        self._primary = primary
        self._fallback = fallback

    def _safe(self, method: str, *args: object) -> JudgeVerdict:
        try:
            return getattr(self._primary, method)(*args)
        except Exception as exc:  # noqa: BLE001 - any failure -> deterministic fallback
            logger.warning("judge.fallback", method=method, error=str(exc))
            return getattr(self._fallback, method)(*args)

    def faithfulness(self, answer: str, context: Sequence[Chunk]) -> JudgeVerdict:
        return self._safe("faithfulness", answer, context)

    def correctness(self, question: str, answer: str, ground_truth: str) -> JudgeVerdict:
        return self._safe("correctness", question, answer, ground_truth)

    def completeness(self, question: str, answer: str, ground_truth: str) -> JudgeVerdict:
        return self._safe("completeness", question, answer, ground_truth)

    def citation_support(self, claim: str, cited_text: str) -> JudgeVerdict:
        return self._safe("citation_support", claim, cited_text)
