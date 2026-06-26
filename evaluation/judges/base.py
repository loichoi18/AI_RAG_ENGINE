"""LLM-as-judge interface.

A :class:`Judge` scores a generated answer along four axes — faithfulness,
correctness, completeness, and per-citation support. Keeping it an interface lets
us swap an LLM-backed judge (Ollama/OpenAI/Anthropic) for a deterministic lexical
judge (CI, reproducibility) or an external framework (Ragas, etc.) without
touching the evaluators that depend on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, Field

from models.domain import Chunk


class JudgeVerdict(BaseModel):
    """A single judgement: a normalized score with optional reasoning."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class Judge(ABC):
    """Scores generated answers against context and ground truth."""

    @abstractmethod
    def faithfulness(self, answer: str, context: Sequence[Chunk]) -> JudgeVerdict:
        """Are all claims in ``answer`` grounded in ``context``?"""
        raise NotImplementedError

    @abstractmethod
    def correctness(self, question: str, answer: str, ground_truth: str) -> JudgeVerdict:
        """Does ``answer`` match the ``ground_truth`` for ``question``?"""
        raise NotImplementedError

    @abstractmethod
    def completeness(self, question: str, answer: str, ground_truth: str) -> JudgeVerdict:
        """Does ``answer`` cover everything the ``ground_truth`` contains?"""
        raise NotImplementedError

    @abstractmethod
    def citation_support(self, claim: str, cited_text: str) -> JudgeVerdict:
        """Does ``cited_text`` actually support ``claim``? (score ~ 1.0 if yes)"""
        raise NotImplementedError
