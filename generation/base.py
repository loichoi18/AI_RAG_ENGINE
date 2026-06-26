"""Abstract interfaces for the generation layer.

Two contracts live here:

* :class:`LLMProvider` — turns a query plus retrieved context into a grounded,
  cited answer. Concrete providers (Ollama, OpenAI, Anthropic) implement this
  behind one interface so the active provider is a configuration choice.
* :class:`ConfidenceGate` — the hallucination guardrail. It inspects the
  retrieved/reranked context and decides whether generation should proceed at
  all. Modeling refusal as its own contract keeps "never hallucinate" an
  architectural property rather than a prompt-only hope.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from models.domain import Chunk, GenerationResult, RetrievalResult


class LLMProvider(ABC):
    """Generates a grounded answer from retrieved context."""

    @abstractmethod
    def generate(self, query: str, context: Sequence[Chunk]) -> GenerationResult:
        """Produce an answer to ``query`` using only ``context``.

        Implementations MUST:

        * answer strictly from the provided context;
        * include inline citations referencing the supporting chunks;
        * report token usage in the returned :class:`GenerationResult`.

        Parameters
        ----------
        query:
            The user's question.
        context:
            Chunks selected by retrieval/reranking, in priority order.
        """
        raise NotImplementedError


class ConfidenceGate(ABC):
    """Decides whether retrieved context is strong enough to answer."""

    @abstractmethod
    def passes(self, results: Sequence[RetrievalResult]) -> bool:
        """Return ``True`` if generation should proceed.

        A ``False`` result triggers a refusal upstream (e.g. "I don't know —
        see these documents"), preventing low-confidence hallucination.

        Parameters
        ----------
        results:
            Reranked retrieval results whose scores inform the decision.
        """
        raise NotImplementedError
