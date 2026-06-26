"""Grounded generator.

Implements the :class:`~generation.base.LLMProvider` contract by composing a
thin :class:`~generation.llm.base.LLMClient` with the grounded prompt builder and
the citation parser. It owns all grounding logic so backends stay thin:

1. build the system + numbered-context user prompt;
2. call the client;
3. if the model emitted the refusal sentinel, return an empty, zero-confidence
   result (the caller renders the refusal message);
4. otherwise parse and validate citations against the provided context.

Confidence is intentionally NOT set here (it depends on rerank scores the
generator does not see); the :class:`~services.answer_service.AnswerService`
blends rerank score with citation coverage.
"""

from __future__ import annotations

from collections.abc import Sequence

from generation.base import LLMProvider
from generation.citations import parse_citations
from generation.llm.base import LLMClient
from generation.prompt import INSUFFICIENT_CONTEXT, GroundedPromptBuilder
from models.domain import Chunk, GenerationResult
from utils.logging import get_logger

logger = get_logger(__name__)


class GroundedGenerator(LLMProvider):
    """Generates grounded, cited answers from retrieved context."""

    def __init__(
        self,
        client: LLMClient,
        prompt_builder: GroundedPromptBuilder | None = None,
    ) -> None:
        self._client = client
        self._prompt = prompt_builder or GroundedPromptBuilder()

    def generate(self, query: str, context: Sequence[Chunk]) -> GenerationResult:
        """Produce a grounded answer with validated citations."""
        if not context:
            return self._refusal()

        user_prompt = self._prompt.build_user_prompt(query, context)
        completion = self._client.complete(self._prompt.system_prompt, user_prompt)
        answer = completion.text.strip()

        token_usage = {
            "prompt": completion.prompt_tokens,
            "completion": completion.completion_tokens,
        }

        if INSUFFICIENT_CONTEXT in answer:
            logger.info("generation.refused", reason="sentinel")
            return GenerationResult(
                answer="", citations=[], confidence=0.0, token_usage=token_usage
            )

        citations = parse_citations(answer, context)
        logger.info(
            "generation.answer",
            citations=len(citations),
            tokens=completion.total_tokens,
        )
        return GenerationResult(
            answer=answer,
            citations=[c.chunk_id for c in citations],
            confidence=0.0,  # set by AnswerService from rerank score x coverage
            token_usage=token_usage,
        )

    @staticmethod
    def _refusal() -> GenerationResult:
        return GenerationResult(answer="", citations=[], confidence=0.0, token_usage={})
