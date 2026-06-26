"""Extractive generator — a deterministic, offline baseline ``LLMProvider``.

Instead of calling an LLM, it selects the context sentence with the highest
lexical overlap with the query and returns it with a citation to its source
block. This makes the full generation pipeline — including faithfulness and
citation metrics — runnable with no model or API key, and provides a baseline to
compare a real LLM against. By construction its answers are grounded and cited.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from generation.base import LLMProvider
from models.domain import Chunk, GenerationResult

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


class ExtractiveGenerator(LLMProvider):
    """Returns the most query-relevant context sentence, with a citation."""

    def generate(self, query: str, context: Sequence[Chunk]) -> GenerationResult:
        """Extract and cite the best-matching sentence from the context."""
        if not context:
            return GenerationResult(answer="", citations=[], confidence=0.0, token_usage={})

        query_words = _words(query)
        best_score = -1.0
        best_sentence = ""
        best_index = 0

        for index, chunk in enumerate(context, start=1):
            for sentence in _SENTENCE_RE.split(chunk.text.strip()):
                if not sentence.strip():
                    continue
                overlap = len(query_words & _words(sentence))
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sentence.strip()
                    best_index = index

        if not best_sentence or best_score <= 0:
            return GenerationResult(answer="", citations=[], confidence=0.0, token_usage={})

        answer = f"{best_sentence} [{best_index}]"
        return GenerationResult(
            answer=answer,
            citations=[context[best_index - 1].chunk_id],
            confidence=0.0,
            token_usage={"prompt": 0, "completion": 0},
        )
