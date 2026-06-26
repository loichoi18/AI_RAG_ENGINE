"""LLM-as-judge.

Scores answers by prompting an LLM (via the Sprint-4 :class:`LLMClient`, which
already abstracts Ollama / OpenAI / Anthropic). Each method asks the model for a
strict JSON object ``{"score": 0..1, "reasoning": "..."}`` and parses it
defensively (the model may wrap JSON in prose). Scores are clamped to [0, 1].

This is the primary judge in production; pair it with the lexical judge via
:class:`~evaluation.judges.fallback.FallbackJudge` so evaluation still runs when
no model is reachable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from evaluation.judges.base import Judge, JudgeVerdict
from generation.llm.base import LLMClient
from models.domain import Chunk

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = (
    "You are a strict evaluation judge. Respond ONLY with a JSON object of the "
    'form {"score": <float 0..1>, "reasoning": "<short>"}. Do not add prose.'
)


def _parse_verdict(text: str) -> JudgeVerdict:
    """Extract a verdict from model output, defaulting to 0.0 on parse failure."""
    match = _JSON_RE.search(text)
    if not match:
        return JudgeVerdict(score=0.0, reasoning="unparseable judge output")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return JudgeVerdict(score=0.0, reasoning="invalid judge JSON")
    score = float(data.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    return JudgeVerdict(score=score, reasoning=str(data.get("reasoning", "")))


class LLMJudge(Judge):
    """Judge that scores answers by prompting an LLM."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def _ask(self, instruction: str) -> JudgeVerdict:
        completion = self._client.complete(_SYSTEM, instruction)
        return _parse_verdict(completion.text)

    def faithfulness(self, answer: str, context: Sequence[Chunk]) -> JudgeVerdict:
        context_text = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(context, 1))
        return self._ask(
            "Score how fully the ANSWER is grounded in the CONTEXT (1.0 = every "
            "claim is supported, 0.0 = unsupported/hallucinated).\n\n"
            f"CONTEXT:\n{context_text}\n\nANSWER:\n{answer}"
        )

    def correctness(self, question: str, answer: str, ground_truth: str) -> JudgeVerdict:
        return self._ask(
            "Score how factually correct the ANSWER is relative to the REFERENCE "
            "for the QUESTION (1.0 = fully correct, 0.0 = wrong).\n\n"
            f"QUESTION:\n{question}\n\nREFERENCE:\n{ground_truth}\n\nANSWER:\n{answer}"
        )

    def completeness(self, question: str, answer: str, ground_truth: str) -> JudgeVerdict:
        return self._ask(
            "Score how completely the ANSWER covers the information in the "
            "REFERENCE for the QUESTION (1.0 = nothing important missing, 0.0 = "
            "misses everything).\n\n"
            f"QUESTION:\n{question}\n\nREFERENCE:\n{ground_truth}\n\nANSWER:\n{answer}"
        )

    def citation_support(self, claim: str, cited_text: str) -> JudgeVerdict:
        return self._ask(
            "Score whether the SOURCE supports the CLAIM (1.0 = directly supports, "
            "0.0 = unrelated/contradicts).\n\n"
            f"CLAIM:\n{claim}\n\nSOURCE:\n{cited_text}"
        )
