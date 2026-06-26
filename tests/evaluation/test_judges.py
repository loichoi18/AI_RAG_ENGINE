"""Tests for the judge layer: lexical, LLM (faked), and fallback."""

from __future__ import annotations

from collections.abc import Sequence

from evaluation.judges.base import Judge, JudgeVerdict
from evaluation.judges.fallback import FallbackJudge
from evaluation.judges.lexical_judge import LexicalJudge
from evaluation.judges.llm_judge import LLMJudge
from generation.llm.base import Completion, LLMClient
from models.domain import Chunk


class _FakeLLMClient(LLMClient):
    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, system_prompt: str, user_prompt: str) -> Completion:
        return Completion(text=self._text, prompt_tokens=1, completion_tokens=1)


def test_llm_judge_parses_json_score() -> None:
    judge = LLMJudge(_FakeLLMClient('{"score": 0.8, "reasoning": "ok"}'))
    verdict = judge.correctness("q", "a", "gt")
    assert verdict.score == 0.8
    assert verdict.reasoning == "ok"


def test_llm_judge_handles_prose_wrapped_json() -> None:
    judge = LLMJudge(_FakeLLMClient('Sure! {"score": 1.0} hope that helps'))
    assert judge.faithfulness("a", []).score == 1.0


def test_llm_judge_defaults_zero_on_garbage() -> None:
    judge = LLMJudge(_FakeLLMClient("no json here"))
    assert judge.citation_support("claim", "src").score == 0.0


def test_lexical_judge_runs_without_model() -> None:
    judge = LexicalJudge()
    chunks = [Chunk(chunk_id="a", document_id="d", text="deploy with docker")]
    assert 0.0 <= judge.faithfulness("we deploy with docker", chunks).score <= 1.0
    assert judge.correctness("q", "deploy docker", "deploy docker").score == 1.0


class _RaisingJudge(Judge):
    def faithfulness(self, answer: str, context: Sequence[Chunk]) -> JudgeVerdict:
        raise RuntimeError("model down")

    def correctness(self, question: str, answer: str, ground_truth: str) -> JudgeVerdict:
        raise RuntimeError("model down")

    def completeness(self, question: str, answer: str, ground_truth: str) -> JudgeVerdict:
        raise RuntimeError("model down")

    def citation_support(self, claim: str, cited_text: str) -> JudgeVerdict:
        raise RuntimeError("model down")


def test_fallback_judge_uses_secondary_on_error() -> None:
    judge = FallbackJudge(_RaisingJudge(), LexicalJudge())
    # Primary raises; fallback (lexical) returns a valid verdict.
    verdict = judge.correctness("q", "deploy docker", "deploy docker")
    assert verdict.score == 1.0
