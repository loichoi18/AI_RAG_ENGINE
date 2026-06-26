"""Unit tests for the grounded generator (LLM client faked)."""

from __future__ import annotations

from generation.generator import GroundedGenerator
from generation.llm.base import Completion, LLMClient
from models.domain import Chunk


class FakeLLMClient(LLMClient):
    """Returns canned text and records the prompts it received."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0
        self.last_user_prompt = ""

    def complete(self, system_prompt: str, user_prompt: str) -> Completion:
        self.calls += 1
        self.last_user_prompt = user_prompt
        return Completion(text=self._text, prompt_tokens=10, completion_tokens=5)


def _context() -> list[Chunk]:
    return [
        Chunk(chunk_id="a", document_id="doc1", text="alpha"),
        Chunk(chunk_id="b", document_id="doc2", text="beta"),
    ]


def test_generator_returns_answer_with_citations() -> None:
    client = FakeLLMClient("Alpha is true [1]. Beta also [2].")
    result = GroundedGenerator(client).generate("q", _context())

    assert "Alpha is true" in result.answer
    assert result.citations == ["a", "b"]
    assert result.token_usage == {"prompt": 10, "completion": 5}
    assert client.calls == 1


def test_generator_drops_hallucinated_citation() -> None:
    client = FakeLLMClient("Claim [1]. Bogus [9].")
    result = GroundedGenerator(client).generate("q", _context())
    assert result.citations == ["a"]  # [9] is invalid


def test_generator_refuses_on_sentinel() -> None:
    client = FakeLLMClient("INSUFFICIENT_CONTEXT")
    result = GroundedGenerator(client).generate("q", _context())
    assert result.answer == ""
    assert result.citations == []


def test_generator_refuses_on_empty_context() -> None:
    client = FakeLLMClient("should not be called")
    result = GroundedGenerator(client).generate("q", [])
    assert result.answer == ""
    assert client.calls == 0  # LLM not invoked without context
