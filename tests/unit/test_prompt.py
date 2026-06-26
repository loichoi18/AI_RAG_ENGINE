"""Unit tests for the grounded prompt builder."""

from __future__ import annotations

from generation.prompt import INSUFFICIENT_CONTEXT, GroundedPromptBuilder
from models.domain import Chunk


def test_system_prompt_states_rules() -> None:
    system = GroundedPromptBuilder().system_prompt
    assert "ONLY" in system
    assert INSUFFICIENT_CONTEXT in system
    assert "cite" in system.lower()


def test_user_prompt_numbers_blocks_and_includes_sources() -> None:
    chunks = [
        Chunk(chunk_id="a", document_id="doc1", text="alpha fact", section_title="Intro"),
        Chunk(chunk_id="b", document_id="doc2", text="beta fact", page_number=4),
    ]
    user = GroundedPromptBuilder().build_user_prompt("what is alpha?", chunks)

    assert "[1]" in user and "[2]" in user
    assert "alpha fact" in user and "beta fact" in user
    assert "doc1 / Intro" in user
    assert "p.4" in user
    assert "what is alpha?" in user


def test_user_prompt_handles_no_context() -> None:
    user = GroundedPromptBuilder().build_user_prompt("q", [])
    assert "no context provided" in user
