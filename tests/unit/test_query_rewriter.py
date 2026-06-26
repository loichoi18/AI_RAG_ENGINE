"""Unit tests for the heuristic query rewriter."""

from __future__ import annotations

from services.query_rewriting import HeuristicQueryRewriter


def test_rewrite_strips_interrogative_and_punctuation() -> None:
    rewriter = HeuristicQueryRewriter()
    assert rewriter.rewrite_query("How do we deploy services?") == "deploy services"


def test_rewrite_handles_plain_query() -> None:
    rewriter = HeuristicQueryRewriter()
    assert rewriter.rewrite_query("docker deployment") == "docker deployment"


def test_expand_includes_normalized_and_synonyms() -> None:
    rewriter = HeuristicQueryRewriter()
    variants = rewriter.expand_query("How do we deploy services?")

    assert variants[0] == "deploy services"
    assert any("rollout" in v or "deployment" in v for v in variants)
    assert len(variants) <= 4


def test_expand_respects_custom_synonyms() -> None:
    rewriter = HeuristicQueryRewriter(synonyms={"kafka": ["event streaming", "message bus"]})
    variants = rewriter.expand_query("what is kafka")
    assert "event streaming" in variants
