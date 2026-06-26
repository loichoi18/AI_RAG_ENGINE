"""Unit tests for retrieval and generation metrics (hand-computed values)."""

from __future__ import annotations

import math

from evaluation.metrics.generation import answer_completeness
from evaluation.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k() -> None:
    # 2 of 3 relevant docs appear in the top 3.
    rel = [True, False, True, False]
    assert recall_at_k(rel, total_relevant=3, k=3) == 2 / 3


def test_recall_empty_relevant() -> None:
    assert recall_at_k([False, False], total_relevant=0, k=2) == 0.0


def test_precision_at_k() -> None:
    rel = [True, False, True, True]
    assert precision_at_k(rel, k=4) == 0.75
    assert precision_at_k(rel, k=2) == 0.5


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([False, False, True]) == 1 / 3
    assert reciprocal_rank([True, False]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    # Two relevant items placed first => DCG == IDCG => 1.0
    assert ndcg_at_k([True, True, False], total_relevant=2, k=3) == 1.0


def test_ndcg_at_k_known_value() -> None:
    # One relevant item at rank 2: DCG = 1/log2(3); IDCG = 1/log2(2) = 1.
    rel = [False, True, False]
    expected = (1.0 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_at_k(rel, total_relevant=1, k=3), expected, rel_tol=1e-9)


def test_hit_rate_at_k() -> None:
    assert hit_rate_at_k([False, True, False], k=3) == 1.0
    assert hit_rate_at_k([False, True], k=1) == 0.0
    assert hit_rate_at_k([False, False], k=5) == 0.0


def test_answer_completeness() -> None:
    # Ground-truth content words: {deploy, docker}; answer covers both.
    assert answer_completeness("deploy docker", "deploy docker") == 1.0
    # Covers one of the two content words.
    assert answer_completeness("deploy", "deploy docker") == 0.5
