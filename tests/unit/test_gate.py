"""Unit tests for the confidence gate."""

from __future__ import annotations

import pytest

from generation.gate import ScoreThresholdGate
from models.domain import Chunk, RetrievalResult


def _results(*scores: float) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=Chunk(chunk_id=str(i), document_id="d", text="t"),
            score=s,
            retriever_name="cross_encoder",
        )
        for i, s in enumerate(scores)
    ]


def test_gate_passes_above_threshold() -> None:
    gate = ScoreThresholdGate(threshold=0.5)
    assert gate.passes(_results(0.2, 0.7)) is True


def test_gate_fails_below_threshold() -> None:
    gate = ScoreThresholdGate(threshold=0.5)
    assert gate.passes(_results(0.2, 0.4)) is False


def test_gate_fails_on_empty() -> None:
    assert ScoreThresholdGate(threshold=0.5).passes([]) is False


def test_gate_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        ScoreThresholdGate(threshold=1.5)
