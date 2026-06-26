"""Abstract interface for the evaluation layer.

The :class:`Evaluator` contract computes metrics over a labeled dataset. Both
retrieval metrics (Recall@k, MRR, nDCG) and generation metrics (faithfulness,
citation accuracy, answer correctness) are produced by implementations of this
single contract so they can be run from one harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


class Evaluator(ABC):
    """Computes evaluation metrics over a dataset of examples."""

    @abstractmethod
    def evaluate(self, dataset: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Evaluate the pipeline against ``dataset``.

        Parameters
        ----------
        dataset:
            A sequence of labeled examples. The exact schema is defined by each
            evaluator (e.g. retrieval examples carry ``query`` and
            ``relevant_chunk_ids``; generation examples add ``ground_truth``).

        Returns
        -------
        dict[str, float]
            Metric name -> aggregate score.
        """
        raise NotImplementedError
