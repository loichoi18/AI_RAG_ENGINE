"""Generation evaluator.

Runs the answer pipeline over the golden set and aggregates faithfulness,
answer correctness, answer completeness, and citation accuracy, plus the refusal
rate and (for unanswerable questions) whether the system correctly declined.

Quality metrics are averaged over **answered** examples; the refusal rate and
unanswerable handling are reported separately. Scoring goes through the
:class:`Judge` interface — an LLM judge in production (with a lexical fallback),
or a deterministic :class:`LexicalJudge` by default so the class runs in CI
without a model.
"""

from __future__ import annotations

from collections.abc import Sequence

from evaluation.citation_eval import evaluate_citations
from evaluation.dataset import GoldenExample, QuestionType
from evaluation.judges.base import Judge
from evaluation.judges.lexical_judge import LexicalJudge
from models.domain import Chunk
from services.answer_service import AnswerService
from utils.logging import get_logger

logger = get_logger(__name__)


class GenerationEvaluator:
    """Computes generation metrics for an answer service over a golden dataset."""

    def __init__(
        self,
        answer_service: AnswerService,
        judge: Judge | None = None,
        support_threshold: float = 0.5,
    ) -> None:
        self._service = answer_service
        self._judge = judge or LexicalJudge()
        self._support_threshold = support_threshold

    def evaluate(self, dataset: Sequence[GoldenExample]) -> dict[str, float]:
        """Return mean generation metrics across ``dataset``."""
        rows = self.evaluate_examples(dataset)

        answered = [r for r in rows if r["answered"]]
        unanswerable = [r for r in rows if r["unanswerable"]]
        refusals = sum(1 for r in rows if r["refused"])

        result = {
            "faithfulness": _mean([r["faithfulness"] for r in answered]),
            "answer_correctness": _mean([r["answer_correctness"] for r in answered]),
            "answer_completeness": _mean([r["answer_completeness"] for r in answered]),
            "citation_accuracy": _mean([r["citation_accuracy"] for r in answered]),
            "refusal_rate": refusals / len(rows) if rows else 0.0,
            "unanswerable_accuracy": _mean([r["unanswerable_correct"] for r in unanswerable]),
            "answered": float(len(answered)),
        }
        logger.info(
            "eval.generation",
            examples=len(rows),
            **{k: round(v, 4) for k, v in result.items()},
        )
        return result

    def evaluate_examples(self, dataset: Sequence[GoldenExample]) -> list[dict[str, float]]:
        """Per-example generation metrics (for detailed reports)."""
        rows: list[dict[str, float]] = []
        for example in dataset:
            rows.append(self._evaluate_one(example))
        return rows

    def _evaluate_one(self, example: GoldenExample) -> dict[str, float]:
        response = self._service.answer(example.query, example.acl)
        row: dict[str, float] = {
            "answered": 0.0,
            "refused": 1.0 if response.refused else 0.0,
            "unanswerable": 0.0,
            "unanswerable_correct": 0.0,
            "faithfulness": 0.0,
            "answer_correctness": 0.0,
            "answer_completeness": 0.0,
            "citation_accuracy": 0.0,
            "confidence": response.confidence,
        }

        # Unanswerable: the correct behavior is to refuse.
        if example.question_type is QuestionType.UNANSWERABLE:
            row["unanswerable"] = 1.0
            row["unanswerable_correct"] = 1.0 if response.refused else 0.0
            return row

        if response.refused:
            return row

        context: list[Chunk] = [r.chunk for r in response.sources]
        row["answered"] = 1.0
        row["faithfulness"] = self._judge.faithfulness(response.answer, context).score
        row["answer_correctness"] = self._judge.correctness(
            example.query, response.answer, example.answer
        ).score
        row["answer_completeness"] = self._judge.completeness(
            example.query, response.answer, example.answer
        ).score
        row["citation_accuracy"] = evaluate_citations(
            response.answer, context, self._judge, self._support_threshold
        ).citation_accuracy_score
        return row


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
