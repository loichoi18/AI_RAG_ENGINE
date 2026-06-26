"""Evaluation runner.

Orchestrates a full evaluation pass over a golden dataset: it runs the retrieval
evaluator and (optionally) the generation evaluator, aggregates per-question rows
into headline metrics, and assembles an :class:`EvaluationReport` that can be
written to JSON / Markdown / CSV.

Generation evaluation is optional — pass an ``answer_service`` (and optionally a
``judge``) to enable it. Without one, the runner still produces the full
retrieval report, which is the fast, model-free path for CI and regression gates.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from evaluation.dataset import GoldenExample
from evaluation.generation_evaluator import GenerationEvaluator
from evaluation.judges.base import Judge
from evaluation.report import EvaluationReport, write_reports
from evaluation.retrieval_evaluator import RetrievalEvaluator
from services.answer_service import AnswerService
from utils.logging import get_logger

logger = get_logger(__name__)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class EvaluationRunner:
    """Runs retrieval (+ optional generation) evaluation and builds a report."""

    def __init__(
        self,
        retriever: object,
        answer_service: AnswerService | None = None,
        judge: Judge | None = None,
        k: int = 5,
        name: str = "evaluation",
    ) -> None:
        self._retriever = retriever
        self._answer_service = answer_service
        self._judge = judge
        self._k = k
        self._name = name

    def run(
        self,
        dataset: Sequence[GoldenExample],
        output_dir: str | Path | None = None,
        formats: Sequence[str] = ("json", "md", "csv"),
    ) -> EvaluationReport:
        """Execute the evaluation and (optionally) write reports to disk."""
        retr_eval = RetrievalEvaluator(self._retriever, k=self._k)  # type: ignore[arg-type]
        retr_rows = retr_eval.evaluate_examples(dataset)
        retrieval = self._aggregate_retrieval(retr_rows)

        gen_rows: list[dict[str, float]] = []
        generation: dict[str, float] = {}
        if self._answer_service is not None:
            gen_eval = GenerationEvaluator(self._answer_service, self._judge)
            gen_rows = gen_eval.evaluate_examples(dataset)
            generation = self._aggregate_generation(gen_rows)

        per_question = self._per_question(dataset, retr_rows, gen_rows)
        latency = {
            "avg_retrieval_latency_ms": _mean([r["latency_ms"] for r in retr_rows]),
        }
        retrieval_stats = {
            "avg_retrieved": _mean([r["retrieved"] for r in retr_rows]),
        }
        citation_stats = (
            {"avg_citation_accuracy": generation.get("citation_accuracy", 0.0)}
            if gen_rows
            else {}
        )

        report = EvaluationReport(
            name=self._name,
            dataset_size=len(dataset),
            k=self._k,
            overall={**retrieval, **generation},
            retrieval=retrieval,
            generation=generation,
            latency=latency,
            retrieval_stats=retrieval_stats,
            citation_stats=citation_stats,
            per_question=per_question,
        )

        if output_dir is not None:
            paths = write_reports(report, output_dir, formats)
            logger.info("eval.report_written", **{fmt: str(p) for fmt, p in paths.items()})
        return report

    def _aggregate_retrieval(self, rows: Sequence[dict[str, float]]) -> dict[str, float]:
        keys = [
            f"recall@{self._k}",
            f"precision@{self._k}",
            "mrr",
            f"ndcg@{self._k}",
            f"hit_rate@{self._k}",
        ]
        return {key: _mean([r[key] for r in rows]) for key in keys}

    @staticmethod
    def _aggregate_generation(rows: Sequence[dict[str, float]]) -> dict[str, float]:
        answered = [r for r in rows if r["answered"]]
        unanswerable = [r for r in rows if r["unanswerable"]]
        refusals = sum(1 for r in rows if r["refused"])
        return {
            "faithfulness": _mean([r["faithfulness"] for r in answered]),
            "answer_correctness": _mean([r["answer_correctness"] for r in answered]),
            "answer_completeness": _mean([r["answer_completeness"] for r in answered]),
            "citation_accuracy": _mean([r["citation_accuracy"] for r in answered]),
            "refusal_rate": refusals / len(rows) if rows else 0.0,
            "unanswerable_accuracy": _mean([r["unanswerable_correct"] for r in unanswerable]),
        }

    @staticmethod
    def _per_question(
        dataset: Sequence[GoldenExample],
        retr_rows: Sequence[dict[str, float]],
        gen_rows: Sequence[dict[str, float]],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, example in enumerate(dataset):
            row: dict[str, object] = {
                "id": example.id,
                "question_type": example.question_type.value,
            }
            row.update(retr_rows[index])
            if gen_rows:
                row.update(gen_rows[index])
            rows.append(row)
        return rows
