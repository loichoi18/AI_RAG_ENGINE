"""Run the evaluation suite over the golden dataset.

Builds the production service graph, runs retrieval + generation evaluation with
an LLM judge (lexical fallback), and writes JSON / Markdown / CSV reports.

Usage:
    python -m scripts.evaluate            # writes reports/ to the repo root
"""

from __future__ import annotations

from api.services import build_services
from configs.settings import get_settings
from evaluation.dataset import load_golden
from evaluation.judges.factory import build_judge
from evaluation.runner import EvaluationRunner
from utils.logging import configure_logging, get_logger


def main() -> None:
    """Execute the evaluation and write reports."""
    settings = get_settings()
    configure_logging(settings.logging)
    logger = get_logger("scripts.evaluate")

    services = build_services(settings)
    dataset = load_golden()

    runner = EvaluationRunner(
        retriever=services.retrievers["hybrid"],
        answer_service=services.answer_services["hybrid"],
        judge=build_judge(settings.llm),
        k=settings.reranker.top_k,
        name="baseline",
    )
    report = runner.run(dataset, output_dir="reports", formats=("json", "md", "csv"))
    logger.info("evaluate.done", retrieval=report.retrieval, generation=report.generation)
    print("Retrieval:", report.retrieval)
    print("Generation:", report.generation)


if __name__ == "__main__":
    main()
