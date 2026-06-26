"""Evaluation layer: golden dataset, metrics, evaluators, comparison, report."""

from evaluation.comparison import compare_chunking, compare_retrieval
from evaluation.dataset import (
    GoldenExample,
    is_relevant,
    load_corpus_documents,
    load_golden,
)
from evaluation.generation_evaluator import GenerationEvaluator
from evaluation.report import render_report
from evaluation.retrieval_evaluator import RetrievalEvaluator

__all__ = [
    "GenerationEvaluator",
    "GoldenExample",
    "RetrievalEvaluator",
    "compare_chunking",
    "compare_retrieval",
    "is_relevant",
    "load_corpus_documents",
    "load_golden",
    "render_report",
]
