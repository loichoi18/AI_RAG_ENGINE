"""Markdown report rendering for evaluation results.

Turns the comparison tables and the generation-metrics row into a single
Markdown document with one table per section, so any configuration change can be
re-run and diffed.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

MetricRow = Mapping[str, float]
ComparisonTable = Mapping[str, MetricRow]


def _table(rows: ComparisonTable, label: str) -> str:
    """Render a comparison table as Markdown (variant rows, metric columns)."""
    if not rows:
        return "_(no data)_\n"
    columns: list[str] = []
    for row in rows.values():
        for key in row:
            if key not in columns:
                columns.append(key)
    header = f"| {label} | " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * (len(columns) + 1)) + " |"
    lines = [header, sep]
    for name, row in rows.items():
        cells = " | ".join(f"{row.get(col, 0.0):.3f}" for col in columns)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines) + "\n"


def _best(rows: ComparisonTable, metric: str) -> str:
    """Name of the variant with the highest value for ``metric``."""
    if not rows:
        return "n/a"
    return max(rows.items(), key=lambda item: item[1].get(metric, 0.0))[0]


def render_report(
    *,
    dataset_size: int,
    k: int,
    chunking: ComparisonTable,
    retrieval: ComparisonTable,
    generation: MetricRow,
) -> str:
    """Render the full evaluation report as Markdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    recall_key = f"recall@{k}"
    ndcg_key = f"ndcg@{k}"

    gen_lines = "\n".join(
        f"| {name} | {value:.3f} |" for name, value in generation.items()
    )

    return f"""# RAG Engine — Evaluation Report

_Generated {now} · golden set: {dataset_size} examples · K={k}_

This report is produced by `scripts/evaluate.py` and is fully reproducible.
Retrieval relevance is judged by answer-span match (chunking-agnostic).

## 1. Chunking comparison (retriever fixed = hybrid)

{_table(chunking, "strategy")}
Best {recall_key}: **{_best(chunking, recall_key)}** · best {ndcg_key}: **{_best(chunking, ndcg_key)}**

## 2. Retrieval comparison (chunking fixed = recursive)

{_table(retrieval, "retriever")}
Best {recall_key}: **{_best(retrieval, recall_key)}** · best MRR: **{_best(retrieval, "mrr")}**

## 3. Generation metrics (extractive baseline, offline)

| metric | value |
| --- | --- |
{gen_lines}

## Notes

- Metrics are deterministic lexical approximations for reproducibility; the
  scorers are pluggable for an embedding- or LLM-judge upgrade.
- Generation metrics are averaged over answered (non-refused) examples; the
  refusal rate is reported alongside.
"""


class EvaluationReport(BaseModel):
    """A full evaluation run, serializable to JSON / Markdown / CSV."""

    name: str
    dataset_size: int
    k: int
    overall: dict[str, float] = Field(default_factory=dict)
    retrieval: dict[str, float] = Field(default_factory=dict)
    generation: dict[str, float] = Field(default_factory=dict)
    latency: dict[str, float] = Field(default_factory=dict)
    retrieval_stats: dict[str, float] = Field(default_factory=dict)
    citation_stats: dict[str, float] = Field(default_factory=dict)
    per_question: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def render_run_markdown(report: EvaluationReport) -> str:
    """Render an :class:`EvaluationReport` as a Markdown summary."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def kv_table(title: str, rows: Mapping[str, float]) -> str:
        if not rows:
            return f"### {title}\n\n_(none)_\n"
        body = "\n".join(f"| {k} | {v:.4f} |" for k, v in rows.items())
        return f"### {title}\n\n| metric | value |\n| --- | --- |\n{body}\n"

    sections = [
        f"# RAG Evaluation Report — {report.name}",
        f"\n_Generated {now} · {report.dataset_size} examples · K={report.k}_\n",
        kv_table("Retrieval", report.retrieval),
        kv_table("Generation", report.generation),
        kv_table("Latency (ms)", report.latency),
        kv_table("Citation statistics", report.citation_stats),
    ]
    return "\n".join(sections) + "\n"


def write_reports(
    report: EvaluationReport,
    output_dir: str | Path,
    formats: Sequence[str] = ("json", "md", "csv"),
) -> dict[str, Path]:
    """Write the report to ``output_dir`` in the requested formats.

    Returns a mapping of format -> written path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if "json" in formats:
        path = out / f"{report.name}.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        written["json"] = path

    if "md" in formats:
        path = out / f"{report.name}.md"
        path.write_text(render_run_markdown(report), encoding="utf-8")
        written["md"] = path

    if "csv" in formats and report.per_question:
        path = out / f"{report.name}.csv"
        columns: list[str] = []
        for row in report.per_question:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(report.per_question)
        written["csv"] = path

    return written
