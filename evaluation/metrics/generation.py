"""Generation metrics: faithfulness, citation accuracy, answer correctness.

These are deterministic, lexical approximations chosen for reproducibility (no
API keys, runs in CI). Each is intentionally simple and explainable:

* **faithfulness** — fraction of answer sentences whose content words are
  sufficiently covered by the retrieved context. Ungrounded sentences (possible
  hallucinations) score 0.
* **citation_accuracy** — fraction of the answer's citations that point to a
  chunk that is actually relevant to the question.
* **answer_correctness** — token-level F1 between the answer and the ground
  truth.

``answer_correctness`` accepts a pluggable scorer, so an embedding- or
LLM-judge-based scorer can replace the lexical default without touching callers.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from generation.citations import extract_citation_indices
from models.domain import Chunk

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "and", "or", "in",
    "on", "for", "with", "by", "at", "as", "that", "this", "it", "be", "from",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def faithfulness(answer: str, context: Sequence[Chunk], coverage_threshold: float = 0.6) -> float:
    """Fraction of answer sentences grounded in the retrieved context.

    A sentence is grounded if at least ``coverage_threshold`` of its content
    words appear in the combined context text.
    """
    sentences = [s for s in _SENTENCE_RE.split(answer.strip()) if s.strip()]
    if not sentences:
        return 0.0
    context_words = set()
    for chunk in context:
        context_words |= _content_words(chunk.text)

    grounded = 0
    for sentence in sentences:
        words = _content_words(sentence)
        if not words:
            continue
        overlap = len(words & context_words) / len(words)
        if overlap >= coverage_threshold:
            grounded += 1
    return grounded / len(sentences)


def citation_accuracy(
    answer: str,
    context: Sequence[Chunk],
    is_relevant: Callable[[Chunk], bool],
) -> float:
    """Fraction of cited chunks that are actually relevant.

    Citations are ``[n]`` markers mapped to ``context[n-1]``. Out-of-range
    citations count against accuracy. Returns 1.0 when there are no citations
    (vacuously correct — nothing wrong was cited).
    """
    indices = [i for i in dict.fromkeys(extract_citation_indices(answer))]
    if not indices:
        return 1.0
    correct = 0
    for index in indices:
        if 1 <= index <= len(context) and is_relevant(context[index - 1]):
            correct += 1
    return correct / len(indices)


def token_f1(answer: str, ground_truth: str) -> float:
    """Token-level F1 overlap between answer and ground truth."""
    pred = _tokens(answer)
    truth = _tokens(ground_truth)
    if not pred or not truth:
        return 0.0
    common: dict[str, int] = {}
    truth_counts: dict[str, int] = {}
    for t in truth:
        truth_counts[t] = truth_counts.get(t, 0) + 1
    overlap = 0
    for t in pred:
        if truth_counts.get(t, 0) - common.get(t, 0) > 0:
            common[t] = common.get(t, 0) + 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(truth)
    return 2 * precision * recall / (precision + recall)


def answer_completeness(answer: str, ground_truth: str) -> float:
    """Fraction of ground-truth content words covered by the answer.

    A lexical proxy for completeness: an answer that omits key facts from the
    reference scores lower. This is the deterministic fallback used when no
    LLM judge is configured. Returns 0.0 when the ground truth has no content
    words.
    """
    truth_words = _content_words(ground_truth)
    if not truth_words:
        return 0.0
    answer_words = _content_words(answer)
    return len(truth_words & answer_words) / len(truth_words)


# Default correctness scorer; swappable for embedding/LLM-judge variants.
AnswerCorrectnessScorer = Callable[[str, str], float]


def answer_correctness(
    answer: str,
    ground_truth: str,
    scorer: AnswerCorrectnessScorer = token_f1,
) -> float:
    """Score answer correctness against ground truth using ``scorer``."""
    return scorer(answer, ground_truth)
