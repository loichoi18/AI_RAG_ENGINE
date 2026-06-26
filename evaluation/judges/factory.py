"""Judge construction from settings.

Builds the production judge: an :class:`LLMJudge` over the configured backend,
wrapped with a :class:`LexicalJudge` fallback so evaluation degrades gracefully
to deterministic scoring when the model is unreachable.
"""

from __future__ import annotations

from configs.settings import LLMSettings
from evaluation.judges.base import Judge
from evaluation.judges.fallback import FallbackJudge
from evaluation.judges.lexical_judge import LexicalJudge
from evaluation.judges.llm_judge import LLMJudge
from generation.llm.factory import build_llm_client


def build_judge(settings: LLMSettings) -> Judge:
    """Return an LLM judge with a lexical fallback (LLM-primary policy)."""
    return FallbackJudge(LLMJudge(build_llm_client(settings)), LexicalJudge())
