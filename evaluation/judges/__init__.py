"""LLM-as-judge layer: interface, LLM/lexical implementations, fallback."""

from evaluation.judges.base import Judge, JudgeVerdict
from evaluation.judges.factory import build_judge
from evaluation.judges.fallback import FallbackJudge
from evaluation.judges.lexical_judge import LexicalJudge
from evaluation.judges.llm_judge import LLMJudge

__all__ = [
    "FallbackJudge",
    "Judge",
    "JudgeVerdict",
    "LLMJudge",
    "LexicalJudge",
    "build_judge",
]
