"""Heuristic (rule-based) query rewriter.

Deterministic and dependency-free, so it ships in the retrieval sprint without
pulling in a generation model. It does two things:

* ``rewrite_query`` strips leading interrogatives and politeness, removes
  trailing punctuation, drops a small stopword set, and collapses whitespace.
* ``expand_query`` adds variants by mapping salient keywords to domain synonyms
  (e.g. ``deploy`` -> ``deployment process`` / ``docker deployment`` /
  ``service rollout``). The normalized query is always included first.

The synonym map is injectable so it can be tuned per knowledge base; an
LLM-backed :class:`QueryRewriter` can replace this later for richer expansions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from services.query_rewriting.base import QueryRewriter

# Leading question/politeness prefixes to strip (longest-first match).
_LEADING_PHRASES = [
    "how do we",
    "how do i",
    "how can i",
    "how to",
    "what is the",
    "what is",
    "what are",
    "where is",
    "where can i",
    "can you tell me",
    "please",
    "tell me",
]

_STOPWORDS = {"the", "a", "an", "do", "we", "i", "to", "of", "for", "is", "are", "how"}

# Default domain synonym/expansion map. Keyword -> additional query phrasings.
_DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "deploy": ["deployment process", "docker deployment", "service rollout"],
    "deployment": ["deployment process", "release process", "rollout"],
    "rollback": ["revert release", "undo deployment"],
    "auth": ["authentication", "authorization", "login"],
    "login": ["authentication", "sign in"],
    "incident": ["outage", "postmortem", "on-call response"],
    "onboarding": ["new hire setup", "getting started"],
}


class HeuristicQueryRewriter(QueryRewriter):
    """Rule-based query normalization and synonym expansion."""

    def __init__(
        self,
        synonyms: Mapping[str, list[str]] | None = None,
        max_variants: int = 4,
    ) -> None:
        self._synonyms = dict(synonyms) if synonyms is not None else dict(_DEFAULT_SYNONYMS)
        self._max_variants = max_variants

    def rewrite_query(self, query: str) -> str:
        """Normalize ``query`` into a cleaner search phrase."""
        text = query.strip().lower().rstrip("?.!")
        for phrase in _LEADING_PHRASES:
            if text.startswith(phrase + " "):
                text = text[len(phrase) :].strip()
                break
        tokens = [t for t in re.findall(r"[a-z0-9]+", text) if t not in _STOPWORDS]
        normalized = " ".join(tokens).strip()
        return normalized or query.strip().lower().rstrip("?.!")

    def expand_query(self, query: str) -> list[str]:
        """Return the normalized query plus synonym-driven variants."""
        normalized = self.rewrite_query(query)
        variants: list[str] = [normalized]
        for token in normalized.split():
            for expansion in self._synonyms.get(token, []):
                if expansion not in variants:
                    variants.append(expansion)
                if len(variants) >= self._max_variants:
                    return variants
        return variants
