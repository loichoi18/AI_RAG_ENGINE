"""Grounded prompt construction.

Builds a system prompt encoding the grounding rules (answer only from context,
cite sources, refuse when unsupported) and a user prompt containing the
**numbered** context blocks followed by the question. The numbering is the
contract the citation parser relies on: block *i* is cited as ``[i]``.
"""

from __future__ import annotations

from collections.abc import Sequence

from models.domain import Chunk

# Sentinel the model must emit when the context does not support an answer.
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You are a precise internal knowledge assistant. Follow these rules strictly:

1. Answer ONLY using the numbered context blocks provided. Do not use outside knowledge.
2. Cite every claim with the supporting block number in square brackets, e.g. "The service is asynchronous [1]." Cite multiple blocks as [1][2] when relevant.
3. If the context does not contain enough information to answer, reply with exactly: {INSUFFICIENT_CONTEXT}
4. Do not invent facts, sources, or citations. Never cite a block number that was not provided.
5. Be concise and factual."""


class GroundedPromptBuilder:
    """Assembles the system and user prompts for grounded generation."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        self._system_prompt = system_prompt

    @property
    def system_prompt(self) -> str:
        """The grounding system prompt."""
        return self._system_prompt

    def build_user_prompt(self, query: str, context: Sequence[Chunk]) -> str:
        """Render numbered context blocks plus the question.

        Each block is labelled with its 1-based index and a short source hint
        (document id and, when present, section/page) to aid traceability.
        """
        blocks: list[str] = []
        for index, chunk in enumerate(context, start=1):
            source = self._source_hint(chunk)
            blocks.append(f"[{index}] (source: {source})\n{chunk.text}")
        context_text = "\n\n".join(blocks) if blocks else "(no context provided)"
        return (
            f"Context blocks:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above, with citations."
        )

    @staticmethod
    def _source_hint(chunk: Chunk) -> str:
        parts = [chunk.document_id]
        if chunk.section_title:
            parts.append(chunk.section_title)
        if chunk.page_number is not None:
            parts.append(f"p.{chunk.page_number}")
        return " / ".join(parts)
