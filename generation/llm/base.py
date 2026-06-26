"""Low-level LLM client abstraction.

An ``LLMClient`` does exactly one thing: turn a fully-formed prompt into text
plus token usage. Grounding logic (prompt construction, citation parsing) lives
one layer up in :class:`~generation.generator.GroundedGenerator`, so every
backend stays thin and the grounding rules are defined once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class Completion(BaseModel):
    """A raw LLM completion with token accounting."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens


class LLMClient(ABC):
    """Generates a text completion from a prompt."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> Completion:
        """Return a completion for the given system and user prompts.

        Parameters
        ----------
        system_prompt:
            Instructions that constrain the model (grounding rules).
        user_prompt:
            The numbered context plus the question.
        """
        raise NotImplementedError
