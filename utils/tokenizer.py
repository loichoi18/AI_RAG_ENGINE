"""Tokenizer abstraction used for chunk sizing and ``token_count``.

Chunkers express their budgets in **tokens**, not characters, so that a chunk
matches the context the embedding model actually consumes. To keep chunkers
unit-testable without downloading model weights, they depend on the
:class:`Tokenizer` *protocol*; production wires in :class:`BGETokenizer` (the
embedding model's own HuggingFace tokenizer), while tests inject a deterministic
fake.

Heavy ``transformers`` import is deferred to :class:`BGETokenizer.__init__` so
importing this module costs nothing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal token interface needed by the chunkers."""

    def encode(self, text: str) -> list[int]:
        """Return token ids for ``text`` (without special tokens)."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Inverse of :meth:`encode`."""
        ...

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...


class WhitespaceTokenizer:
    """Dependency-free whitespace tokenizer.

    A real, lightweight default and a good fit for offline evaluation: one token
    per whitespace-delimited word, round-trippable via a growing vocabulary.
    Not as accurate as the model tokenizer for production chunk sizing, but
    requires no downloads.
    """

    def __init__(self) -> None:
        self._id_to_word: list[str] = []
        self._word_to_id: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            if word not in self._word_to_id:
                self._word_to_id[word] = len(self._id_to_word)
                self._id_to_word.append(word)
            ids.append(self._word_to_id[word])
        return ids

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self._id_to_word[t] for t in tokens)

    def count(self, text: str) -> int:
        return len(text.split())


class BGETokenizer:
    """Token interface backed by a HuggingFace ``AutoTokenizer``.

    Parameters
    ----------
    model_name:
        HF model id whose tokenizer to load (defaults to the bge embedding
        model so token counts match the embedder).
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        from transformers import AutoTokenizer  # lazy: avoid import at module load

        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    @property
    def model_name(self) -> str:
        """The underlying model id."""
        return self._model_name

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` to token ids, excluding special tokens."""
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids back to text, skipping special tokens."""
        return str(self._tokenizer.decode(tokens, skip_special_tokens=True))

    def count(self, text: str) -> int:
        """Count tokens in ``text``."""
        return len(self.encode(text))
