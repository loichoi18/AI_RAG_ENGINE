"""LLM client factory.

Constructs the configured backend from :class:`LLMSettings`, so the active
provider is a one-line configuration choice. Ollama is the key-free default.
"""

from __future__ import annotations

from configs.settings import LLMProvider, LLMSettings
from generation.llm.anthropic_client import AnthropicClient
from generation.llm.base import LLMClient
from generation.llm.ollama_client import OllamaClient
from generation.llm.openai_client import OpenAIClient


def build_llm_client(settings: LLMSettings) -> LLMClient:
    """Return the LLM client for the configured provider."""
    match settings.provider:
        case LLMProvider.OLLAMA:
            return OllamaClient(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        case LLMProvider.OPENAI:
            return OpenAIClient(
                api_key=settings.api_key,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        case LLMProvider.ANTHROPIC:
            return AnthropicClient(
                api_key=settings.api_key,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
