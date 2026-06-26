"""LLM clients: thin, swappable backends behind one interface."""

from generation.llm.anthropic_client import AnthropicClient
from generation.llm.base import Completion, LLMClient
from generation.llm.factory import build_llm_client
from generation.llm.ollama_client import OllamaClient
from generation.llm.openai_client import OpenAIClient

__all__ = [
    "AnthropicClient",
    "Completion",
    "LLMClient",
    "OllamaClient",
    "OpenAIClient",
    "build_llm_client",
]
