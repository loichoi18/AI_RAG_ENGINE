"""Unit tests for the configuration system.

Testing strategy
----------------
Configuration is the contract between the deployment environment and the code,
so we test three things: (1) defaults are sane and present, (2) environment
variables override nested settings via the documented delimiter, and (3)
invalid values are rejected at load time. All Settings are built with
``_env_file=None`` to stay hermetic.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from configs.settings import Environment, LLMProvider, LogLevel, Settings, get_settings


def test_defaults_are_sane(default_settings: Settings) -> None:
    """Defaults must yield a runnable local configuration with no env file."""
    s = default_settings
    assert s.app_name == "rag-engine"
    assert s.environment is Environment.LOCAL
    assert s.debug is False
    assert s.api.host == "0.0.0.0"
    assert s.api.port == 8000
    assert s.qdrant.host == "localhost"
    assert s.qdrant.port == 6333
    assert s.qdrant.collection_name == "documents"
    assert s.embeddings.embedding_model == "BAAI/bge-base-en-v1.5"
    assert s.llm.provider is LLMProvider.OLLAMA
    assert s.llm.ollama_base_url == "http://localhost:11434"
    assert s.retrieval.top_k == 5
    assert s.logging.log_level is LogLevel.INFO
    assert s.logging.json_logs is True


def test_qdrant_url_property(default_settings: Settings) -> None:
    """The convenience URL is derived from host and port."""
    assert default_settings.qdrant.url == "http://localhost:6333"


def test_env_overrides_nested_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested fields are overridable via RAG_<DOMAIN>__<FIELD>."""
    monkeypatch.setenv("RAG_QDRANT__HOST", "qdrant.internal")
    monkeypatch.setenv("RAG_QDRANT__PORT", "7000")
    monkeypatch.setenv("RAG_LLM__PROVIDER", "openai")
    monkeypatch.setenv("RAG_RETRIEVAL__TOP_K", "12")

    s = Settings(_env_file=None)

    assert s.qdrant.host == "qdrant.internal"
    assert s.qdrant.port == 7000
    assert s.llm.provider is LLMProvider.OPENAI
    assert s.retrieval.top_k == 12


def test_top_level_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-level fields use the RAG_ prefix without a nested delimiter."""
    monkeypatch.setenv("RAG_ENVIRONMENT", "prod")
    monkeypatch.setenv("RAG_DEBUG", "true")

    s = Settings(_env_file=None)

    assert s.environment is Environment.PROD
    assert s.debug is True


def test_invalid_enum_value_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown provider must fail fast at load time."""
    monkeypatch.setenv("RAG_LLM__PROVIDER", "not-a-provider")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_port_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ports outside the valid range are rejected."""
    monkeypatch.setenv("RAG_API__PORT", "70000")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached() -> None:
    """get_settings returns a process-wide singleton until the cache is cleared."""
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()
    assert get_settings() is not first
