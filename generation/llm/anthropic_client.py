"""Anthropic (Claude) LLM client.

Calls the Messages API. The system prompt is a top-level field (not a message),
per the Anthropic API. Same :class:`LLMClient` interface as the other backends.
"""

from __future__ import annotations

import httpx

from generation.llm.base import Completion, LLMClient


class AnthropicClient(LLMClient):
    """Message completions via the Anthropic API."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-latest",
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        api_version: str = "2023-06-01",
    ) -> None:
        if not api_key:
            raise ValueError("AnthropicClient requires an API key (set RAG_LLM__API_KEY)")
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._api_version = api_version

    def complete(self, system_prompt: str, user_prompt: str) -> Completion:
        payload = {
            "model": self._model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        response = httpx.post(
            f"{self._base_url}/messages",
            json=payload,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": self._api_version,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = "".join(block.get("text", "") for block in data.get("content", []))
        usage = data.get("usage", {})
        return Completion(
            text=text,
            prompt_tokens=int(usage.get("input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
        )
