"""OpenAI LLM client.

Calls the Chat Completions API. Requires an API key (supplied via settings).
Implemented behind the same :class:`LLMClient` interface as Ollama so switching
providers is a configuration change, not a code change.
"""

from __future__ import annotations

import httpx

from generation.llm.base import Completion, LLMClient


class OpenAIClient(LLMClient):
    """Chat completions via the OpenAI API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAIClient requires an API key (set RAG_LLM__API_KEY)")
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> Completion:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"],
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )
