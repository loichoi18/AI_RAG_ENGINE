"""Ollama LLM client (local, default backend).

Calls the Ollama chat API over HTTP. Temperature defaults to 0 for grounded,
reproducible answers. No API key required, which keeps the project clone-and-run.
"""

from __future__ import annotations

import httpx

from generation.llm.base import Completion, LLMClient


class OllamaClient(LLMClient):
    """Chat completions via a local Ollama server."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
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
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        response = httpx.post(
            f"{self._base_url}/api/chat", json=payload, timeout=self._timeout
        )
        response.raise_for_status()
        data = response.json()
        return Completion(
            text=data.get("message", {}).get("content", ""),
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
        )
