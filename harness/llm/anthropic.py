"""Anthropic (Claude) Messages API client."""
from __future__ import annotations

import os
from typing import Any, Optional

import httpx

from harness.llm.base import (
    ChatMessage,
    LLMClient,
    LLMError,
    LLMResponse,
    TransientLLMError,
)
from harness.llm.retry import with_retries

_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class AnthropicClient(LLMClient):
    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = "https://api.anthropic.com",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        extra: Optional[dict] = None,
    ) -> None:
        self._model = model
        self._key = api_key or os.environ.get(api_key_env or "", "")
        self._base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._extra = extra or {}

    @property
    def name(self) -> str:
        return f"anthropic({self._model})"

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system_parts = [m.content for m in messages if m.role == "system" and isinstance(m.content, str)]
        msgs = [m.to_anthropic() for m in messages if m.role != "system"]

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "messages": msgs,
            "temperature": self._temperature if temperature is None else temperature,
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        payload.update(self._extra)
        payload.update(kwargs)

        def call() -> dict:
            url = f"{self._base_url}/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self._key,
                "anthropic-version": "2023-06-01",
            }
            try:
                resp = httpx.post(url, headers=headers, json=payload, timeout=timeout or self._timeout)
            except httpx.TransportError as e:
                raise TransientLLMError(f"network error: {e}") from e
            if resp.status_code in _TRANSIENT_STATUS:
                raise TransientLLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            if resp.status_code >= 400:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            return resp.json()

        data = with_retries(call, retries=3, exceptions=(TransientLLMError,))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return LLMResponse(
            content=text,
            model=data.get("model", self._model),
            usage=data.get("usage") or {},
            raw=data,
            finish_reason=data.get("stop_reason", ""),
        )
