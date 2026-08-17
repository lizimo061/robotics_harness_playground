"""OpenAI-compatible Chat Completions client.

One client covers DeepSeek, Kimi (Moonshot), OpenAI, vLLM, Ollama, and any
other endpoint that speaks the OpenAI /v1/chat/completions protocol.
"""
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


class OpenAICompatClient(LLMClient):
    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        api_key_env: str = "",
        base_url: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        extra: Optional[dict] = None,
    ) -> None:
        self._model = model
        self._key = api_key or os.environ.get(api_key_env or "", "")
        self._base_url = (base_url or "").rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._extra = extra or {}
        self._headers = {"Content-Type": "application/json"}
        if self._key:
            self._headers["Authorization"] = f"Bearer {self._key}"

    @property
    def name(self) -> str:
        return f"openai-compat({self._model})"

    def _post(self, payload: dict, timeout: float) -> dict:
        url = f"{self._base_url}/chat/completions"
        try:
            resp = httpx.post(url, headers=self._headers, json=payload, timeout=timeout)
        except httpx.TransportError as e:
            raise TransientLLMError(f"network error: {e}") from e
        if resp.status_code in _TRANSIENT_STATUS:
            raise TransientLLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [m.to_openai() for m in messages],
            "temperature": self._temperature if temperature is None else temperature,
        }
        mt = self._max_tokens if max_tokens is None else max_tokens
        if mt:
            payload["max_tokens"] = mt
        payload.update(self._extra)
        payload.update(kwargs)

        data = with_retries(
            lambda: self._post(payload, timeout or self._timeout),
            retries=3,
            exceptions=(TransientLLMError,),
        )
        return self._parse(data)

    def _parse(self, data: dict) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"no choices in response: {str(data)[:300]}")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if content is None:
            # some reasoning models return reasoning_content with empty content
            content = msg.get("reasoning_content", "") or ""
        return LLMResponse(
            content=content,
            model=data.get("model", self._model),
            usage=data.get("usage") or {},
            raw=data,
            finish_reason=choices[0].get("finish_reason", ""),
        )
