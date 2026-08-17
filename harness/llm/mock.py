"""Offline mock LLM for tests and demos without network/API access."""
from __future__ import annotations

import json
from typing import Any

from harness.llm.base import ChatMessage, LLMClient, LLMResponse


class MockLLMClient(LLMClient):
    """Returns scripted responses in order, then repeats the last one.

    Configure through the LLM config's extra mapping::

        extra:
          responses: ["...", "..."]   # verbatim assistant replies
          script: [{...}, {...}]      # JSON actions, json-serialised for you
          fallback: '{"action":"stop"}'
    """

    def __init__(self, *, model: str = "mock", extra: dict | None = None, **_: Any) -> None:
        self._model = model
        extra = extra or {}
        self._responses = list(extra.get("responses") or [])
        script = extra.get("script") or []
        self._responses += [json.dumps(a) for a in script]
        self._fallback = extra.get("fallback", '{"action": "stop"}')
        self._i = 0
        self.history: list[list[dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return f"mock({self._model})"

    def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
        self.history.append([m.to_openai() for m in messages])
        text = self._responses[self._i] if self._i < len(self._responses) else self._fallback
        self._i += 1
        return LLMResponse(
            content=text,
            model=self._model,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            raw={},
        )
