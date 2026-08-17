"""Base protocol and message types for LLM clients."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union


class LLMError(RuntimeError):
    """A non-retryable error from the LLM backend."""


class TransientLLMError(LLMError):
    """A retryable error (rate limit, 5xx, network blip)."""


@dataclass
class ChatMessage:
    """A single chat message.

    The content is either a plain string or a list of multimodal blocks
    (OpenAI-style), e.g.::

        [{"type": "text", "text": "..."},
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]
    """

    role: str  # "system" | "user" | "assistant"
    content: Union[str, list[dict[str, Any]]]

    @staticmethod
    def system(text: str) -> "ChatMessage":
        return ChatMessage("system", text)

    @staticmethod
    def user(text: str) -> "ChatMessage":
        return ChatMessage("user", text)

    @staticmethod
    def assistant(text: str) -> "ChatMessage":
        return ChatMessage("assistant", text)

    @staticmethod
    def user_vision(text: str, image_b64: str, media_type: str = "image/png") -> "ChatMessage":
        content = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
            },
        ]
        return ChatMessage("user", content)

    def to_openai(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}

    def to_anthropic(self) -> dict[str, Any]:
        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}
        blocks: list[dict[str, Any]] = []
        for blk in self.content:
            if blk.get("type") == "text":
                blocks.append({"type": "text", "text": blk["text"]})
            elif blk.get("type") == "image_url":
                url = blk["image_url"]["url"]  # data:<media>;base64,<payload>
                header, _, data = url.partition(",")
                media_type = "image/png"
                if ":" in header:
                    media_type = header.split(":")[1].split(";")[0]
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    }
                )
        return {"role": self.role, "content": blocks}


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""


class LLMClient(ABC):
    """Minimal synchronous chat-completions interface."""

    name: str = "base"

    @abstractmethod
    def complete(self, messages: list[ChatMessage], **kwargs) -> LLMResponse:
        """Return a completion for the given messages."""
        raise NotImplementedError
