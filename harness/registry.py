"""Tiny generic registry used for provider-style factories (LLM, env, agent)."""
from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, key: str):
        def deco(fn):
            self._items[key] = fn
            return fn

        return deco

    def get(self, key: str):
        if key not in self._items:
            raise KeyError(
                f"Unknown {self.name} '{key}'. Available: {sorted(self._items)}"
            )
        return self._items[key]

    def available(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items
