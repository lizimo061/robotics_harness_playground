"""Agent interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from harness.envs.base import Env
from harness.types import Episode


class Agent(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, env: Env, *, seed: Optional[int] = None) -> Episode:
        """Run one episode and return the recorded trajectory."""
        raise NotImplementedError
