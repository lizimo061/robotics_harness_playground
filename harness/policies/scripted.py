"""Offline policies for tests and demos without a policy server or GPU."""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np

from harness.policies.base import Policy
from harness.types import ActionSpace


class ScriptedPolicy(Policy):
    """Emit actions from a fixed sequence or a callable.

    ``actions`` may be a list of vectors (replayed in order, last one repeated)
    or a callable ``fn(observation_text, image, step) -> vector``.
    """

    name = "scripted_policy"

    def __init__(
        self,
        actions: Union[Sequence, Callable] = (),
        *,
        action_dim: int = 8,
    ) -> None:
        self._actions = actions
        self._action_dim = action_dim
        self._step = 0
        self.instruction = ""
        self.instructions: list[str] = []

    def begin(self, instruction: str, action_space: Optional[ActionSpace] = None) -> None:
        self.instruction = instruction or ""
        self.instructions.append(self.instruction)
        if action_space is not None and action_space.dim:
            self._action_dim = action_space.dim
        self._step = 0

    def act(self, observation_text: str, image: Optional[np.ndarray] = None) -> np.ndarray:
        if callable(self._actions):
            vec = self._actions(observation_text, image, self._step)
        elif len(self._actions):
            i = min(self._step, len(self._actions) - 1)
            vec = self._actions[i]
        else:
            vec = np.zeros(self._action_dim, dtype=np.float32)
        self._step += 1
        return np.asarray(vec, dtype=np.float32).ravel()

    def reset(self) -> None:
        self._step = 0
