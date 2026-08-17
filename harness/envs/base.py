"""Unified environment interface.

Every simulator backend (pure-numpy toy, Gymnasium/MuJoCo, Genesis, robosuite)
adapts to this minimal contract so the agent and evaluation layers never need
to know which backend they are driving.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from harness.types import Obs, Action, StepResult, ActionSpace, ObservationSpace


class Env(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def observation_space(self) -> ObservationSpace:
        ...

    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        ...

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None) -> Obs:
        ...

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        ...

    def render(self) -> Optional[np.ndarray]:
        """Return an RGB frame (H, W, 3) uint8, or None if unavailable."""
        return None

    def get_text_state(self) -> str:
        """Return a human/LLM-readable summary of the current state."""
        return ""

    def is_success(self, info: Optional[dict] = None) -> bool:
        # with an info dict, read it; without one, ask the env for its current
        # success state via _check_success (if the backend defines one).
        if info:
            return bool(info.get("success", False))
        checker = getattr(self, "_check_success", None)
        return bool(checker()) if checker is not None else False

    # -- object-aware query API (optional; safe defaults for simple envs) -- #
    # Harder manipulation tasks need these so tools/agents can perceive the
    # scene. Multi-object backends (TabletopEnv, future Franka wrappers) override.
    def list_objects(self) -> list[str]:
        return []

    def get_object_pos(self, name: str):
        return None

    def list_goals(self) -> list[str]:
        return []

    def get_goal_pos(self, name: str):
        return None

    def list_obstacles(self) -> list[str]:
        return []

    def get_ee_pos(self):
        return None

    def is_grasped(self) -> bool:
        return False

    def grasped_object(self):
        return None

    # -- containers / buttons / subgoals (optional; for long-horizon tasks) -- #
    def list_containers(self) -> list[str]:
        return []

    def get_container_interior(self, name: str):
        return None

    def get_door_pos(self, name: str):
        return None

    def is_container_open(self, name: str) -> bool:
        return True

    def list_buttons(self) -> list[str]:
        return []

    def get_button_pos(self, name: str):
        return None

    def is_button_pressed(self, name: str) -> bool:
        return False

    def check_subgoal(self, name: str) -> bool:
        return False

    def close(self) -> None:
        pass

    def __enter__(self) -> "Env":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
