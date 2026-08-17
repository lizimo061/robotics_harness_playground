"""Skill abstraction: a subgoal-level, closed-loop capability.

A Skill sits above a Tool: it knows how to check whether its subgoal is
satisfied (satisfied(env)) and how to produce the next primitive action until it
is (plan_action(env)). Skills are the building blocks of long-horizon tasks:
the planner sequences skills, and the executor runs each one closed-loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from harness.types import Action


def dist2d(env, a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a[:2] - b[:2]))


@dataclass
class SkillResult:
    success: bool
    feedback: str = ""
    steps: int = 0


class Skill(ABC):
    name: str = ""
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})
    tolerance: float = 0.15  # distance threshold for grasp / actuate / place

    def __init__(self, **args):
        self.args = args

    def reset(self) -> None:
        pass

    @abstractmethod
    def satisfied(self, env) -> bool:
        """Whether this skill's subgoal is currently achieved."""

    @abstractmethod
    def plan_action(self, env) -> Optional[Action]:
        """Return the next primitive action, or None if no progress is possible."""

    def signature(self) -> str:
        props = self.parameters.get("properties", {})
        return f"{self.name}({', '.join(props)})"

    def __repr__(self) -> str:
        return f"Skill({self.name}{self.args})"
