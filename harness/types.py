"""Shared core types for the robotics harness.

These dataclasses are the common currency that flows between the LLM layer,
the agent loop, and the environment adapters. Keeping them dependency-light
(numpy only) lets every optional environment backend map onto one interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Spaces
# --------------------------------------------------------------------------- #
@dataclass
class ObservationSpace:
    """Static description of what an environment observes."""

    state_dim: int = 0
    state_names: tuple[str, ...] = ()
    image_shape: tuple[int, int, int] = (0, 0, 0)  # (H, W, C)
    has_image: bool = False
    has_depth: bool = False
    description: str = ""


@dataclass
class ActionSpace:
    """Static description of the actions an environment accepts."""

    kind: str = "ee_delta"  # joint_position | ee_delta | ee_pose | discrete | noop
    dim: int = 0
    low: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    high: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    joint_names: tuple[str, ...] = ()
    gripper_dim: int = 0  # trailing dimensions reserved for gripper (0..1 open..close)
    description: str = ""

    def clip(self, value) -> np.ndarray:
        v = np.asarray(value, dtype=np.float32)
        if self.low.size and self.high.size:
            return np.clip(v, self.low, self.high)
        return v

    def to_text(self) -> str:
        parts = [f"kind={self.kind}", f"dim={self.dim}"]
        if self.joint_names:
            parts.append("joints=" + ",".join(self.joint_names))
        if self.low.size and self.high.size:
            parts.append(f"low={np.round(self.low, 3).tolist()}")
            parts.append(f"high={np.round(self.high, 3).tolist()}")
        if self.description:
            parts.append(self.description)
        return " ".join(parts)


# --------------------------------------------------------------------------- #
# Runtime containers
# --------------------------------------------------------------------------- #
@dataclass
class Obs:
    """An observation produced by an environment."""

    state: Optional[np.ndarray] = None  # low-dimensional state vector
    image: Optional[np.ndarray] = None  # RGB uint8 (H, W, 3)
    depth: Optional[np.ndarray] = None  # optional depth (H, W)
    text: str = ""  # optional human/LLM-readable summary
    info: dict[str, Any] = field(default_factory=dict)

    def to_text(self, state_names: tuple[str, ...] = ()) -> str:
        parts: list[str] = []
        if self.state is not None:
            arr = np.asarray(self.state).ravel()
            if state_names and len(state_names) == len(arr):
                parts.append(" | ".join(f"{n}={v:.4g}" for n, v in zip(state_names, arr)))
            else:
                parts.append("state=" + np.array2string(arr, precision=3))
        if self.text:
            parts.append(self.text)
        return "\n".join(parts)


@dataclass
class Action:
    """A single action applied to an environment.

    ``kind`` selects the interpretation of ``value``; ``gripper`` is a
    convenience channel so position + gripper can be sent in one step.
    """

    kind: str = "ee_delta"
    value: Optional[np.ndarray] = None
    gripper: Optional[float] = None  # 0.0 open .. 1.0 closed
    duration: float = 0.0  # seconds; 0 => environment default
    comment: str = ""


@dataclass
class StepResult:
    obs: Obs
    reward: float = 0.0
    terminated: bool = False  # reached a terminal state (incl. success)
    truncated: bool = False  # hit a time/step budget
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def success(self) -> bool:
        return bool(self.info.get("success", False))


@dataclass
class Episode:
    """A recorded episode of interaction."""

    actions: list[Action] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    observations: list[Obs] = field(default_factory=list)
    success: bool = False
    total_reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return len(self.actions)
