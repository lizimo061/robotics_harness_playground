"""Convenience constructors for Action objects."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from harness.types import Action


def move_delta(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, gripper: Optional[float] = None, comment: str = "") -> Action:
    return Action(kind="ee_delta", value=np.array([dx, dy, dz], dtype=np.float32), gripper=gripper, comment=comment)


def move_to(x: float, y: float, z: float = 0.0, gripper: Optional[float] = None, comment: str = "") -> Action:
    return Action(kind="ee_pose", value=np.array([x, y, z], dtype=np.float32), gripper=gripper, comment=comment)


def set_gripper(value: float, comment: str = "") -> Action:
    return Action(kind="noop", gripper=float(value), comment=comment)


def set_joint_positions(positions: Sequence[float], comment: str = "") -> Action:
    return Action(kind="joint_position", value=np.asarray(positions, dtype=np.float32), comment=comment)


def noop(comment: str = "") -> Action:
    return Action(kind="noop", comment=comment)


def stop(comment: str = "") -> Action:
    return Action(kind="stop", comment=comment)
