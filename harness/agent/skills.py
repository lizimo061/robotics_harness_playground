"""Skill library for Code-as-Policies mode.

In "code" mode the LLM writes a short Python snippet that calls these
functions; the harness executes the snippet in a restricted sandbox and the
functions drive the environment.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from harness.envs.base import Env
from harness.types import Action, StepResult


class SkillContext:
    """Bind a running environment to a set of skill functions the LLM may call."""

    def __init__(self, env: Env, on_step: Optional[Callable[[Action, StepResult], None]] = None) -> None:
        self.env = env
        self.on_step = on_step
        self.finished = False

    def _step(self, action: Action) -> StepResult:
        result = self.env.step(action)
        if self.on_step is not None:
            self.on_step(action, result)
        return result

    # -- skills ---------------------------------------------------------- #
    def move_delta(self, dx=0.0, dy=0.0, dz=0.0, gripper=None) -> str:
        r = self._step(Action(kind="ee_delta", value=np.array([dx, dy, dz], dtype=np.float32), gripper=gripper))
        return self._status(r)

    def move_to(self, x, y, z=0.0, gripper=None) -> str:
        r = self._step(Action(kind="ee_pose", value=np.array([x, y, z], dtype=np.float32), gripper=gripper))
        return self._status(r)

    def set_gripper(self, value) -> str:
        r = self._step(Action(kind="noop", gripper=float(value)))
        return self._status(r)

    def grasp(self) -> str:
        return self.set_gripper(1.0)

    def release(self) -> str:
        return self.set_gripper(0.0)

    def set_joints(self, positions) -> str:
        r = self._step(Action(kind="joint_position", value=np.asarray(positions, dtype=np.float32)))
        return self._status(r)

    def state(self) -> str:
        return self.env.get_text_state()

    def done(self) -> str:
        self.finished = True
        return "done"

    def _status(self, r: StepResult) -> str:
        suffix = " [SUCCESS]" if r.success else ""
        return self.env.get_text_state() + suffix

    # -- sandbox namespace ---------------------------------------------- #
    def namespace(self) -> dict:
        return {
            "move_delta": self.move_delta,
            "move_to": self.move_to,
            "set_gripper": self.set_gripper,
            "grasp": self.grasp,
            "release": self.release,
            "set_joints": self.set_joints,
            "state": self.state,
            "done": self.done,
            "np": np,
            "math": math,
        }


def get_skill_docs() -> str:
    lines = [
        "move_delta(dx, dy, dz=0, gripper=None)  -> move the end-effector by a small delta (meters)",
        "move_to(x, y, z=0, gripper=None)        -> move the end-effector to an absolute position",
        "set_gripper(value)                      -> set gripper: 0 open, 1 closed",
        "grasp()                                 -> close the gripper",
        "release()                               -> open the gripper",
        "set_joints(positions)                   -> set joint positions (radians)",
        "state()                                 -> return the current state as text",
        "done()                                  -> declare the task finished",
    ]
    return chr(10).join(lines)
