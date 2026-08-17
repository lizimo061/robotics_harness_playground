"""A pure-numpy toy tabletop pick-and-place environment.

This backend needs no simulator: it models a 2D end-effector with a binary
gripper moving a single object toward a goal region. It exists so the whole
harness (LLM <-> environment <-> evaluation) can run end-to-end on any machine,
and so the agent/test layers have a fast, deterministic target.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from harness.envs.base import Env
from harness.types import Obs, Action, StepResult, ActionSpace, ObservationSpace


class ToyTabletopEnv(Env):
    name = "toy_tabletop"

    TABLE = np.array([1.0, 1.0])
    GRASP_RADIUS = 0.09
    GOAL_RADIUS = 0.06
    MAX_DELTA = 0.25

    def __init__(
        self,
        *,
        task: str = "pick_and_place",
        seed: int = 0,
        max_episode_steps: int = 200,
        render: bool = False,
        render_size: int = 256,
        **kwargs,
    ) -> None:
        self.task = task
        self._seed = seed
        self._max_steps = max_episode_steps
        self._render_flag = render
        self._render_size = render_size
        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._trail: list = []

        self._obj_home = np.asarray(kwargs.get("obj_pos", [0.5, 0.5]), dtype=float)
        self._goal = np.asarray(kwargs.get("goal_pos", [0.85, 0.85]), dtype=float)
        self._ee_home = np.asarray(kwargs.get("ee_pos", [0.1, 0.1]), dtype=float)

        self._ee = self._ee_home.copy()
        self._gripper = 0.0
        self._grasped = False
        self._obj = self._obj_home.copy()

    # -- spaces ----------------------------------------------------------- #
    @property
    def observation_space(self) -> ObservationSpace:
        return ObservationSpace(
            state_dim=8,
            state_names=("ee_x", "ee_y", "gripper", "obj_x", "obj_y", "grasped", "goal_x", "goal_y"),
            has_image=True,
            image_shape=(self._render_size, self._render_size, 3),
            description="2D tabletop: end-effector (ee_x, ee_y), gripper, object, goal.",
        )

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            kind="ee_delta",
            dim=2,
            low=np.array([-self.MAX_DELTA, -self.MAX_DELTA], dtype=np.float32),
            high=np.array([self.MAX_DELTA, self.MAX_DELTA], dtype=np.float32),
            gripper_dim=1,
            description="move by (dx, dy) in [-0.25, 0.25]; set gripper 0 (open) or 1 (close).",
        )

    # -- lifecycle -------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None) -> Obs:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._ee = self._ee_home.copy()
        self._obj = self._obj_home.copy()
        self._gripper = 0.0
        self._grasped = False
        self._steps = 0
        self._trail = [self._ee.copy()]
        return self._observe()

    def step(self, action: Action) -> StepResult:
        self._steps += 1

        # apply end-effector movement
        if action.kind in ("ee_delta", "move", "delta") and action.value is not None:
            delta = np.asarray(action.value, dtype=float).ravel()[:2]
            delta = np.clip(delta, -self.MAX_DELTA, self.MAX_DELTA)
            self._ee = np.clip(self._ee + delta, 0.0, 1.0)
        elif action.kind in ("ee_pose", "move_to") and action.value is not None:
            target = np.asarray(action.value, dtype=float).ravel()[:2]
            self._ee = np.clip(target, 0.0, 1.0)

        # apply gripper
        if action.gripper is not None:
            self._set_gripper(action.gripper)

        # grasped object follows the end-effector
        if self._grasped:
            self._obj = self._ee.copy()

        self._trail.append(self._ee.copy())
        if len(self._trail) > 60:
            self._trail.pop(0)

        dist = float(np.linalg.norm(self._obj - self._goal))
        success = dist < self.GOAL_RADIUS
        reward = 1.0 if success else -dist - 0.01
        terminated = success
        truncated = self._steps >= self._max_steps

        return StepResult(
            obs=self._observe(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"success": success, "distance_to_goal": dist},
        )

    def _set_gripper(self, value: float) -> None:
        close = float(value) > 0.5
        self._gripper = 1.0 if close else 0.0
        if close and not self._grasped:
            if float(np.linalg.norm(self._ee - self._obj)) < self.GRASP_RADIUS:
                self._grasped = True
        elif not close:
            self._grasped = False

    # -- observation ------------------------------------------------------ #
    def _state(self) -> np.ndarray:
        return np.array(
            [
                self._ee[0], self._ee[1], self._gripper,
                self._obj[0], self._obj[1], float(self._grasped),
                self._goal[0], self._goal[1],
            ],
            dtype=np.float32,
        )

    def _observe(self) -> Obs:
        return Obs(state=self._state(), text=self.get_text_state())

    def get_text_state(self) -> str:
        gripper = "closed" if self._grasped else "open"
        dist = float(np.linalg.norm(self._obj - self._goal))
        lines = [
            f"End-effector: ({self._ee[0]:.3f}, {self._ee[1]:.3f}), gripper={gripper}",
            f"Object: ({self._obj[0]:.3f}, {self._obj[1]:.3f})",
            f"Goal: ({self._goal[0]:.3f}, {self._goal[1]:.3f})",
            f"Distance(object, goal): {dist:.3f}",
        ]
        return "\n".join(lines)

    def render(self) -> Optional[np.ndarray]:
        s = self._render_size
        img = np.zeros((s, s, 3), dtype=np.uint8)
        img[..., :] = (22, 22, 32)  # dark background

        def px(p):
            x = int(round(np.clip(p[0], 0.0, 1.0) * (s - 1)))
            y = int(round((1.0 - np.clip(p[1], 0.0, 1.0)) * (s - 1)))
            return y, x

        # faint grid
        step = max(8, s // 10)
        img[::step, :, :] = (40, 40, 56)
        img[:, ::step, :] = (40, 40, 56)

        # end-effector trail
        n = len(self._trail)
        for k, pos in enumerate(self._trail):
            y, x = px(pos)
            fade = int(40 + 90 * (k + 1) / max(1, n))
            img[y - 1:y + 2, x - 1:x + 2] = (fade, 40, 40)

        # goal ring (blue)
        gy, gx = px(self._goal)
        radius = max(3, int(self.GOAL_RADIUS * s))
        yy, xx = np.ogrid[:s, :s]
        dist = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
        ring = np.abs(dist - radius) < 2
        img[ring] = (90, 150, 255)

        # object (green square)
        oy, ox = px(self._obj)
        half = max(3, int(0.05 * s))
        img[oy - half:oy + half + 1, ox - half:ox + half + 1] = (90, 200, 90)

        # end-effector (red square)
        ey, ex = px(self._ee)
        half2 = max(3, int(0.035 * s))
        img[ey - half2:ey + half2 + 1, ex - half2:ex + half2 + 1] = (255, 90, 90)
        return img

    def close(self) -> None:
        pass
