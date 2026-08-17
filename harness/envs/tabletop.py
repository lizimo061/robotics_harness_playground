"""Multi-object 2D tabletop environment for harder manipulation tasks.

Consumes a declarative TaskSpec (from harness.tasks) and implements the physics
and success check for every task kind (pick_place, obstacle, push, stack, sort,
reach_avoid). This is the fast, dependency-free backend for iterating on harder
tasks; a 3D Franka backend maps the same TaskSpec onto MuJoCo/Genesis.

It also implements the object-aware query API so tools/agents can perceive the
scene (list_objects, get_object_pos, get_goal_pos, is_grasped, ...).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from harness.envs.base import Env
from harness.tasks import generate_task
from harness.tasks.base import TaskSpec
from harness.types import Obs, Action, StepResult, ActionSpace, ObservationSpace

_PALETTE = {
    "red": (220, 70, 70),
    "green": (90, 200, 90),
    "blue": (90, 140, 240),
    "cube": (240, 180, 60),
    "base": (150, 150, 160),
    "top": (240, 200, 70),
}


class TabletopEnv(Env):
    name = "tabletop"

    MAX_DELTA = 0.25

    def __init__(
        self,
        *,
        task_spec: Optional[TaskSpec] = None,
        task: str = "pick_place",
        seed: int = 0,
        max_episode_steps: int = 200,
        difficulty: float = 0.5,
        render_size: int = 256,
        **kwargs: Any,
    ) -> None:
        self._seed = seed
        self._max_steps = max_episode_steps
        self._render_size = render_size
        self._difficulty = difficulty
        self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._trail: list = []

        self.task_spec = task_spec if task_spec is not None else generate_task(task, seed=seed, difficulty=difficulty)
        self._objects: dict[str, np.ndarray] = {}
        self._targets: dict[str, str] = {}
        self._roles: dict[str, str] = {}
        for o in self.task_spec.objects:
            name = o["name"]
            self._objects[name] = np.asarray(o["pos"], dtype=float)
            if o.get("target"):
                self._targets[name] = o["target"]
            if o.get("role"):
                self._roles[name] = o["role"]
        self._goals: dict[str, np.ndarray] = {k: np.asarray(v, dtype=float) for k, v in self.task_spec.goals.items()}
        self._obstacles: list[dict] = [dict(o) for o in self.task_spec.obstacles]

        self._ee_home = np.asarray(self.task_spec.ee_start, dtype=float)
        self._ee = self._ee_home.copy()
        self._gripper = 0.0
        self._grasped: Optional[str] = None
        self._collided = False

        self._goal_radius = self.task_spec.params.get("goal_radius", 0.08)
        self._grasp_radius = self.task_spec.params.get("grasp_radius", 0.12)
        self._stack_radius = self.task_spec.params.get("stack_radius", 0.10)
        self._target_radius = self.task_spec.params.get("target_radius", 0.07)
        self._require_release = self.task_spec.params.get("require_release", False)

    # -- spaces ----------------------------------------------------------- #
    @property
    def observation_space(self) -> ObservationSpace:
        return ObservationSpace(
            state_dim=4,
            state_names=("ee_x", "ee_y", "gripper", "grasped"),
            has_image=True,
            image_shape=(self._render_size, self._render_size, 3),
            description="2D tabletop: end-effector + gripper + grasped flag; objects/goals in text state.",
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
        self._gripper = 0.0
        self._grasped = None
        self._collided = False
        self._steps = 0
        self._trail = [self._ee.copy()]
        return self._observe()

    def step(self, action: Action) -> StepResult:
        self._steps += 1

        if action.kind in ("ee_delta", "move", "delta") and action.value is not None:
            delta = np.asarray(action.value, dtype=float).ravel()[:2]
            delta = np.clip(delta, -self.MAX_DELTA, self.MAX_DELTA)
            self._ee = np.clip(self._ee + delta, 0.0, 1.0)
        elif action.kind in ("ee_pose", "move_to") and action.value is not None:
            self._ee = np.clip(np.asarray(action.value, dtype=float).ravel()[:2], 0.0, 1.0)

        if action.gripper is not None:
            self._set_gripper(action.gripper)

        if self._grasped is not None:
            self._objects[self._grasped] = self._ee.copy()

        self._trail.append(self._ee.copy())
        if len(self._trail) > 60:
            self._trail.pop(0)

        collided = self._check_collision()
        self._collided = self._collided or collided
        success = (not self._collided) and self._check_success()
        cost = self._primary_cost()

        if collided:
            reward, terminated, truncated = -1.0, True, False
        elif success:
            reward, terminated, truncated = 1.0, True, False
        else:
            reward, terminated, truncated = -cost - 0.01, False, (self._steps >= self._max_steps)

        return StepResult(
            obs=self._observe(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"success": success, "collided": self._collided, "cost": cost},
        )

    # -- physics / success ------------------------------------------------ #
    def _set_gripper(self, value: float) -> None:
        close = float(value) > 0.5
        self._gripper = 1.0 if close else 0.0
        if close and self._grasped is None:
            nearest, best = None, self._grasp_radius
            for name, pos in self._objects.items():
                d = float(np.linalg.norm(self._ee - pos))
                if d < best:
                    nearest, best = name, d
            self._grasped = nearest
        elif not close:
            self._grasped = None

    def _check_collision(self) -> bool:
        ee = self._ee
        for o in self._obstacles:
            c = np.asarray(o["pos"], dtype=float)
            r = float(o.get("radius", 0.1))
            if float(np.linalg.norm(ee - c)) < r:
                return True
            if self._grasped is not None:
                obj = self._objects[self._grasped]
                if float(np.linalg.norm(obj - c)) < r:
                    return True
        return False

    def _check_success(self) -> bool:
        kind = self.task_spec.kind
        if kind in ("reach", "reach_avoid"):
            if self.task_spec.ee_target is None:
                return False
            d = float(np.linalg.norm(self._ee - np.asarray(self.task_spec.ee_target)))
            return d < self._target_radius
        if kind == "stack":
            top = self._objects.get("top")
            base = self._objects.get("base")
            if top is None or base is None:
                return False
            d = float(np.linalg.norm(top - base))
            return d < self._stack_radius and self._grasped != "top"
        # place / push / sort
        if not self._objects:
            return False
        ok = True
        for name, pos in self._objects.items():
            goal_name = self._targets.get(name)
            if goal_name is None:
                continue
            goal = self._goals.get(goal_name)
            if goal is None:
                ok = False
                continue
            if float(np.linalg.norm(pos - goal)) >= self._goal_radius:
                ok = False
            if self._require_release and self._grasped == name:
                ok = False
        return ok

    def _primary_cost(self) -> float:
        kind = self.task_spec.kind
        if kind in ("reach", "reach_avoid") and self.task_spec.ee_target is not None:
            return float(np.linalg.norm(self._ee - np.asarray(self.task_spec.ee_target)))
        if kind == "stack" and "top" in self._objects and "base" in self._objects:
            return float(np.linalg.norm(self._objects["top"] - self._objects["base"]))
        total = 0.0
        for name, pos in self._objects.items():
            goal_name = self._targets.get(name)
            if goal_name and goal_name in self._goals:
                total += float(np.linalg.norm(pos - self._goals[goal_name]))
        return total

    # -- observation / query API ----------------------------------------- #
    def _state(self) -> np.ndarray:
        return np.array([self._ee[0], self._ee[1], self._gripper, float(self._grasped is not None)], dtype=np.float32)

    def _observe(self) -> Obs:
        return Obs(state=self._state(), text=self.get_text_state())

    def get_text_state(self) -> str:
        lines = [f"Task: {self.task_spec.description}"]
        grip = "closed" if self._grasped is not None else "open"
        lines.append(f"End-effector: ({self._ee[0]:.3f}, {self._ee[1]:.3f}), gripper={grip}, grasping={self._grasped or 'none'}")
        if self._objects:
            lines.append("Objects:")
            for name, pos in self._objects.items():
                tgt = f" -> {self._targets[name]}" if name in self._targets else ""
                lines.append(f"  {name} ({pos[0]:.3f}, {pos[1]:.3f}){tgt}")
        if self._goals:
            lines.append("Goals:")
            for name, pos in self._goals.items():
                lines.append(f"  {name} ({pos[0]:.3f}, {pos[1]:.3f})")
        if self._obstacles:
            lines.append("Obstacles:")
            for o in self._obstacles:
                lines.append(f"  {o['name']} ({o['pos'][0]:.3f}, {o['pos'][1]:.3f}) r={o.get('radius', 0.1):.2f}")
        lines.append(f"Distance to success: {self._primary_cost():.3f}")
        return "\n".join(lines)

    def list_objects(self) -> list[str]:
        return list(self._objects.keys())

    def get_object_pos(self, name: str):
        return self._objects.get(name)

    def list_goals(self) -> list[str]:
        return list(self._goals.keys())

    def get_goal_pos(self, name: str):
        return self._goals.get(name)

    def list_obstacles(self) -> list[str]:
        return [o["name"] for o in self._obstacles]

    def get_ee_pos(self):
        return self._ee.copy()

    def is_grasped(self) -> bool:
        return self._grasped is not None

    def grasped_object(self):
        return self._grasped

    # -- rendering -------------------------------------------------------- #
    def _color(self, name: str, idx: int, default) -> tuple:
        return _PALETTE.get(name, default)

    def render(self) -> Optional[np.ndarray]:
        s = self._render_size
        img = np.zeros((s, s, 3), dtype=np.uint8)
        img[..., :] = (22, 22, 32)

        def px(p):
            x = int(round(np.clip(p[0], 0.0, 1.0) * (s - 1)))
            y = int(round((1.0 - np.clip(p[1], 0.0, 1.0)) * (s - 1)))
            return y, x

        step = max(8, s // 10)
        img[::step, :, :] = (40, 40, 56)
        img[:, ::step, :] = (40, 40, 56)

        # trail
        n = len(self._trail)
        for k, pos in enumerate(self._trail):
            y, x = px(pos)
            fade = int(40 + 90 * (k + 1) / max(1, n))
            img[y - 1:y + 2, x - 1:x + 2] = (fade, 40, 40)

        # obstacles (grey circles)
        for o in self._obstacles:
            cy, cx = px(o["pos"])
            r = max(2, int(o.get("radius", 0.1) * s))
            yy, xx = np.ogrid[:s, :s]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
            img[mask] = (120, 120, 130)

        # goals (rings)
        for i, (name, pos) in enumerate(self._goals.items()):
            gy, gx = px(pos)
            radius = max(3, int(self._goal_radius * s))
            yy, xx = np.ogrid[:s, :s]
            dist = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
            ring = np.abs(dist - radius) < 2
            img[ring] = (90, 150, 255)

        # objects (colored squares)
        for i, (name, pos) in enumerate(self._objects.items()):
            oy, ox = px(pos)
            half = max(3, int(0.05 * s))
            col = _PALETTE.get(name, (90, 200, 90))
            img[oy - half:oy + half + 1, ox - half:ox + half + 1] = col

        # end-effector
        ey, ex = px(self._ee)
        half2 = max(3, int(0.035 * s))
        img[ey - half2:ey + half2 + 1, ex - half2:ex + half2 + 1] = (255, 90, 90)
        return img

    def close(self) -> None:
        pass
