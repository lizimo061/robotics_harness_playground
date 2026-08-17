"""KitchenEnv: a long-horizon multi-object env with containers and buttons.

2D abstraction of a manipulation scene with:
  - objects   (movable; e.g. bread)
  - containers (fixed, with an interior region and a door; e.g. oven)
  - buttons   (fixed, pressable; e.g. heat button)

It consumes a long-horizon TaskSpec (harness.tasks.specs_long.cook_bread) whose
steps are ordered subgoals. Success = every subgoal's check passes. The env
implements the container/button/subgoal query API so composite skills
(harness.skills) can plan against it.

Interaction model (2D): closing the gripper near a button presses it; near a
container door opens it; near an object grasps it. All idempotent, so skills can
re-issue gripper close without flapping.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from harness.envs.base import Env
from harness.tasks import generate_task
from harness.tasks.base import TaskSpec
from harness.types import Obs, Action, StepResult, ActionSpace, ObservationSpace


def _d(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a[:2] - b[:2]))


class KitchenEnv(Env):
    name = "kitchen"

    MAX_DELTA = 0.25

    def __init__(
        self,
        *,
        task_spec: Optional[TaskSpec] = None,
        task: str = "cook_bread",
        seed: int = 0,
        difficulty: float = 0.5,
        max_episode_steps: int = 200,
        render_size: int = 256,
        **kwargs: Any,
    ) -> None:
        self._seed = seed
        self._max_steps = max_episode_steps
        self._render_size = render_size
        self._steps = 0
        self._trail: list = []

        if task_spec is None:
            task_spec = generate_task(task, seed=seed, difficulty=difficulty, dims=2)
        elif isinstance(task_spec, dict):
            task_spec = TaskSpec(**task_spec)
        self.task_spec = task_spec

        self._objects: dict[str, np.ndarray] = {o["name"]: np.asarray(o["pos"], dtype=float) for o in task_spec.objects}
        self._obj_home = {k: v.copy() for k, v in self._objects.items()}
        self._containers: dict[str, dict] = {c["name"]: dict(c) for c in task_spec.containers}
        self._buttons: dict[str, dict] = {b["name"]: dict(b) for b in task_spec.buttons}
        self._steps_spec: list[dict] = list(task_spec.steps)

        self._ee_home = np.asarray(task_spec.ee_start, dtype=float)
        self._ee = self._ee_home.copy()
        self._gripper = 0.0
        self._grasped: Optional[str] = None

        self._grasp_radius = task_spec.params.get("grasp_radius", 0.15)
        self._actuate_radius = task_spec.params.get("actuate_radius", 0.12)
        self._inside_radius = task_spec.params.get("inside_radius", 0.12)

    # -- spaces ----------------------------------------------------------- #
    @property
    def observation_space(self) -> ObservationSpace:
        return ObservationSpace(
            state_dim=4,
            state_names=("ee_x", "ee_y", "gripper", "grasped"),
            has_image=True,
            image_shape=(self._render_size, self._render_size, 3),
            description="2D kitchen: end-effector + gripper; objects/containers/buttons in text state.",
        )

    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(
            kind="ee_delta",
            dim=2,
            low=np.array([-self.MAX_DELTA, -self.MAX_DELTA], dtype=np.float32),
            high=np.array([self.MAX_DELTA, self.MAX_DELTA], dtype=np.float32),
            gripper_dim=1,
            description="move by (dx, dy) in [-0.25, 0.25]; gripper 0 (open) or 1 (close).",
        )

    # -- lifecycle -------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None) -> Obs:
        self._ee = self._ee_home.copy()
        self._objects = {k: v.copy() for k, v in self._obj_home.items()}
        for c in self._containers.values():
            c["open"] = False
        for b in self._buttons.values():
            b["pressed"] = False
        self._gripper = 0.0
        self._grasped = None
        self._steps = 0
        self._trail = [self._ee.copy()]
        return self._observe()

    def step(self, action: Action) -> StepResult:
        self._steps += 1

        if action.kind in ("ee_delta", "move", "delta") and action.value is not None:
            delta = np.clip(np.asarray(action.value, dtype=float).ravel()[:2], -self.MAX_DELTA, self.MAX_DELTA)
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

        success = self._check_success()
        cost = self._primary_cost()
        reward = 1.0 if success else -cost - 0.01
        truncated = self._steps >= self._max_steps

        return StepResult(
            obs=self._observe(),
            reward=reward,
            terminated=success,
            truncated=truncated,
            info={"success": success, "cost": cost},
        )

    def _set_gripper(self, value: float) -> None:
        close = float(value) > 0.5
        self._gripper = 1.0 if close else 0.0
        if close and self._grasped is None:
            ee = self._ee
            # press button (idempotent)
            for b in self._buttons.values():
                if _d(ee, b["pos"]) < self._actuate_radius:
                    b["pressed"] = True
                    return
            # open container door (idempotent)
            for c in self._containers.values():
                if _d(ee, c["door"]) < self._actuate_radius:
                    c["open"] = True
                    return
            # grasp nearest object
            nearest, best = None, self._grasp_radius
            for name, pos in self._objects.items():
                d = _d(ee, pos)
                if d < best:
                    nearest, best = name, d
            self._grasped = nearest
        elif not close:
            self._grasped = None

    # -- success / cost --------------------------------------------------- #
    def _check_success(self) -> bool:
        if not self._steps_spec:
            return False
        return all(self.check_subgoal(s.get("check", "")) for s in self._steps_spec)

    def _primary_cost(self) -> float:
        total = 0.0
        for s in self._steps_spec:
            total += (0.0 if self.check_subgoal(s.get("check", "")) else 1.0)
        return total

    def check_subgoal(self, name: str) -> bool:
        if name == "oven_open":
            return all(c["open"] for c in self._containers.values()) if self._containers else False
        if name == "bread_in_oven":
            return self._object_in_container("bread", "oven")
        if name == "button_pressed":
            return all(b["pressed"] for b in self._buttons.values()) if self._buttons else False
        return False

    def _object_in_container(self, obj: str, container: str) -> bool:
        if obj not in self._objects or container not in self._containers:
            return False
        interior = self._containers[container]["interior"]
        return _d(self._objects[obj], interior) < self._inside_radius

    # -- observation / query API ----------------------------------------- #
    def _state(self) -> np.ndarray:
        return np.array([self._ee[0], self._ee[1], self._gripper, float(self._grasped is not None)], dtype=np.float32)

    def _observe(self) -> Obs:
        return Obs(state=self._state(), text=self.get_text_state())

    def get_text_state(self) -> str:
        grip = "closed" if self._grasped is not None else "open"
        lines = [f"Task: {self.task_spec.description}"]
        lines.append(f"End-effector: ({self._ee[0]:.3f}, {self._ee[1]:.3f}), gripper={grip}, grasping={self._grasped or 'none'}")
        for name, pos in self._objects.items():
            lines.append(f"Object {name}: ({pos[0]:.3f}, {pos[1]:.3f})")
        for name, c in self._containers.items():
            door = "open" if c["open"] else "closed"
            lines.append(f"Container {name}: interior ({c['interior'][0]:.3f}, {c['interior'][1]:.3f}), door {door}")
        for name, b in self._buttons.items():
            lines.append(f"Button {name}: ({b['pos'][0]:.3f}, {b['pos'][1]:.3f}), pressed={b['pressed']}")
        return "\n".join(lines)

    def list_objects(self) -> list[str]:
        return list(self._objects.keys())

    def get_object_pos(self, name: str):
        return self._objects.get(name)

    def list_containers(self) -> list[str]:
        return list(self._containers.keys())

    def get_container_interior(self, name: str):
        c = self._containers.get(name)
        return np.asarray(c["interior"], dtype=float) if c else None

    def get_door_pos(self, name: str):
        c = self._containers.get(name)
        return np.asarray(c["door"], dtype=float) if c else None

    def is_container_open(self, name: str) -> bool:
        return self._containers.get(name, {}).get("open", False)

    def list_buttons(self) -> list[str]:
        return list(self._buttons.keys())

    def get_button_pos(self, name: str):
        b = self._buttons.get(name)
        return np.asarray(b["pos"], dtype=float) if b else None

    def is_button_pressed(self, name: str) -> bool:
        return self._buttons.get(name, {}).get("pressed", False)

    def get_ee_pos(self):
        return self._ee.copy()

    def is_grasped(self) -> bool:
        return self._grasped is not None

    def grasped_object(self):
        return self._grasped

    # -- rendering -------------------------------------------------------- #
    def render(self) -> Optional[np.ndarray]:
        s = self._render_size
        img = np.zeros((s, s, 3), dtype=np.uint8)
        img[..., :] = (24, 22, 30)

        def px(p):
            x = int(round(np.clip(p[0], 0.0, 1.0) * (s - 1)))
            y = int(round((1.0 - np.clip(p[1], 0.0, 1.0)) * (s - 1)))
            return y, x

        step = max(8, s // 10)
        img[::step, :, :] = (42, 40, 52)
        img[:, ::step, :] = (42, 40, 52)

        # oven interior (darker box) + door marker
        for c in self._containers.values():
            iy, ix = px(c["interior"])
            half = max(5, int(self._inside_radius * s))
            img[iy - half:iy + half + 1, ix - half:ix + half + 1] = (60, 60, 70)
            dy, dx = px(c["door"])
            door_col = (90, 200, 90) if c["open"] else (200, 90, 90)
            img[dy - 2:dy + 3, dx - 2:dx + 3] = door_col

        # buttons
        for b in self._buttons.values():
            by, bx = px(b["pos"])
            r = max(3, int(0.04 * s))
            yy, xx = np.ogrid[:s, :s]
            mask = (xx - bx) ** 2 + (yy - by) ** 2 < r ** 2
            col = (240, 200, 60) if b["pressed"] else (90, 90, 110)
            img[mask] = col

        # objects
        for name, pos in self._objects.items():
            oy, ox = px(pos)
            half = max(3, int(0.05 * s))
            img[oy - half:oy + half + 1, ox - half:ox + half + 1] = (90, 200, 90)

        # end-effector
        ey, ex = px(self._ee)
        half2 = max(3, int(0.035 * s))
        img[ey - half2:ey + half2 + 1, ex - half2:ex + half2 + 1] = (255, 90, 90)
        return img

    def close(self) -> None:
        pass
