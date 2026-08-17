"""Genesis (genesis-world) Franka environment.

Requires: pip install genesis-world  (pulls in PyTorch)

Drives a Franka Panda through the Genesis API and consumes a 3D TaskSpec from
harness.tasks (pick_place, stack, sort, push, reach, obstacle variants). It
implements inverse-kinematics end-effector control, finger control, a kinematic
grasp model, and the object-aware query API so tools/agents can perceive the
scene.

The Genesis API calls follow the official franka_cube example (MJCF Franka,
inverse_kinematics, control_dofs_position). Genesis evolves quickly; if your
installed version differs, this file is the single place to adjust.

Grasp model: kinematic attachment. When the fingers close over an object within
grasp_radius, the object is attached to the end-effector (moved with it); opening
the fingers releases it. This is robust for LLM-driven control. For contact-based
grasping see Genesis' SAP example: examples/sap_coupling/franka_grasp_rigid_cube.py.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from harness.envs.base import Env
from harness.tasks import generate_task
from harness.tasks.base import TaskSpec
from harness.types import Obs, Action, StepResult, ActionSpace, ObservationSpace
from harness.utils.logging import get_logger

log = get_logger("harness.envs.genesis")


class GenesisFrankaEnv(Env):
    name = "genesis"

    _HOME_QPOS = np.array(
        [-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]
    )
    _EE_LINK = "hand"
    _EE_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, gripper pointing down
    _GRASP_OFFSET = np.array([0.0, 0.0, 0.0])  # object follows the EE (kinematic grasp)
    _MAX_EE_DELTA = 0.03
    _CUBE_SIZE = 0.04

    def __init__(
        self,
        *,
        task_spec: Optional[TaskSpec] = None,
        task: str = "pick_place",
        seed: int = 0,
        difficulty: float = 0.5,
        max_episode_steps: int = 200,
        backend: str = "cpu",
        show_viewer: bool = False,
        control_mode: str = "ee_delta",  # "ee_delta" | "joint_position"
        substeps: int = 60,  # physics substeps per action (PD control needs many)
        record_video: bool = False,  # capture frames and write a video on close()
        video_path: str = "video.mp4",
        video_fps: float = 30.0,
        camera_res: tuple = (320, 320),
        robot_file: str = "xml/franka_emika_panda/panda.xml",
        **kwargs: Any,
    ) -> None:
        import genesis as gs  # type: ignore

        self._gs = gs
        self._seed = seed
        self._max_steps = max_episode_steps
        self._control_mode = control_mode
        self._substeps = substeps
        self._record_video = record_video
        self._video_path = video_path
        self._video_fps = video_fps
        self._recorded_frames: list = []
        self._show_viewer = show_viewer
        self._camera_res = camera_res
        self._steps = 0

        if task_spec is None:
            task_spec = generate_task(task, seed=seed, difficulty=difficulty, dims=3)
        elif isinstance(task_spec, dict):
            task_spec = TaskSpec(**task_spec)
        self.task_spec = task_spec

        # -- parse the TaskSpec into scene data -------------------------- #
        self._targets: dict[str, str] = {}
        self._roles: dict[str, str] = {}
        for o in self.task_spec.objects:
            if o.get("target"):
                self._targets[o["name"]] = o["target"]
            if o.get("role"):
                self._roles[o["name"]] = o["role"]
        self._goals: dict[str, np.ndarray] = {k: np.asarray(v, dtype=float) for k, v in self.task_spec.goals.items()}
        self._obj_home: dict[str, np.ndarray] = {o["name"]: np.asarray(o["pos"], dtype=float) for o in self.task_spec.objects}
        self._obstacles: list[dict] = [dict(o) for o in self.task_spec.obstacles]
        self._ee_target = np.asarray(self.task_spec.ee_target, dtype=float) if self.task_spec.ee_target is not None else None

        self._goal_radius = self.task_spec.params.get("goal_radius", 0.06)
        self._grasp_radius = self.task_spec.params.get("grasp_radius", 0.13)
        self._stack_radius = self.task_spec.params.get("stack_radius", 0.07)
        self._stack_height = self.task_spec.params.get("stack_height", 0.03)
        self._target_radius = self.task_spec.params.get("target_radius", 0.05)
        self._require_release = self.task_spec.params.get("require_release", False)

        # -- build the scene --------------------------------------------- #
        gs.init(backend=getattr(gs, backend, gs.cpu), precision="32")
        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.1, -0.5, 1.0), camera_lookat=(0.5, 0.0, 0.2), camera_fov=40, res=(960, 640)
            ),
            sim_options=gs.options.SimOptions(dt=0.01),
            rigid_options=gs.options.RigidOptions(box_box_detection=True),
            show_viewer=show_viewer,
        )
        self._scene = scene
        scene.add_entity(gs.morphs.Plane())
        self._franka = scene.add_entity(gs.morphs.MJCF(file=robot_file))

        self._object_entities: dict[str, Any] = {}
        for o in self.task_spec.objects:
            self._object_entities[o["name"]] = scene.add_entity(
                gs.morphs.Box(size=(self._CUBE_SIZE,) * 3, pos=o["pos"])
            )
        for gname, gpos in self.task_spec.goals.items():
            scene.add_entity(gs.morphs.Box(size=(0.06, 0.06, 0.004), pos=gpos))
        for ob in self.task_spec.obstacles:
            r = float(ob.get("radius", 0.05))
            scene.add_entity(gs.morphs.Box(size=(r * 2.0, r * 2.0, 0.14), pos=ob["pos"]))

        self._cam = scene.add_camera(
            res=camera_res, pos=(1.1, -0.5, 1.0), lookat=(0.5, 0.0, 0.2), fov=40
        )
        scene.build()

        self._arm_dof = np.arange(7)
        self._finger_dof = np.arange(7, 9)
        self._ee_link = self._franka.get_link(self._EE_LINK)
        self._franka.set_dofs_kp([100.0, 100.0], self._finger_dof)
        self._franka.set_dofs_kv([10.0, 10.0], self._finger_dof)
        self._franka.set_qpos(self._HOME_QPOS.copy())
        scene.step()

        self._gripper = 0.0
        self._grasped: Optional[str] = None
        self._grasp_offset: Optional[np.ndarray] = None
        self._collided = False

    # -- spaces ----------------------------------------------------------- #
    @property
    def observation_space(self) -> ObservationSpace:
        return ObservationSpace(
            state_dim=5,
            state_names=("ee_x", "ee_y", "ee_z", "gripper", "grasped"),
            has_image=True,
            image_shape=(self._camera_res[1], self._camera_res[0], 3),
            description="Franka EE pose + gripper + grasped flag; objects/goals in text state.",
        )

    @property
    def action_space(self) -> ActionSpace:
        if self._control_mode == "joint_position":
            return ActionSpace(
                kind="joint_position",
                dim=7,
                low=-np.pi * np.ones(7, dtype=np.float32),
                high=np.pi * np.ones(7, dtype=np.float32),
                joint_names=tuple(f"joint{i}" for i in range(7)),
                description="7 Franka arm joint angles (radians).",
            )
        return ActionSpace(
            kind="ee_delta",
            dim=3,
            low=np.full(3, -self._MAX_EE_DELTA, dtype=np.float32),
            high=np.full(3, self._MAX_EE_DELTA, dtype=np.float32),
            gripper_dim=1,
            description="end-effector delta (dx, dy, dz) in meters; gripper 0 (open) .. 1 (close).",
        )

    # -- lifecycle -------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None) -> Obs:
        self._franka.set_qpos(self._HOME_QPOS.copy())
        self._scene.step()
        for name, ent in self._object_entities.items():
            ent.set_pos(self._obj_home[name].copy())
        self._gripper = 0.0
        self._grasped = None
        self._grasp_offset = None
        self._collided = False
        self._steps = 0
        self._recorded_frames = []
        self._scene.step()
        return self._observe()

    def step(self, action: Action) -> StepResult:
        self._steps += 1

        # 1. determine the arm control target (joint positions)
        arm_target = None
        if action.kind in ("ee_delta", "move", "delta") and action.value is not None:
            delta = np.clip(np.asarray(action.value, dtype=float).ravel()[:3], -self._MAX_EE_DELTA, self._MAX_EE_DELTA)
            arm_target = self._ik(self._get_ee_pos() + delta)
        elif action.kind in ("ee_pose", "move_to") and action.value is not None:
            arm_target = self._ik(np.asarray(action.value, dtype=float).ravel()[:3])
        elif action.kind == "joint_position" and action.value is not None:
            q = np.asarray(action.value, dtype=float).ravel()
            n = len(self._arm_dof)
            if q.size >= n:
                arm_target = q[:n]
            else:
                log.warning("joint_position action has %d values, need %d; ignoring", q.size, n)

        # 2. gripper target
        finger_target = None
        if action.gripper is not None:
            g = float(np.clip(action.gripper, 0.0, 1.0))
            self._gripper = g
            finger_target = np.array([g, g])

        # 3. PD control loop: control_dofs_position sets a target, and the arm
        #    needs many physics substeps to actually reach it. Update grasp +
        #    carried object every substep so the object tracks the gripper.
        for _ in range(self._substeps):
            if arm_target is not None:
                self._franka.control_dofs_position(arm_target, self._arm_dof)
            if finger_target is not None:
                self._franka.control_dofs_position(finger_target, self._finger_dof)
            self._update_grasp()
            if self._grasped is not None:
                off = self._grasp_offset if self._grasp_offset is not None else self._GRASP_OFFSET
                self._object_entities[self._grasped].set_pos(self._get_ee_pos() + off)
            self._scene.step()
            if self._record_video:
                self._recorded_frames.append(self.render())

        # 5. outcome
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

    # -- motion / grasp --------------------------------------------------- #
    def _ik(self, target):
        """Solve IK for an EE position; return arm joint target (7,) or None."""
        t = np.asarray(target, dtype=float).ravel()
        if t.size >= 3:
            t[2] = float(np.clip(t[2], 0.12, 0.45))  # Franka wrist reachable z band
        try:
            qpos = self._franka.inverse_kinematics(link=self._ee_link, pos=t, quat=self._EE_QUAT)
            if qpos is None:
                return None
            return np.asarray(qpos)[:-2]
        except Exception:
            return None

    def _update_grasp(self) -> None:
        if self._gripper > 0.5 and self._grasped is None:
            ee = self._get_ee_pos()
            nearest, best = None, self._grasp_radius
            for name, ent in self._object_entities.items():
                d = float(np.linalg.norm(ee - np.asarray(ent.get_pos(), dtype=float)))
                if d < best:
                    nearest, best = name, d
            if nearest is not None:
                self._grasped = nearest
                # remember the offset so the object keeps its height while carried
                self._grasp_offset = np.asarray(self._object_entities[nearest].get_pos(), dtype=float) - ee
        elif self._gripper <= 0.5:
            self._grasped = None
            self._grasp_offset = None

    # -- success / collision --------------------------------------------- #
    def _check_collision(self) -> bool:
        ee = self._get_ee_pos()
        for o in self._obstacles:
            c = np.asarray(o["pos"], dtype=float)
            r = float(o.get("radius", 0.05))
            if float(np.linalg.norm(ee[:2] - c[:2])) < r:
                return True
            if self._grasped is not None:
                obj = np.asarray(self._object_entities[self._grasped].get_pos(), dtype=float)
                if float(np.linalg.norm(obj[:2] - c[:2])) < r:
                    return True
        return False

    def _check_success(self) -> bool:
        kind = self.task_spec.kind
        if kind in ("reach", "reach_avoid"):
            if self._ee_target is None:
                return False
            return float(np.linalg.norm(self._get_ee_pos() - self._ee_target)) < self._target_radius
        if kind == "stack":
            top = self._object_entities.get("top")
            base = self._object_entities.get("base")
            if top is None or base is None:
                return False
            tp = np.asarray(top.get_pos(), dtype=float)
            bp = np.asarray(base.get_pos(), dtype=float)
            xy = float(np.linalg.norm(tp[:2] - bp[:2]))
            stacked = (tp[2] - bp[2]) > self._stack_height
            return xy < self._stack_radius and stacked and self._grasped != "top"
        if not self._object_entities:
            return False
        ok = True
        for name, ent in self._object_entities.items():
            goal_name = self._targets.get(name)
            if goal_name is None:
                continue
            goal = self._goals.get(goal_name)
            if goal is None:
                ok = False
                continue
            p = np.asarray(ent.get_pos(), dtype=float)
            if float(np.linalg.norm(p[:2] - goal[:2])) >= self._goal_radius:
                ok = False
            if self._require_release and self._grasped == name:
                ok = False
        return ok

    def _primary_cost(self) -> float:
        kind = self.task_spec.kind
        if kind in ("reach", "reach_avoid") and self._ee_target is not None:
            return float(np.linalg.norm(self._get_ee_pos() - self._ee_target))
        if kind == "stack" and "top" in self._object_entities and "base" in self._object_entities:
            tp = np.asarray(self._object_entities["top"].get_pos(), dtype=float)
            bp = np.asarray(self._object_entities["base"].get_pos(), dtype=float)
            return float(np.linalg.norm(tp[:2] - bp[:2])) + float(abs(tp[2] - (bp[2] + self._CUBE_SIZE)))
        total = 0.0
        for name, ent in self._object_entities.items():
            gn = self._targets.get(name)
            if gn and gn in self._goals:
                total += float(np.linalg.norm(np.asarray(ent.get_pos(), dtype=float)[:2] - self._goals[gn][:2]))
        return total

    # -- observation / query API ----------------------------------------- #
    def _object_names(self) -> list[str]:
        return list(self._object_entities.keys())

    def _get_ee_pos(self) -> np.ndarray:
        return np.asarray(self._ee_link.get_pos(), dtype=float)

    def _observe(self) -> Obs:
        ee = self._get_ee_pos()
        state = np.array([ee[0], ee[1], ee[2], self._gripper, float(self._grasped is not None)], dtype=np.float32)
        return Obs(state=state, text=self.get_text_state())

    def get_text_state(self) -> str:
        ee = self._get_ee_pos()
        grip = "closed" if self._grasped is not None else "open"
        lines = [f"Task: {self.task_spec.description}"]
        lines.append(f"End-effector: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}), gripper={grip}, grasping={self._grasped or 'none'}")
        for name, ent in self._object_entities.items():
            p = np.asarray(ent.get_pos(), dtype=float)
            tgt = f" -> {self._targets[name]}" if name in self._targets else ""
            lines.append(f"Object {name}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}){tgt}")
        for name, pos in self._goals.items():
            lines.append(f"Goal {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        for o in self._obstacles:
            lines.append(f"Obstacle {o['name']}: ({o['pos'][0]:.3f}, {o['pos'][1]:.3f}) r={o.get('radius', 0.05):.2f}")
        lines.append(f"Distance to success: {self._primary_cost():.3f}")
        return "\n".join(lines)

    def list_objects(self) -> list[str]:
        return list(self._object_entities.keys())

    def get_object_pos(self, name: str):
        ent = self._object_entities.get(name)
        return np.asarray(ent.get_pos(), dtype=float) if ent is not None else None

    def list_goals(self) -> list[str]:
        return list(self._goals.keys())

    def get_goal_pos(self, name: str):
        return self._goals.get(name)

    def list_obstacles(self) -> list[str]:
        return [o["name"] for o in self._obstacles]

    def get_ee_pos(self):
        return self._get_ee_pos()

    def is_grasped(self) -> bool:
        return self._grasped is not None

    def grasped_object(self):
        return self._grasped

    # -- rendering -------------------------------------------------------- #
    def render(self) -> Optional[np.ndarray]:
        try:
            res = self._cam.render(rgb=True)
        except Exception:
            return None
        if isinstance(res, dict):
            rgb = res.get("rgb")
        elif isinstance(res, tuple):
            rgb = res[0]
        else:
            rgb = res
        if rgb is None:
            return None
        rgb = np.asarray(rgb)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        return rgb.astype(np.uint8)

    def save_video(self, path=None, fps=None):
        from harness.viz.video import write_video

        path = path or self._video_path
        fps = fps if fps is not None else self._video_fps
        if not self._recorded_frames:
            self._recorded_frames.append(self.render())
        return write_video(self._recorded_frames, path, fps=fps)

    def close(self) -> None:
        if self._record_video:
            try:
                self.save_video()
            except Exception as e:
                log.warning("video save failed: %s", e)


# Backward-compatible alias
GenesisEnv = GenesisFrankaEnv
