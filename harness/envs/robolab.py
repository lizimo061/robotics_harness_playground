"""RoboLab (NVIDIA Isaac Lab) environment adapter.

RoboLab is a 120+ task manipulation benchmark built on Isaac Lab / Isaac Sim.
This adapter wraps a RoboLab task behind the harness Env interface so the
harness agent (tools / skills / json / code modes) can drive it in-process.

Requires: RoboLab installed in its own venv (Linux + CUDA + Isaac Sim).
See docs/modules/robolab.md. The heavy imports are deferred to __init__ so this
module (and the rest of the harness) imports cleanly without RoboLab installed.

Interface notes (verify against your RoboLab version on the Linux box):
- robolab.core.environments.runtime.create_env(task, device, num_envs, use_fabric)
    -> (env, env_cfg)
- env_cfg.instruction -> the language instruction (used as the task description)
- use_fabric=False -> gymnasium-style env (reset()/step() with an obs dict);
  use_fabric=True  -> torchrl RLTaskEnv (TensorDict). We default to False (simpler).
- auto_register_droid_envs() must run before get_envs()/create_env().
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from harness.envs.base import Env
from harness.types import Obs, Action, StepResult, ActionSpace, ObservationSpace
from harness.utils.logging import get_logger

log = get_logger("harness.envs.robolab")


def _fmt_vec(v) -> str:
    a = np.asarray(v).ravel()
    return "(" + ", ".join(f"{float(x):.3f}" for x in a) + ")"


def _rotate_by_quat(vec, quat) -> np.ndarray:
    """Rotate `vec` by the (w, x, y, z) quaternion `quat`.

    Needed because the fingertip offset is fixed in the *gripper's* frame, not the
    world's. Assuming it always points along world -z is right only while the
    gripper hangs straight down; the moment the agent tilts the wrist, a fixed
    world-space offset aims the grasp somewhere else entirely.
    """
    v = np.asarray(vec, dtype=np.float32).ravel()[:3]
    q = np.asarray(quat, dtype=np.float32).ravel()[:4]
    n = float(np.linalg.norm(q))
    if n < 1e-6:
        return v
    w, x, y, z = q / n
    # v' = v + 2w(q_vec x v) + 2 q_vec x (q_vec x v)
    qv = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(qv, v)
    return (v + w * t + np.cross(qv, t)).astype(np.float32)


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    return np.asarray(x)


class RoboLabEnv(Env):
    name = "robolab"

    def __init__(
        self,
        task: str,
        *,
        num_envs: int = 1,
        device: str = "cuda:0",
        use_fabric: bool = False,
        headless: bool = True,
        action_mode: str = "ee_pose",  # "ee_pose" (abs IK) | "ee_delta" (rel IK) | "joint_position"
        #: Distance from the controlled body to the fingertips, in metres.
        #: RoboLab's IK drives the gripper's BASE FLANGE (body_offset is left
        #: commented out in DroidIKActionCfg) and the Robotiq 2F-85 fingertips are
        #: 162.8mm below it, per the spec figure RoboLab cites. Commanding a
        #: grasp at an object's own z therefore drives the fingers ~16cm through
        #: the table: the IK saturates against it and nothing is ever picked up.
        #: The adapter treats tool coordinates as fingertip space and converts, so
        #: "move to the cube" means the fingers, not the flange.
        tcp_offset: float = 0.1628,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        try:
            from robolab.core.environments.factory import get_envs  # noqa: F401
            from robolab.core.environments.runtime import create_env, end_episode  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "RoboLab (Isaac Lab / Isaac Sim) is not installed. "
                "Install it in its own venv on Linux + CUDA; see docs/modules/robolab.md."
            ) from e

        self._create_env = create_env
        self._end_episode = end_episode

        # The action mode decides which registration to use, and getting this
        # wrong is silent. The joint-position flavour takes SEVEN ABSOLUTE JOINT
        # ANGLES; feeding it a Cartesian (dx, dy, dz) writes those numbers into
        # joints 1-3 and zeroes the rest, so the arm snaps to the same near-zero
        # pose every step and never approaches anything. It looks like the model
        # failing. The IK flavours take Cartesian targets, which is what the
        # move_to / move_delta tools actually mean.
        self._auto_register, suffix = self._registrar_for(action_mode)
        # The suffix must be requested explicitly. RoboLab's abs/rel IK
        # registrars document that they append "AbsIK"/"RelIK" to every env name,
        # but they forward a caller-supplied `env_postfix` that defaults to empty
        # -- so by default all three flavours register under the SAME names and
        # each call silently replaces the last (hence RoboLab's own
        # "previous registration will be replaced" warnings). Passing it keeps the
        # flavours distinct, so the name we create is the interface we asked for.
        self._auto_register(**({"env_postfix": suffix} if suffix else {}))

        task_name = self._resolve_task(get_envs, task, suffix)

        self._env, self._env_cfg = self._create_env(
            task_name, device=device, num_envs=num_envs, use_fabric=use_fabric
        )
        self.task = task_name
        self._instruction = str(getattr(self._env_cfg, "instruction", task_name) or task_name)
        self._action_mode = action_mode
        self._tcp_offset = float(tcp_offset)
        self._seed = seed
        self._use_fabric = use_fabric
        self._num_envs = num_envs
        self._last_image = None
        self._last_proprio: dict = {}
        self._step_idx = 0
        #: Commanded orientation, held as a SETPOINT rather than read back from
        #: the measurement each step. IsaacLab's differential IK tracks position
        #: cleanly but drifts in orientation (RoboLab's own DroidIKActionCfg
        #: docstring says so), and echoing the drifted measurement back as the
        #: next command compounds it -- measured here going from (0.707,0,0.707,0)
        #: at reset to (-0.81,-0.08,-0.57,0.10) after a few dozen steps.
        self._quat_setpoint: Optional[np.ndarray] = None

    #: action_mode -> (registration module, function, env-name suffix).
    #: RoboLab registers one env per flavour and distinguishes them by suffix,
    #: so the flavour is part of the task name.
    _REGISTRARS = {
        "ee_pose": ("robolab.registrations.droid.auto_env_registrations_abs_ik",
                    "auto_register_droid_abs_ik_envs", "AbsIK"),
        "ee_delta": ("robolab.registrations.droid.auto_env_registrations_rel_ik",
                     "auto_register_droid_rel_ik_envs", "RelIK"),
        "joint_position": ("robolab.registrations.droid.auto_env_registrations_jointpos",
                           "auto_register_droid_envs", ""),
    }

    @classmethod
    def _registrar_for(cls, action_mode: str):
        import importlib

        if action_mode not in cls._REGISTRARS:
            raise ValueError(
                f"unknown action_mode {action_mode!r}; expected one of "
                f"{sorted(cls._REGISTRARS)}")
        module_name, func_name, suffix = cls._REGISTRARS[action_mode]
        module = importlib.import_module(module_name)
        return getattr(module, func_name), suffix

    @staticmethod
    def _resolve_task(get_envs, task: str, suffix: str) -> str:
        """Find the registered env for this task and action flavour.

        A bare task name can match several flavours once more than one is
        registered, and picking the first would silently hand back a different
        control interface than the caller asked for.
        """
        if not task:
            found = get_envs()
            if not found:
                raise ValueError("no RoboLab tasks are registered")
            return found[0]

        def query(name):
            # get_envs raises for an unknown name rather than returning empty
            try:
                return get_envs(task=[name]) or []
            except Exception:  # noqa: BLE001 - unknown name for this registry
                return []

        wanted = f"{task}{suffix}"
        for candidate in (wanted, task):
            exact = [f for f in query(candidate) if str(f) == wanted]
            if exact:
                return exact[0]
        found = query(task)
        if found:
            log.warning("no %r variant of %r; falling back to %s -- verify its "
                        "action space matches the requested mode",
                        suffix or "joint-position", task, found[0])
            return found[0]
        raise ValueError(f"RoboLab task '{task}' not found in the factory")

    # -- spaces ----------------------------------------------------------- #
    @property
    def observation_space(self) -> ObservationSpace:
        return ObservationSpace(
            state_dim=0,
            has_image=True,
            description=f"RoboLab '{self.task}': {self._instruction}",
        )

    @property
    def action_space(self) -> ActionSpace:
        sp = getattr(self._env, "action_space", None)
        if sp is None:
            return ActionSpace(kind=self._action_mode, dim=0, description="unknown action space")
        dim = int(np.prod(sp.shape)) if hasattr(sp, "shape") else 0
        low = np.asarray(sp.low, dtype=np.float32).ravel() if hasattr(sp, "low") else -np.ones(dim, np.float32)
        high = np.asarray(sp.high, dtype=np.float32).ravel() if hasattr(sp, "high") else np.ones(dim, np.float32)
        kind = "ee_delta" if self._action_mode == "ee_delta" else "joint_position"
        return ActionSpace(
            kind=kind, dim=dim, low=low, high=high,
            description=f"RoboLab {self._action_mode} action (dim {dim}); last dim is usually the gripper.",
        )

    # -- lifecycle -------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None) -> Obs:
        self._step_idx = 0
        self._quat_setpoint = None  # re-seeded from the fresh pose below
        out = self._env.reset()
        obs, info = self._unwrap_reset(out)
        return self._to_obs(obs, info)

    def step(self, action: Action) -> StepResult:
        self._step_idx += 1
        act = self._to_env_action(action)
        out = self._env.step(act)
        obs, reward, terminated, truncated, info = self._unwrap_step(out)
        info = dict(info or {})
        success = self._extract_success(info)
        info["success"] = success

        # reward/terminated/truncated come back as CUDA tensors; np.asarray on
        # one raises "can't convert cuda:0 device type tensor to numpy".
        def _scalar(x, cast, default):
            if x is None:
                return default
            arr = _to_numpy(x)
            arr = np.asarray(arr).ravel()
            return cast(arr[0]) if arr.size else default

        return StepResult(
            obs=self._to_obs(obs, info),
            reward=_scalar(reward, float, 0.0),
            terminated=_scalar(terminated, bool, False),
            truncated=_scalar(truncated, bool, False),
            info=info,
        )

    def close(self) -> None:
        try:
            self._end_episode(self._env)
        except Exception:
            pass
        try:
            self._env.close()
        except Exception:
            pass

    # -- obs / action mapping -------------------------------------------- #
    def _unwrap_reset(self, out):
        # gym: (obs, info); torchrl/tensordict: a single TensorDict
        if isinstance(out, tuple) and len(out) >= 2:
            return out[0], out[1]
        return out, {}

    def _unwrap_step(self, out):
        # gym: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple):
            o = out[0]
            r = out[1] if len(out) > 1 else 0.0
            t = out[2] if len(out) > 2 else False
            tr = out[3] if len(out) > 3 else False
            i = out[4] if len(out) > 4 else {}
            return o, r, t, tr, i
        # torchrl/tensordict: a single TensorDict
        return out, out.get("reward", 0.0), out.get("terminated", out.get("done", False)), out.get("truncated", False), {}

    def _to_obs(self, obs, info) -> Obs:
        # image/proprio are read from the RAW obs: _to_numpy flattens torch
        # tensors but also loses the group structure these scans rely on.
        self._last_image = self._extract_image(obs)
        self._last_proprio = self._extract_proprio(obs)
        o = _to_numpy(obs)
        return Obs(
            state=self._extract_state(o),
            image=self._last_image,
            text=self.get_text_state(),
            info=dict(info or {}),
        )

    def _extract_state(self, obs):
        if obs is None:
            return None
        if isinstance(obs, dict):
            for key in ("observation", "proprio", "state"):
                if key in obs:
                    return self._flatten(obs[key])
            policy = obs.get("policy")
            if isinstance(policy, dict):
                for key in ("observation", "proprio", "state"):
                    if key in policy:
                        return self._flatten(policy[key])
            for v in obs.values():  # fall back to the first array value
                f = self._flatten(v)
                if f is not None:
                    return f
            return None
        return self._flatten(obs)

    def _extract_image(self, obs):
        """Find a camera frame anywhere in the observation.

        RoboLab nests observations in groups (image_obs, proprio_obs,
        viewport_cam, ...) whose camera terms are named per scene camera --
        `over_shoulder_left_camera`, `egocentric_mirrored_camera`,
        `wrist_camera` -- so a fixed key list finds nothing. Scan for any
        image-shaped tensor instead, and rank the views.

        The ranking is what makes a video worth watching: RoboLab documents the
        mirrored/egocentric third-person camera as the one intended for video
        recording, an over-shoulder view usually frames the arm, and the wrist
        camera moves *with* the gripper -- so a wrist recording shows the scene
        sliding around with the robot itself never in shot, which is exactly the
        footage you cannot judge a manipulation attempt from.
        """
        best = None
        best_rank = -1
        best_path = ""

        def visit(node, path=""):
            nonlocal best, best_rank
            if node is None:
                return
            if hasattr(node, "shape"):
                shape = tuple(node.shape)
                # (..., H, W, C) with C in {3, 4} and a plausible frame size
                if len(shape) >= 3 and shape[-1] in (3, 4) and shape[-2] >= 32 and shape[-3] >= 32:
                    low = path.lower()
                    if "viewport" in low:
                        rank = 4
                    elif "mirror" in low or "egocentric" in low or "third" in low:
                        rank = 3
                    elif "shoulder" in low or "head" in low or "front" in low:
                        rank = 2
                    elif "wrist" in low:
                        rank = 0  # moves with the gripper; the robot is never in it
                    else:
                        rank = 1
                    if rank > best_rank:
                        best, best_rank, best_path = node, rank, path
                return
            keys = getattr(node, "keys", None)
            if keys is None:
                return
            for k in list(keys()):
                visit(node[k], f"{path}/{k}")

        visit(obs)
        if best is None:
            return None
        if best_path != getattr(self, "_image_source", None):
            self._image_source = best_path
            log.info("rendering from camera %s (rank %d)", best_path, best_rank)

        arr = _to_numpy(best)
        while arr.ndim > 3:  # drop the env-batch dimension
            arr = arr[0]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = (arr * 255 if float(arr.max() or 0) <= 1.0 else arr)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _extract_proprio(self, obs) -> dict:
        """Pull end-effector and joint state out of the observation.

        Without this the agent sees only the instruction string and is acting
        blind. These terms live under the proprio observation group.
        """
        out: dict = {}
        wanted = ("ee_pos", "ee_quat", "arm_joint_pos", "gripper_pos")

        def visit(node, path=""):
            if node is None:
                return
            if hasattr(node, "shape"):
                leaf = path.rsplit("/", 1)[-1]
                if leaf in wanted and leaf not in out:
                    arr = _to_numpy(node)
                    while arr.ndim > 1:
                        arr = arr[0]
                    out[leaf] = arr
                return
            keys = getattr(node, "keys", None)
            if keys is None:
                return
            for k in list(keys()):
                visit(node[k], f"{path}/{k}")

        visit(obs)
        return out

    def _flatten(self, x):
        x = _to_numpy(x)
        if x is None:
            return None
        if isinstance(x, dict):
            parts = []
            for v in x.values():
                f = self._flatten(v)
                if f is not None:
                    parts.append(f)
            return np.concatenate(parts) if parts else None
        return np.asarray(x).ravel().astype(np.float32)

    def _clip_delta(self, delta: np.ndarray, dim: int) -> np.ndarray:
        """Clip a positional delta to the env's own per-step action limits.

        A single step cannot cross the workspace: the controller saturates, and
        an unclipped command wastes the step while telling the agent nothing. By
        clipping here the agent's feedback (its new pose) reflects the real
        motion, so it can iterate toward the target across turns.
        """
        limit = getattr(self, "_delta_limit", None)
        if limit is None:
            sp = getattr(self._env, "action_space", None)
            high = getattr(sp, "high", None)
            arr = _to_numpy(high)
            if arr is not None:
                arr = np.abs(np.asarray(arr, dtype=np.float32).ravel())
                # the last dim is the gripper; positional limits are the rest
                positional = arr[:-1] if arr.size > 1 else arr
                finite = positional[np.isfinite(positional)]
                limit = float(np.min(finite)) if finite.size else 1.0
            else:
                limit = 1.0
            self._delta_limit = limit
        return np.clip(delta, -limit, limit).astype(np.float32)

    def _current_quat(self) -> np.ndarray:
        """Current end-effector orientation as (qw, qx, qy, qz).

        AbsIK commands an absolute pose, so the orientation slots must hold a
        valid unit quaternion. Zero-filling them (the obvious default) is not a
        rotation at all and the differential IK diverges, so an agent that only
        supplies a position keeps whatever orientation it currently has.
        """
        q = (self._last_proprio or {}).get("ee_quat")
        arr = None if q is None else np.asarray(_to_numpy(q), dtype=np.float32).ravel()
        if arr is None or arr.size < 4 or not np.isfinite(arr[:4]).all():
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity
        arr = arr[:4]
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0, 0.0], np.float32)

    def _pack_action(self, action: Action, dim: int) -> np.ndarray:
        """Lay the requested motion out the way this action flavour expects.

        Three different contracts, and using the wrong one fails silently:
          joint_position -> 7 absolute joint angles + gripper
          ee_pose (AbsIK) -> (x, y, z, qw, qx, qy, qz) + gripper, robot-root frame
          ee_delta (RelIK) -> (dx, dy, dz, droll, dpitch, dyaw) + gripper
        """
        v = action.value
        v = (np.zeros(0, dtype=np.float32) if v is None
             else np.asarray(v, dtype=np.float32).ravel())
        absolute = action.kind in ("ee_pose", "pose", "absolute")
        relative = action.kind in ("ee_delta", "delta", "move")
        # A 7-vector carries the orientation the caller wants: (x,y,z,qw,qx,qy,qz).
        # Without a way to command it, a grasp is impossible -- the wrist keeps
        # whatever orientation it started in.
        if v.size >= 7:
            q = np.asarray(v[3:7], dtype=np.float32)
            n = float(np.linalg.norm(q))
            if n > 1e-6:
                self._quat_setpoint = q / n
            v = v[:3]

        if self._action_mode == "ee_pose":
            cur = self.get_ee_pos()
            cur = (np.zeros(3, dtype=np.float32) if cur is None
                   else np.asarray(cur, dtype=np.float32).ravel()[:3])
            if absolute and v.size >= 2:
                target = cur.copy()
                target[:min(3, v.size)] = v[:min(3, v.size)]
            elif relative and v.size >= 1:
                target = cur.copy()
                target[:min(3, v.size)] += v[:min(3, v.size)]
            else:
                target = cur  # hold position (e.g. a pure gripper command)
            # the agent aims the fingertips; the IK steers the flange, so undo the
            # gripper's own offset in whatever direction the wrist is pointing
            flange = target.copy()
            flange[:3] = flange[:3] - self._tcp_vector()
            out = np.concatenate([flange, self._target_quat()])
        elif self._action_mode == "ee_delta":
            delta = np.zeros(3, dtype=np.float32)
            if absolute and v.size >= 2:
                cur = self.get_ee_pos()
                if cur is None:
                    log.warning("absolute move requested but the end-effector pose "
                                "is unknown; holding position")
                else:
                    cur = np.asarray(cur, dtype=np.float32).ravel()
                    n = min(3, v.size, cur.size)
                    delta[:n] = v[:n] - cur[:n]
            elif v.size:
                delta[:min(3, v.size)] = v[:min(3, v.size)]
            clipped = self._clip_delta(delta, dim)
            # Saturation is invisible to the agent otherwise: it asks to move to a
            # target, the arm moves a few cm, and nothing says why. Recording it
            # lets get_text_state() tell the agent to keep going, which turns a
            # dead end into an iterable control loop.
            remaining = float(np.linalg.norm(delta - clipped))
            self._last_move_clipped = remaining > 1e-4
            self._last_move_remaining = remaining
            out = np.concatenate([clipped, np.zeros(3, dtype=np.float32)])  # no rotation
        else:  # joint_position: the values are joint targets already
            out = v.astype(np.float32, copy=True)

        if action.gripper is not None:
            out = np.concatenate([out, np.array([float(action.gripper)], np.float32)])
        return out

    def _to_env_action(self, action: Action):
        sp = getattr(self._env, "action_space", None)
        dim = int(np.prod(sp.shape)) if sp is not None and hasattr(sp, "shape") else 0
        v = self._pack_action(action, dim)

        if action.gripper is not None and dim and v.size < dim:
            # pad between the motion block and the gripper, which is the last dim
            v = np.concatenate([v[:-1], np.zeros(dim - v.size, np.float32), v[-1:]])
        if dim:
            if v.size < dim:
                v = np.concatenate([v, np.zeros(dim - v.size, dtype=np.float32)])
            v = v[:dim]

        # Isaac Lab always expects a batched torch tensor on the sim device --
        # its action manager calls action.to(device), so a numpy array raises
        # AttributeError before the step ever runs.
        v = v.reshape(self._num_envs, -1) if self._num_envs > 1 else v.reshape(1, -1)
        try:
            import torch

            device = getattr(self._env, "device", None) or "cuda:0"
            return torch.as_tensor(v, dtype=torch.float32, device=device)
        except ImportError:  # pragma: no cover - torch ships with Isaac Lab
            return v

    def _extract_success(self, info: dict) -> bool:
        for key in ("success", "is_success", "task_success", "goal_achieved"):
            if key in info:
                arr = np.asarray(_to_numpy(info[key])).ravel()
                if arr.size:
                    return bool(arr[0])
        return False

    # -- text / subgoal --------------------------------------------------- #
    # -- scene query API -------------------------------------------------- #
    # A text-only model cannot see the camera frame, so without these it knows
    # the instruction and its own arm pose and nothing else: it cannot locate the
    # cube it is asked to pick up. Measured effect of that blindness on a live
    # DeepSeek run: it called `done` after a single environment step, because
    # flailing and stopping are indistinguishable when you have no feedback.
    #
    # The poses come from Isaac's ground-truth scene state, which makes this a
    # privileged-state evaluation -- it measures planning and grounding, not
    # perception. That is the same contract the tabletop env offers, and it is
    # worth stating plainly in any result: a model doing well here has not been
    # shown to perceive anything.

    def _scene(self):
        unwrapped = getattr(self._env, "unwrapped", self._env)
        return getattr(unwrapped, "scene", None)

    def _scene_objects(self) -> dict:
        """Map name -> scene entity for every manipulable rigid body.

        IsaacLab's InteractiveScene holds rigid objects in a dict-like
        ``rigid_objects``; articulations (the robot) live separately and are
        excluded, since "objects" here means things the task is about.
        """
        scene = self._scene()
        if scene is None:
            return {}
        found: dict = {}
        for attr in ("rigid_objects", "rigid_object_collections", "deformable_objects"):
            group = getattr(scene, attr, None)
            if not group:
                continue
            try:
                for name in list(group):
                    found[str(name)] = group[name]
            except Exception:  # noqa: BLE001 - not dict-like; skip this group
                continue
        return found

    def _entity_pos(self, entity):
        """Position in the env frame, which is what the actions are relative to.

        Isaac reports world coordinates; with num_envs > 1 every env is offset on
        a grid, so a world pose is not comparable to the arm's own frame.
        """
        data = getattr(entity, "data", None)
        pos = getattr(data, "root_pos_w", None)
        if pos is None:
            pos = getattr(data, "object_pos_w", None)
        if pos is None:
            return None
        arr = _to_numpy(pos)
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, 3)[0]
        scene = self._scene()
        origins = _to_numpy(getattr(scene, "env_origins", None))
        if origins is not None:
            origins = np.asarray(origins, dtype=np.float32).reshape(-1, 3)
            if len(origins):
                arr = arr - origins[0]
        return arr

    def list_objects(self) -> list:
        return sorted(self._scene_objects())

    def get_object_pos(self, name: str):
        entity = self._scene_objects().get(str(name))
        return None if entity is None else self._entity_pos(entity)

    def list_goals(self) -> list:
        """RoboLab states goals in language, not as coordinates.

        Returning [] is honest: a container the task names ("the bowl") shows up
        in list_objects with a real pose, so the agent is not missing anything --
        inventing goal coordinates would be.
        """
        return []

    def get_text_state(self) -> str:
        """Instruction, scene objects, and whatever proprioception we have.

        Returning the bare instruction leaves the agent blind: identical text
        every step, no feedback that an action did anything.
        """
        lines = [f"Task: {self._instruction}"]
        objects = self._scene_objects()
        if objects:
            lines.append("Objects in the scene (x, y, z in the robot's frame):")
            for name in sorted(objects):
                pos = self._entity_pos(objects[name])
                lines.append(f"  {name}: " + (_fmt_vec(pos) if pos is not None else "unknown"))
        p = self._last_proprio or {}
        if "ee_pos" in p:
            lines.append("End-effector position: " + _fmt_vec(p["ee_pos"]))
        if "ee_quat" in p:
            lines.append("End-effector orientation (w,x,y,z): " + _fmt_vec(p["ee_quat"]))
        if "arm_joint_pos" in p:
            lines.append("Arm joint positions (rad): " + _fmt_vec(p["arm_joint_pos"]))
        if "gripper_pos" in p:
            g = np.asarray(p["gripper_pos"]).ravel()
            lines.append(f"Gripper: {_fmt_vec(g)} (higher = more closed)")
        if getattr(self, "_last_move_clipped", False):
            lines.append(
                f"The last move was capped at this environment's per-step limit; "
                f"{self._last_move_remaining:.3f} m of it remains. Repeat the move "
                f"to keep approaching the target.")
        if self._step_idx:
            lines.append(f"Step {self._step_idx}.")
        return "\n".join(lines)

    def get_ee_pos(self):
        """Fingertip position, not the controlled flange.

        The agent compares this against object positions and aims at them, so it
        must be expressed in the same space its move commands are interpreted in.
        Reporting the flange while accepting fingertip targets would put a fixed
        16cm error in every comparison the agent makes.
        """
        p = (self._last_proprio or {}).get("ee_pos")
        if p is None:
            return None
        arr = np.asarray(_to_numpy(p), dtype=np.float32).ravel().copy()
        if arr.size >= 3 and self._tcp_offset:
            arr[:3] = arr[:3] + self._tcp_vector()
        return arr

    def _target_quat(self) -> np.ndarray:
        """The orientation to command: the setpoint, seeded once from the pose."""
        if self._quat_setpoint is None:
            self._quat_setpoint = self._current_quat()
        return self._quat_setpoint

    def grasp_orientation(self) -> np.ndarray:
        """A top-down approach orientation, as (qw, qx, qy, qz).

        The gripper's approach axis is its local +z. At reset these tasks start
        with the wrist pointing along world +x (a 90-degree rotation about y), so
        a gripper that is never reoriented cannot close on an object lying on the
        table however accurately it reaches. 180 degrees about x maps local +z to
        world -z, which is straight down.
        """
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    def _tcp_vector(self) -> np.ndarray:
        """Flange -> fingertip offset in the env frame, for the current wrist pose.

        The offset lives along the gripper's local +z; with the wrist hanging down
        that resolves to world -z, which is why a hardcoded -z looks correct until
        the agent rotates the wrist.
        """
        local = np.array([0.0, 0.0, self._tcp_offset], dtype=np.float32)
        return _rotate_by_quat(local, self._target_quat())

    def get_flange_pos(self):
        """The body the IK actually controls -- exposed for debugging."""
        p = (self._last_proprio or {}).get("ee_pos")
        return None if p is None else np.asarray(_to_numpy(p), dtype=np.float32).ravel()

    def check_subgoal(self, name: str) -> bool:
        # TODO(verify): RoboLab exposes composable success predicates. Map them
        # here once the env exposes predicate values in info or via env_cfg.
        return False

    def render(self) -> Optional[np.ndarray]:
        """Latest camera frame, cached from the last observation."""
        return self._last_image
