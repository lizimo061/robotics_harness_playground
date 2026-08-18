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
        action_mode: str = "ee_delta",  # "ee_delta" | "joint_position"
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        try:
            from robolab.core.environments.factory import get_envs  # noqa: F401
            from robolab.core.environments.runtime import create_env, end_episode  # noqa: F401
            from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "RoboLab (Isaac Lab / Isaac Sim) is not installed. "
                "Install it in its own venv on Linux + CUDA; see docs/modules/robolab.md."
            ) from e

        self._create_env = create_env
        self._end_episode = end_episode
        self._auto_register = auto_register_droid_envs
        self._auto_register()  # populate the env factory

        # resolve the task (exact name, or a tag)
        resolved = get_envs(task=[task]) if task else get_envs()
        if not resolved:
            raise ValueError(f"RoboLab task '{task}' not found in the factory")
        task_name = resolved[0]

        self._env, self._env_cfg = self._create_env(
            task_name, device=device, num_envs=num_envs, use_fabric=use_fabric
        )
        self.task = task_name
        self._instruction = str(getattr(self._env_cfg, "instruction", task_name) or task_name)
        self._action_mode = action_mode
        self._seed = seed
        self._use_fabric = use_fabric
        self._num_envs = num_envs
        self._last_image = None
        self._last_proprio: dict = {}
        self._step_idx = 0

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
        image-shaped tensor instead, preferring a viewport/exterior view.
        """
        best = None
        best_rank = -1

        def visit(node, path=""):
            nonlocal best, best_rank
            if node is None:
                return
            if hasattr(node, "shape"):
                shape = tuple(node.shape)
                # (..., H, W, C) with C in {3, 4} and a plausible frame size
                if len(shape) >= 3 and shape[-1] in (3, 4) and shape[-2] >= 32 and shape[-3] >= 32:
                    low = path.lower()
                    rank = 2 if "viewport" in low else (1 if "wrist" not in low else 0)
                    if rank > best_rank:
                        best, best_rank = node, rank
                return
            keys = getattr(node, "keys", None)
            if keys is None:
                return
            for k in list(keys()):
                visit(node[k], f"{path}/{k}")

        visit(obs)
        if best is None:
            return None

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

    def _to_env_action(self, action: Action):
        sp = getattr(self._env, "action_space", None)
        dim = int(np.prod(sp.shape)) if sp is not None and hasattr(sp, "shape") else 0
        v = action.value
        if v is None:
            v = np.zeros(dim, dtype=np.float32)
        else:
            v = np.asarray(v, dtype=np.float32).ravel()

        # Respect the action *kind*. RoboLab's controller consumes relative
        # end-effector deltas, so passing an absolute target through unchanged
        # makes `move_to(0.43, -0.10, 0.03)` a command to jump 43cm -- the agent
        # can then never deliberately reach anything, and the resulting failures
        # look like the model's when they are ours.
        if action.kind in ("ee_pose", "pose", "absolute") and v.size >= 2:
            current = self.get_ee_pos()
            if current is None:
                log.warning("absolute move requested but the end-effector pose is "
                            "unknown; treating it as a delta")
            else:
                cur = np.asarray(current, dtype=np.float32).ravel()
                target = v[:3] if v.size >= 3 else np.concatenate([v[:2], cur[2:3]])
                n = min(len(target), len(cur))
                delta = np.zeros(3, dtype=np.float32)
                delta[:n] = target[:n] - cur[:n]
                v = self._clip_delta(delta, dim)
        if action.gripper is not None:
            v = v.copy()
            if v.size >= 1:
                v[-1] = float(action.gripper)  # gripper is the last action dim
            else:
                v = np.array([float(action.gripper)], dtype=np.float32)
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
        if self._step_idx:
            lines.append(f"Step {self._step_idx}.")
        return "\n".join(lines)

    def get_ee_pos(self):
        p = (self._last_proprio or {}).get("ee_pos")
        return None if p is None else np.asarray(p, dtype=np.float32)

    def check_subgoal(self, name: str) -> bool:
        # TODO(verify): RoboLab exposes composable success predicates. Map them
        # here once the env exposes predicate values in info or via env_cfg.
        return False

    def render(self) -> Optional[np.ndarray]:
        """Latest camera frame, cached from the last observation."""
        return self._last_image
