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
        out = self._env.reset()
        obs, info = self._unwrap_reset(out)
        return self._to_obs(obs, info)

    def step(self, action: Action) -> StepResult:
        act = self._to_env_action(action)
        out = self._env.step(act)
        obs, reward, terminated, truncated, info = self._unwrap_step(out)
        info = dict(info or {})
        success = self._extract_success(info)
        info["success"] = success
        return StepResult(
            obs=self._to_obs(obs, info),
            reward=float(np.asarray(reward).ravel()[0]) if reward is not None else 0.0,
            terminated=bool(np.asarray(terminated).ravel()[0]) if terminated is not None else False,
            truncated=bool(np.asarray(truncated).ravel()[0]) if truncated is not None else False,
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
        o = _to_numpy(obs)
        return Obs(
            state=self._extract_state(o),
            image=self._extract_image(o),
            text=self._instruction,
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
        if not isinstance(obs, dict):
            return None
        for key in ("image", "rgb", "rgb_image"):
            if key in obs and obs[key] is not None:
                return _to_numpy(obs[key])
        policy = obs.get("policy")
        if isinstance(policy, dict):
            for key in ("image", "rgb"):
                if key in policy and policy[key] is not None:
                    return _to_numpy(policy[key])
        return None

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

    def _to_env_action(self, action: Action):
        sp = getattr(self._env, "action_space", None)
        dim = int(np.prod(sp.shape)) if sp is not None and hasattr(sp, "shape") else 0
        v = action.value
        if v is None:
            v = np.zeros(dim, dtype=np.float32)
        else:
            v = np.asarray(v, dtype=np.float32).ravel()
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
        # Isaac Lab expects a batch: (num_envs, action_dim)
        return v.reshape(self._num_envs, -1) if self._num_envs > 1 else v

    def _extract_success(self, info: dict) -> bool:
        for key in ("success", "is_success", "task_success", "goal_achieved"):
            if key in info:
                return bool(np.asarray(info[key]).ravel()[0])
        return False

    # -- text / subgoal --------------------------------------------------- #
    def get_text_state(self) -> str:
        return self._instruction

    def check_subgoal(self, name: str) -> bool:
        # TODO(verify): RoboLab exposes composable success predicates. Map them
        # here once the env exposes predicate values in info or via env_cfg.
        return False

    def render(self) -> Optional[np.ndarray]:
        # TODO(verify): RoboLab records its own videos (enable_cameras/save_videos);
        # a raw RGB grab may not be available on the gym path. Return None so the
        # harness text trace still works; episode videos come from RoboLab's recorder.
        return None
