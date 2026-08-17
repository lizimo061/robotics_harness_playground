"""Adapter that wraps any gymnasium.Env.

This covers MuJoCo manipulation via gymnasium-robotics (FetchPickAndPlace,
FetchReach, Panda/UR5e tasks, ...) as well as classic control and many other
environments, all through one interface.

Requires: pip install gymnasium  (and gymnasium-robotics for MuJoCo tasks)
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from harness.envs.base import Env
from harness.types import Obs, Action, StepResult, ActionSpace, ObservationSpace


class GymnasiumEnv(Env):
    name = "gymnasium"

    def __init__(self, env_id: str, *, max_episode_steps: int = 200, seed: int = 0, **kwargs: Any) -> None:
        try:
            import gymnasium as gym  # type: ignore
        except ImportError as e:
            raise ImportError(
                "gymnasium is not installed. Run: pip install gymnasium gymnasium-robotics mujoco"
            ) from e
        self._gym = gym
        self._env = gym.make(env_id, max_episode_steps=max_episode_steps, **kwargs)
        self._env_id = env_id
        self._seed = seed

    # -- spaces ----------------------------------------------------------- #
    @property
    def observation_space(self) -> ObservationSpace:
        sp = self._env.observation_space
        if isinstance(getattr(sp, "spaces", None), dict):
            inner = sp.spaces
            state = inner.get("observation")
            image = inner.get("image") or inner.get("rgb") or inner.get("pixels")
            state_dim = int(np.asarray(state).size) if state is not None else 0
            image_shape = tuple(getattr(image, "shape", (0, 0, 0)))[:3] if image is not None else (0, 0, 0)
            return ObservationSpace(
                state_dim=state_dim,
                image_shape=image_shape,
                has_image=image is not None,
                description=f"{self._env_id} dict observation",
            )
        if hasattr(sp, "shape"):
            return ObservationSpace(state_dim=int(np.prod(sp.shape)), description=str(sp))
        return ObservationSpace(description=str(sp))

    @property
    def action_space(self) -> ActionSpace:
        sp = self._env.action_space
        if hasattr(sp, "n"):
            return ActionSpace(kind="discrete", dim=1, description=f"Discrete({sp.n})")
        low = np.asarray(getattr(sp, "low", np.array([], dtype=np.float32)), dtype=np.float32)
        high = np.asarray(getattr(sp, "high", np.array([], dtype=np.float32)), dtype=np.float32)
        dim = int(np.prod(sp.shape)) if hasattr(sp, "shape") else int(low.size)
        return ActionSpace(
            kind="joint_position", dim=dim, low=low, high=high, description=str(sp)
        )

    # -- lifecycle -------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None) -> Obs:
        obs, info = self._env.reset(seed=seed if seed is not None else self._seed)
        return self._to_obs(obs, info)

    def step(self, action: Action) -> StepResult:
        gym_act = self._to_gym_action(action)
        obs, reward, terminated, truncated, info = self._env.step(gym_act)
        info = dict(info or {})
        success = bool(info.get("success") or info.get("is_success") or (terminated and info.get("goal_achieved", False)))
        info["success"] = success
        return StepResult(
            obs=self._to_obs(obs, info),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    # -- mapping ---------------------------------------------------------- #
    def _to_obs(self, obs, info) -> Obs:
        if isinstance(obs, dict):
            state = obs.get("observation")
            image = obs.get("image") or obs.get("rgb") or obs.get("pixels")
            state_arr = np.asarray(state, dtype=np.float32).ravel() if state is not None else None
            return Obs(state=state_arr, image=image, info=dict(info or {}))
        if isinstance(obs, (np.ndarray, list, tuple)):
            return Obs(state=np.asarray(obs, dtype=np.float32).ravel(), info=dict(info or {}))
        return Obs(text=str(obs), info=dict(info or {}))

    def _to_gym_action(self, action: Action):
        sp = self._env.action_space
        if hasattr(sp, "n"):
            if action.value is not None:
                return int(np.asarray(action.value).ravel()[0])
            return 0
        if action.value is not None:
            return np.asarray(action.value, dtype=sp.dtype).reshape(sp.shape)
        return np.zeros(sp.shape, dtype=sp.dtype)

    def render(self) -> Optional[np.ndarray]:
        try:
            frame = self._env.render()
        except Exception:
            return None
        if frame is None:
            return None
        return np.asarray(frame)

    def close(self) -> None:
        self._env.close()
