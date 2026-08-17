"""robosuite adapter (modular MuJoCo manipulation).

Requires: pip install robosuite torch

robosuite provides standardized environments, controllers (OSC_POSE,
JOINT_POSITION, ...) and robots (Panda, Sawyer, UR5e, ...). This adapter maps
its low-level API onto the harness interface.

Note: action dimensionality depends on the chosen controller; the harness
treats the robosuite action vector opaquely (the LLM is told the bounds, not
the semantics), which is fine for policies that learn/emit in action space.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from harness.envs.base import Env
from harness.types import Obs, Action, StepResult, ActionSpace, ObservationSpace


class RobosuiteEnv(Env):
    name = "robosuite"

    def __init__(
        self,
        *,
        task: str = "Lift",
        seed: int = 0,
        robot: str = "Panda",
        controller: str = "OSC_POSE",
        horizon: int = 200,
        **kwargs: Any,
    ) -> None:
        import robosuite as suite  # type: ignore

        self._suite = suite
        self._env = suite.make(
            env_name=task,
            robots=robot,
            controller_configs=suite.load_controller_config(default_controller=controller),
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=horizon,
            control_freq=20,
            **kwargs,
        )
        self.task = task
        self._seed = seed

    # -- spaces ----------------------------------------------------------- #
    @property
    def observation_space(self) -> ObservationSpace:
        dim = 0
        if self._env.observation_spec():
            first = self._env.observation_spec()[0]
            dim = int(np.asarray(first).size)
        return ObservationSpace(state_dim=dim, description="robosuite low-dim observation")

    @property
    def action_space(self) -> ActionSpace:
        spec = self._env.action_spec
        if spec:
            low, high = spec[0]
            low = np.asarray(low, dtype=np.float32)
            high = np.asarray(high, dtype=np.float32)
            return ActionSpace(
                kind="joint_position",
                dim=int(low.size),
                low=low,
                high=high,
                description="robosuite action (controller-dependent)",
            )
        return ActionSpace()

    # -- lifecycle -------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None) -> Obs:
        obs = self._env.reset()
        return self._to_obs(obs)

    def step(self, action: Action) -> StepResult:
        if action.value is None:
            act = np.zeros(self.action_space.dim, dtype=np.float32)
        else:
            act = np.asarray(action.value, dtype=np.float32)
        obs, reward, done, info = self._env.step(act)
        info = dict(info or {})
        success = bool(info.get("success", False))
        return StepResult(
            obs=self._to_obs(obs),
            reward=float(reward),
            terminated=bool(done),
            truncated=False,
            info={"success": success},
        )

    def _to_obs(self, obs) -> Obs:
        if isinstance(obs, dict):
            state = obs.get("robot0_proprio-state")
            state_arr = np.asarray(state, dtype=np.float32).ravel() if state is not None else None
            return Obs(state=state_arr)
        return Obs(state=np.asarray(obs, dtype=np.float32).ravel())

    def close(self) -> None:
        self._env.close()
