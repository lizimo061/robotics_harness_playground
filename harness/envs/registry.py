"""Environment factory: turn an EnvConfig into a concrete Env."""
from __future__ import annotations

from harness.config import EnvConfig
from harness.envs.base import Env


def get_env(cfg: EnvConfig) -> Env:
    name = cfg.name

    if name in ("toy", "toy_tabletop", "toy_pick_and_place"):
        from harness.envs.toy import ToyTabletopEnv

        return ToyTabletopEnv(
            task=cfg.task,
            seed=cfg.seed,
            max_episode_steps=cfg.max_episode_steps,
            render=cfg.render,
            **cfg.params,
        )

    if name in ("tabletop", "tabletop_tasks", "tasks"):
        from harness.envs.tabletop import TabletopEnv

        return TabletopEnv(
            task=cfg.task,
            seed=cfg.seed,
            max_episode_steps=cfg.max_episode_steps,
            **cfg.params,
        )

    if name.startswith("gymnasium:") or name.startswith("gym:"):
        env_id = name.split(":", 1)[1]
        from harness.envs.gymnasium import GymnasiumEnv

        return GymnasiumEnv(env_id, max_episode_steps=cfg.max_episode_steps, seed=cfg.seed, **cfg.params)

    if name in ("kitchen", "kitchen_tasks"):
        from harness.envs.kitchen import KitchenEnv

        return KitchenEnv(
            task=cfg.task,
            seed=cfg.seed,
            max_episode_steps=cfg.max_episode_steps,
            **cfg.params,
        )

    if name.startswith("robolab:"):
        task = name.split(":", 1)[1] or cfg.task
        from harness.envs.robolab import RoboLabEnv

        return RoboLabEnv(task=task, seed=cfg.seed, **cfg.params)

    if name.startswith("genesis:"):
        task = name.split(":", 1)[1] or cfg.task
        from harness.envs.genesis import GenesisFrankaEnv

        return GenesisFrankaEnv(task=task, seed=cfg.seed, max_episode_steps=cfg.max_episode_steps, **cfg.params)

    if name.startswith("robosuite:"):
        task = name.split(":", 1)[1] or cfg.task
        from harness.envs.robosuite import RobosuiteEnv

        return RobosuiteEnv(task=task, seed=cfg.seed, **cfg.params)

    raise KeyError(
        f"Unknown environment '{name}'. Use toy_tabletop, gymnasium:<id>, genesis:<task>, or robosuite:<task>."
    )
