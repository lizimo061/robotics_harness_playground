"""3D task generators for the Franka / Genesis backend.

These reuse the same task KINDS as the 2D generators (pick_place, stack, ...)
but produce 3D positions on the Franka tabletop (z = table height) and set
dims=3. The GenesisFrankaEnv consumes these TaskSpecs.

Positions are in meters in the Franka base frame. The reachable workspace is
roughly x in [0.35, 0.7], y in [-0.25, 0.25], z in [0.05, 0.4].
"""
from __future__ import annotations

import numpy as np

from harness.tasks.base import TaskSpec, register_task_3d

_TABLE_Z = 0.02  # center height of a 0.04 m cube resting on the table


def _pos(x: float, y: float, z: float = _TABLE_Z) -> list:
    return [float(x), float(y), float(z)]


def _clip_xy(x: float, y: float) -> list:
    return [float(np.clip(x, 0.36, 0.68)), float(np.clip(y, -0.26, 0.26))]


@register_task_3d("pick_place")
def make_pick_place_3d(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    rng = np.random.default_rng(seed)
    obj = _pos(rng.uniform(0.42, 0.50), rng.uniform(-0.12, 0.12))
    ang = rng.uniform(0.0, 2.0 * np.pi)
    dist = 0.10 + 0.20 * difficulty
    gx, gy = _clip_xy(obj[0] + dist * np.cos(ang), obj[1] + dist * np.sin(ang))
    goal = _pos(gx, gy)
    return TaskSpec(
        kind="pick_place",
        dims=3,
        description="Pick up the cube and place it on the goal plate.",
        difficulty=difficulty,
        objects=[{"name": "cube", "pos": obj, "target": "goal"}],
        goals={"goal": goal},
        ee_start=[0.40, 0.0, 0.30],
        params={"goal_radius": 0.06, "grasp_radius": 0.13, "require_release": True},
        seed=seed,
    )


@register_task_3d("pick_place_obstacle")
def make_pick_place_obstacle_3d(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    base = make_pick_place_3d(seed=seed, difficulty=difficulty)
    obj = base.objects[0]["pos"]
    goal = base.goals["goal"]
    mid = [(obj[0] + goal[0]) / 2.0, (obj[1] + goal[1]) / 2.0]
    d = np.array(goal[:2]) - np.array(obj[:2])
    n = np.linalg.norm(d) + 1e-9
    perp = np.array([-d[1], d[0]]) / n
    offset = 0.04 + 0.04 * difficulty
    ox, oy = _clip_xy(mid[0] + perp[0] * offset, mid[1] + perp[1] * offset)
    base.kind = "pick_place_obstacle"
    base.description = "Pick up the cube and place it on the goal, avoiding the obstacle."
    base.obstacles = [{"name": "wall", "pos": _pos(ox, oy, 0.07), "radius": 0.05 + 0.05 * difficulty}]
    return base


@register_task_3d("push")
def make_push_3d(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    spec = make_pick_place_3d(seed=seed, difficulty=difficulty)
    spec.kind = "push"
    spec.description = "Push the cube to the goal plate (grasping is optional)."
    spec.params["require_release"] = False
    return spec


@register_task_3d("stack")
def make_stack_3d(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    rng = np.random.default_rng(seed)
    base = _pos(rng.uniform(0.46, 0.54), rng.uniform(-0.08, 0.08))
    ang = rng.uniform(0.0, 2.0 * np.pi)
    sep = 0.12 + 0.10 * difficulty
    tx, ty = _clip_xy(base[0] + sep * np.cos(ang), base[1] + sep * np.sin(ang))
    top = _pos(tx, ty)
    return TaskSpec(
        kind="stack",
        dims=3,
        description="Stack the top block on top of the base block.",
        difficulty=difficulty,
        objects=[
            {"name": "base", "pos": base, "role": "base"},
            {"name": "top", "pos": top, "role": "top"},
        ],
        goals={},
        ee_start=[0.40, 0.0, 0.30],
        params={"stack_radius": 0.07, "stack_height": 0.03, "grasp_radius": 0.13},
        seed=seed,
    )


@register_task_3d("sort")
def make_sort_3d(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    rng = np.random.default_rng(seed)
    n = 2 if difficulty < 0.6 else 3
    names = ["red", "green", "blue"][:n]
    goal_names = [f"bin_{c}" for c in names]
    objs = []
    used = []
    for name in names:
        x = rng.uniform(0.42, 0.52)
        y = rng.uniform(-0.15, 0.15)
        objs.append({"name": name, "pos": _pos(x, y), "target": goal_names[len(objs)]})
    goals = {}
    for i, gn in enumerate(goal_names):
        ang = i * (2.0 * np.pi / n) + rng.uniform(0.0, 0.5)
        gx, gy = _clip_xy(0.56 + 0.08 * np.cos(ang), 0.14 * np.sin(ang))
        goals[gn] = _pos(gx, gy)
    desc_pairs = ", ".join(f"{names[i]} -> {goal_names[i]}" for i in range(n))
    return TaskSpec(
        kind="sort",
        dims=3,
        description="Sort the objects into their matching bins: " + desc_pairs + ".",
        difficulty=difficulty,
        objects=objs,
        goals=goals,
        ee_start=[0.40, 0.0, 0.30],
        params={"goal_radius": 0.06, "grasp_radius": 0.13, "require_release": True},
        seed=seed,
    )


@register_task_3d("reach")
def make_reach_3d(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    rng = np.random.default_rng(seed)
    target = _pos(rng.uniform(0.5, 0.64), rng.uniform(-0.15, 0.15), 0.15)
    return TaskSpec(
        kind="reach",
        dims=3,
        description="Move the end-effector to the target point.",
        difficulty=difficulty,
        objects=[],
        goals={},
        ee_start=[0.40, 0.0, 0.30],
        ee_target=target,
        params={"target_radius": 0.05, "grasp_radius": 0.13},
        seed=seed,
    )


@register_task_3d("reach_avoid")
def make_reach_avoid_3d(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    base = make_reach_3d(seed=seed, difficulty=difficulty)
    tgt = base.ee_target
    obs = _pos(tgt[0] - 0.16, tgt[1], 0.07)
    base.kind = "reach_avoid"
    base.description = "Move the end-effector to the target while avoiding the obstacle."
    base.obstacles = [{"name": "wall", "pos": obs, "radius": 0.05 + 0.05 * difficulty}]
    return base
