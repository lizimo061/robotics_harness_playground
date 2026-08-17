"""Procedural generators for harder robot-arm tasks (2D tabletop abstraction).

Each generator returns a TaskSpec with a randomized but deterministic layout
(seeded). Difficulty scales distances, obstacle sizes, and the number of
objects/goals. The same TaskSpec concept transfers to a 3D Franka backend by
using 3D positions in the generator (see docs/guides/add-task.md).
"""
from __future__ import annotations

import numpy as np

from harness.tasks.base import TaskSpec, register_task


def _clip(p, lo: float = 0.08, hi: float = 0.92) -> list:
    return [float(np.clip(p[0], lo, hi)), float(np.clip(p[1], lo, hi))]


def _far(others: list, min_dist: float, rng) -> list:
    """Sample a point at least min_dist from every point in others."""
    for _ in range(200):
        cand = [float(rng.uniform(0.12, 0.88)), float(rng.uniform(0.12, 0.88))]
        if all(np.linalg.norm(np.array(cand) - np.array(o)) >= min_dist for o in others):
            return cand
    return [float(rng.uniform(0.12, 0.88)), float(rng.uniform(0.12, 0.88))]


@register_task("pick_place")
def make_pick_place(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    rng = np.random.default_rng(seed)
    obj = _clip(rng.uniform(0.2, 0.5, 2))
    ang = rng.uniform(0.0, 2.0 * np.pi)
    dist = 0.25 + 0.35 * difficulty
    goal = _clip([obj[0] + dist * np.cos(ang), obj[1] + dist * np.sin(ang)])
    return TaskSpec(
        kind="pick_place",
        description="Pick up the cube and place it on the goal plate.",
        difficulty=difficulty,
        objects=[{"name": "cube", "pos": obj, "target": "goal"}],
        goals={"goal": goal},
        ee_start=[0.1, 0.1],
        params={"goal_radius": 0.08, "grasp_radius": 0.12, "require_release": True},
        seed=seed,
    )


@register_task("pick_place_obstacle")
def make_pick_place_obstacle(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    base = make_pick_place(seed=seed, difficulty=difficulty)
    rng = np.random.default_rng(seed + 1000)
    obj = base.objects[0]["pos"]
    goal = base.goals["goal"]
    mid = [(obj[0] + goal[0]) / 2.0, (obj[1] + goal[1]) / 2.0]
    d = np.array(goal) - np.array(obj)
    n = np.linalg.norm(d) + 1e-9
    perp = np.array([-d[1], d[0]]) / n
    offset = 0.06 + 0.08 * difficulty
    obs = _clip([mid[0] + perp[0] * offset, mid[1] + perp[1] * offset], 0.1, 0.9)
    radius = 0.06 + 0.10 * difficulty
    base.kind = "pick_place_obstacle"
    base.description = "Pick up the cube and place it on the goal, avoiding the obstacle."
    base.obstacles = [{"name": "wall", "pos": obs, "radius": radius}]
    return base


@register_task("push")
def make_push(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    spec = make_pick_place(seed=seed, difficulty=difficulty)
    spec.kind = "push"
    spec.description = "Push the cube to the goal plate (grasping is optional)."
    spec.params["require_release"] = False
    return spec


@register_task("stack")
def make_stack(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    rng = np.random.default_rng(seed)
    base = _clip(rng.uniform(0.3, 0.6, 2))
    ang = rng.uniform(0.0, 2.0 * np.pi)
    sep = 0.25 + 0.2 * difficulty
    top = _clip([base[0] + sep * np.cos(ang), base[1] + sep * np.sin(ang)])
    return TaskSpec(
        kind="stack",
        description="Stack the top block on top of the base block.",
        difficulty=difficulty,
        objects=[
            {"name": "base", "pos": base, "role": "base"},
            {"name": "top", "pos": top, "role": "top"},
        ],
        goals={},
        ee_start=[0.1, 0.1],
        params={"stack_radius": 0.10, "grasp_radius": 0.12},
        seed=seed,
    )


@register_task("sort")
def make_sort(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    rng = np.random.default_rng(seed)
    n = 2 if difficulty < 0.6 else 3
    names = ["red", "green", "blue"][:n]
    goal_names = [f"bin_{c}" for c in names]
    objs = []
    used = []
    for i, name in enumerate(names):
        p = _far(used, 0.22, rng)
        used.append(p)
        objs.append({"name": name, "pos": p, "target": goal_names[i]})
    goals = {}
    used_g = list(used)
    for gn in goal_names:
        g = _far(used_g, 0.2, rng)
        used_g.append(g)
        goals[gn] = g
    desc_pairs = ", ".join(f"{names[i]} -> {goal_names[i]}" for i in range(n))
    return TaskSpec(
        kind="sort",
        description="Sort the objects into their matching bins: " + desc_pairs + ".",
        difficulty=difficulty,
        objects=objs,
        goals=goals,
        ee_start=[0.1, 0.1],
        params={"goal_radius": 0.08, "grasp_radius": 0.12, "require_release": True},
        seed=seed,
    )


@register_task("reach_avoid")
def make_reach_avoid(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    rng = np.random.default_rng(seed)
    target = _clip(rng.uniform(0.6, 0.9, 2))
    obs = _clip([target[0] - 0.25, target[1] - 0.05 + rng.uniform(-0.1, 0.1)], 0.15, 0.85)
    radius = 0.08 + 0.08 * difficulty
    return TaskSpec(
        kind="reach_avoid",
        description="Move the end-effector to the target while avoiding the obstacle.",
        difficulty=difficulty,
        objects=[],
        goals={},
        obstacles=[{"name": "wall", "pos": obs, "radius": radius}],
        ee_start=[0.1, 0.1],
        ee_target=target,
        params={"target_radius": 0.07, "grasp_radius": 0.12},
        seed=seed,
    )
