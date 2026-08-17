"""Task framework: typed task specifications and a generator registry.

A task is a *declarative* description of a scenario (object / goal / obstacle
layout plus success criteria), independent of the simulator backend. An
environment (TabletopEnv for 2D, GenesisFrankaEnv for 3D) consumes a TaskSpec
and implements the physics + success check for each kind.

Keeping tasks as data is what lets us *generate* harder tasks procedurally, and
reuse the same task across toy/tabletop/MuJoCo/Genesis backends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class TaskSpec:
    kind: str
    description: str  # natural-language instruction given to the LLM
    difficulty: float = 0.5  # 0.0 (easy) .. 1.0 (hard)
    dims: int = 2  # 2 (tabletop) or 3 (Franka / Genesis)
    objects: list[dict] = field(default_factory=list)  # [{"name","pos":[x,y(,z)],"target":goal|None,"role":...}]
    goals: dict = field(default_factory=dict)  # name -> [x, y(, z)]
    obstacles: list[dict] = field(default_factory=list)  # [{"name","pos":[x,y(,z)],"radius":r}]
    containers: list[dict] = field(default_factory=list)  # [{"name","interior":[x,y],"door":[x,y],"open":bool}]
    buttons: list[dict] = field(default_factory=list)  # [{"name","pos":[x,y],"pressed":bool}]
    steps: list[dict] = field(default_factory=list)  # ordered subgoals: {"description","skill","args","check"}
    ee_start: list = field(default_factory=lambda: [0.1, 0.1])
    ee_target: Optional[list] = None  # for reach / reach_avoid tasks
    params: dict[str, Any] = field(default_factory=dict)  # radii, limits, etc.
    seed: int = 0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "description": self.description,
            "difficulty": self.difficulty,
            "dims": self.dims,
            "objects": self.objects,
            "goals": self.goals,
            "obstacles": self.obstacles,
            "containers": self.containers,
            "buttons": self.buttons,
            "steps": self.steps,
            "ee_start": self.ee_start,
            "ee_target": self.ee_target,
            "params": self.params,
            "seed": self.seed,
        }


_TASK_GENERATORS_2D: dict[str, Callable[..., TaskSpec]] = {}
_TASK_GENERATORS_3D: dict[str, Callable[..., TaskSpec]] = {}


def register_task(kind: str):
    """Register a 2D (tabletop) task generator."""

    def deco(fn: Callable[..., TaskSpec]) -> Callable[..., TaskSpec]:
        _TASK_GENERATORS_2D[kind] = fn
        return fn

    return deco


def register_task_3d(kind: str):
    """Register a 3D (Franka / Genesis) task generator."""

    def deco(fn: Callable[..., TaskSpec]) -> Callable[..., TaskSpec]:
        _TASK_GENERATORS_3D[kind] = fn
        return fn

    return deco


def generate_task(name: str, seed: int = 0, difficulty: float = 0.5, dims: int = 2, **kw) -> TaskSpec:
    reg = _TASK_GENERATORS_3D if dims == 3 else _TASK_GENERATORS_2D
    if name not in reg:
        raise KeyError(f"Unknown {dims}d task '{name}'. Available: {sorted(reg)}")
    return reg[name](seed=seed, difficulty=difficulty, **kw)


def available_tasks(dims: int = 2) -> list[str]:
    reg = _TASK_GENERATORS_3D if dims == 3 else _TASK_GENERATORS_2D
    return sorted(reg)


def generate_curriculum(kind: str, seeds: list[int], difficulties: list[float], dims: int = 2) -> list[TaskSpec]:
    """Return a list of TaskSpecs of increasing difficulty (one per seed/level)."""
    return [generate_task(kind, seed=s, difficulty=d, dims=dims) for s, d in zip(seeds, difficulties)]
