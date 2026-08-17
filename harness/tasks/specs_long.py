"""Long-horizon task generators (multi-step, sequenced subgoals).

These tasks are a natural-language instruction plus an ordered list of subgoals.
The planner decomposes the instruction into skills; the executor runs them and
the env verifies each subgoal (check_subgoal).
"""
from __future__ import annotations

import numpy as np

from harness.tasks.base import TaskSpec, register_task


@register_task("cook_bread")
def make_cook_bread(seed: int = 0, difficulty: float = 0.5, **kw) -> TaskSpec:
    """Kitchen task: put the bread into the oven, then press the button to heat it."""
    rng = np.random.default_rng(seed)
    bread = [float(rng.uniform(0.22, 0.38)), float(rng.uniform(0.2, 0.5))]
    interior = [float(rng.uniform(0.7, 0.82)), float(rng.uniform(0.45, 0.65))]
    door = [interior[0] - 0.16, interior[1]]
    button = [float(rng.uniform(0.6, 0.85)), float(rng.uniform(0.12, 0.3))]
    return TaskSpec(
        kind="cook_bread",
        description="Put the bread into the oven, then press the button to heat the bread.",
        difficulty=difficulty,
        objects=[{"name": "bread", "pos": bread}],
        containers=[{"name": "oven", "interior": interior, "door": door, "open": False}],
        buttons=[{"name": "button", "pos": button, "pressed": False}],
        steps=[
            {"description": "Open the oven", "skill": "open", "args": {"container": "oven"}, "check": "oven_open"},
            {"description": "Put the bread in the oven", "skill": "put_in", "args": {"object": "bread", "container": "oven"}, "check": "bread_in_oven"},
            {"description": "Press the button", "skill": "press", "args": {"button": "button"}, "check": "button_pressed"},
        ],
        ee_start=[0.1, 0.1],
        params={"grasp_radius": 0.15, "actuate_radius": 0.12, "inside_radius": 0.12},
        seed=seed,
    )
