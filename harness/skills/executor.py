"""Run a skill closed-loop until its subgoal is satisfied (or it fails)."""
from __future__ import annotations

from typing import Callable, Optional

from harness.skills.base import Skill, SkillResult
from harness.types import Action, StepResult


def run_skill(
    env,
    skill: Skill,
    *,
    budget: int = 40,
    on_step: Optional[Callable[[Action, StepResult], None]] = None,
) -> SkillResult:
    skill.reset()
    for i in range(budget):
        if skill.satisfied(env):
            return SkillResult(success=True, feedback=f"subgoal reached in {i} steps", steps=i)
        action = skill.plan_action(env)
        if action is None:
            return SkillResult(success=False, feedback="skill produced no action", steps=i)
        result = env.step(action)
        if on_step is not None:
            on_step(action, result)
        if result.info.get("collided"):
            return SkillResult(success=False, feedback="collision", steps=i + 1)
        if result.terminated or result.truncated:
            return SkillResult(success=result.info.get("success", False), feedback="episode ended", steps=i + 1)
    return SkillResult(success=skill.satisfied(env), feedback="skill budget exhausted", steps=budget)
