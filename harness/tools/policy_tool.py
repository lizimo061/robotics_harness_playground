"""Expose a low-level policy as a tool the LLM can call.

This is the hierarchical control seam: the LLM plans and verifies, a trained VLA
executes. In tools mode the LLM issues

    {"tool": "run_policy", "args": {"instruction": "pick up the banana", "steps": 60}}

and this tool runs the policy closed-loop against the environment for up to
``steps`` steps, stopping early on success or termination, then reports what
happened so the LLM can decide what to do next.

Unlike the single-action tools in builtin.py, this one drives the env itself
(``closed_loop = True``) because a policy needs many steps per sub-instruction.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from harness.tools.base import Tool, ToolResult
from harness.types import Action
from harness.utils.logging import get_logger

log = get_logger("harness.tools.policy_tool")


class RunPolicyTool(Tool):
    name = "run_policy"
    description = (
        "Delegate low-level motor control to the trained manipulation policy. "
        "Give ONE short, concrete sub-instruction in plain English (e.g. 'pick up "
        "the banana', 'place the banana in the bowl') and the policy will attempt "
        "it for a bounded number of steps. Returns how many steps ran, whether the "
        "task succeeded, and the resulting scene state. Use this for all physical "
        "motion; use the perception tools to check the result."
    )
    parameters = {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "one short sub-instruction for the policy to execute",
            },
            "steps": {
                "type": "integer",
                "description": "max environment steps to run (optional; bounded by the tool)",
            },
        },
        "required": ["instruction"],
    }
    closed_loop = True

    def __init__(
        self,
        policy,
        *,
        default_steps: int = 50,
        max_steps: int = 400,
        use_vision: bool = False,
        stop_on_success: bool = True,
    ) -> None:
        self._policy = policy
        self._default_steps = default_steps
        self._max_steps = max_steps
        self._use_vision = use_vision
        self._stop_on_success = stop_on_success

    # -- tool ------------------------------------------------------------- #
    def run(
        self,
        env,
        instruction: str = "",
        steps: Optional[int] = None,
        on_step: Optional[Callable[[Action, Any], None]] = None,
        **kw: Any,
    ) -> ToolResult:
        instruction = str(instruction or "").strip()
        if not instruction:
            return ToolResult(feedback="run_policy needs a non-empty 'instruction' argument.")

        budget = self._resolve_budget(steps)
        kind = getattr(env.action_space, "kind", "") or "joint_position"

        self._policy.begin(instruction, action_space=env.action_space)

        ran = 0
        success = False
        stopped = "budget exhausted"

        for _ in range(budget):
            obs_text = env.get_text_state() or ""
            # Refresh the frame every step. Rendering once before the loop fed the
            # policy the same initial image for the whole rollout, which for a
            # visuomotor policy means acting blind on a stale observation -- the
            # scene it is closing the loop on stopped existing after step 1.
            image = env.render() if self._use_vision else None
            try:
                vec = self._policy.act(obs_text, image=image)
            except Exception as e:  # noqa: BLE001 - surface policy failures to the LLM
                log.warning("policy.act failed: %s", e)
                return ToolResult(
                    feedback=f"policy error after {ran} steps: {e}",
                    steps=ran,
                    success=False,
                )

            action = Action(
                kind=kind,
                value=np.asarray(vec, dtype=np.float32).ravel(),
                comment=f"run_policy: {instruction}",
            )
            result = env.step(action)
            ran += 1
            if on_step is not None:
                on_step(action, result)

            image = result.obs.image if self._use_vision else None

            if result.success:
                success, stopped = True, "success"
                if self._stop_on_success:
                    break
            if result.terminated:
                stopped = "episode terminated"
                break
            if result.truncated:
                stopped = "step budget of the environment reached"
                break

        return ToolResult(
            feedback=self._summarize(instruction, ran, success, stopped, env),
            steps=ran,
            success=success,
        )

    # -- helpers ---------------------------------------------------------- #
    def _resolve_budget(self, steps: Optional[int]) -> int:
        if steps is None:
            return self._default_steps
        try:
            n = int(steps)
        except (TypeError, ValueError):
            return self._default_steps
        return max(1, min(n, self._max_steps))

    def _summarize(self, instruction: str, ran: int, success: bool, stopped: str, env) -> str:
        verdict = "TASK SUCCESS" if success else "not yet successful"
        parts = [
            f"Ran the policy on '{instruction}' for {ran} step(s); stopped: {stopped}. {verdict}.",
        ]
        state = env.get_text_state()
        if state:
            parts.append("Resulting state:")
            parts.append(state)
        return "\n".join(parts)
