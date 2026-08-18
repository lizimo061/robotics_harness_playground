"""Expose a low-level policy as a tool the LLM can call.

This is the hierarchical control seam: the LLM plans and verifies, a trained VLA
executes. In tools mode the LLM issues

    {"tool": "run_policy", "args": {"instruction": "pick up the banana", "steps": 60}}

and this tool runs the policy closed-loop against the environment for up to
``steps`` steps, stopping early on success or termination, then reports what
happened so the LLM can decide what to do next.

Unlike the single-action tools in builtin.py, this one drives the env itself
(``closed_loop = True``) because a policy needs many steps per sub-instruction.

**Interruptibility.** VoLoAgent's central claim is that the physical world does not
pause for reasoning, so a VLA must be one *interruptible* capability rather than a
fixed executor: the agent monitors mid-rollout and chooses to continue, advance to
the next subgoal, or recover. With ``monitor_every`` set, this tool returns after
that many steps with a **status** instead of a verdict, leaving the rollout open;
``continue_policy`` resumes it and ``abort_policy`` ends it.

That is *chunked*, not threaded, and the distinction is deliberate. VoLo's asynchrony
is a property of real hardware -- the arm keeps moving while the VLM thinks. IsaacLab
envs are stepped from one thread and are not safe to drive concurrently, so the
faithful analogue in simulation is: run n steps, pause the world, monitor, resume.
Episodes stay reproducible, which the evaluation depends on. The interface is shaped
so a real-robot backend can be genuinely asynchronous later; the simulated version
must never be described as measuring reaction time under motion.
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
        #: steps between monitor breakpoints; 0 disables interruption entirely and
        #: restores the original run-to-completion behaviour
        monitor_every: int = 0,
        stop_on_success: bool = True,
    ) -> None:
        self._policy = policy
        self._default_steps = default_steps
        self._max_steps = max_steps
        self._use_vision = use_vision
        self._monitor_every = max(0, int(monitor_every))
        #: an open rollout, when monitoring is on
        self._active: Optional[dict] = None
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
        self._policy.begin(instruction, action_space=env.action_space)
        self._active = {"instruction": instruction, "remaining": budget, "ran": 0}
        return self._rollout(env, on_step=on_step)

    # -- the rollout, resumable -------------------------------------------- #
    def _rollout(self, env, *, on_step=None) -> ToolResult:
        """Run up to the next monitor breakpoint, or to the end if monitoring is off."""
        state = self._active
        if state is None:
            return ToolResult(feedback="No policy rollout is running. Call run_policy first.")
        instruction = state["instruction"]
        kind = getattr(env.action_space, "kind", "") or "joint_position"

        chunk = state["remaining"]
        if self._monitor_every:
            chunk = min(chunk, self._monitor_every)

        ran = 0
        success = False
        stopped = "budget exhausted"
        closed = True

        for _ in range(chunk):
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
                self._active = None
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

        state["ran"] += ran
        state["remaining"] -= ran
        # A rollout stays OPEN when it merely hit a monitor breakpoint: steps remain,
        # nothing terminal happened. success=None is the signal for "still running" --
        # reporting False here would tell the agent the sub-instruction failed when it
        # has not finished being attempted.
        if (self._monitor_every and state["remaining"] > 0 and not success
                and stopped == "budget exhausted"):
            closed = False
            stopped = f"monitor breakpoint after {ran} step(s)"
        if closed:
            total = state["ran"]
            self._active = None
            return ToolResult(
                feedback=self._summarize(instruction, total, success, stopped, env),
                steps=ran,
                success=success,
            )
        return ToolResult(
            feedback=self._status(instruction, state, stopped, env),
            steps=ran,
            success=None,
        )

    def is_running(self) -> bool:
        return self._active is not None

    def resume(self, env, *, on_step=None) -> ToolResult:
        return self._rollout(env, on_step=on_step)

    def abort(self, env) -> ToolResult:
        state, self._active = self._active, None
        if state is None:
            return ToolResult(feedback="No policy rollout was running.")
        return ToolResult(
            feedback=(f"Aborted '{state['instruction']}' after {state['ran']} step(s); "
                      f"the arm holds its current pose.\n" + (env.get_text_state() or "")),
            steps=0,
            success=False,
        )

    def _status(self, instruction: str, state: dict, stopped: str, env) -> str:
        return "\n".join([
            f"Policy '{instruction}' is STILL RUNNING: {state['ran']} step(s) done, "
            f"{state['remaining']} of its budget left ({stopped}).",
            "Choose one: continue_policy to keep going, abort_policy to stop it and "
            "take another action.",
            "Current state:",
            env.get_text_state() or "",
        ])

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


class ContinuePolicyTool(Tool):
    """Resume an open rollout. One of VoLoAgent's three monitor decisions.

    Shares the RunPolicyTool instance so the rollout state lives in one place; a
    separate copy would let the agent "continue" a rollout that no longer exists.
    """

    name = "continue_policy"
    description = ("Let the running policy continue for another stretch. Use when the "
                   "monitor observation shows it is making progress.")
    parameters = {"type": "object", "properties": {}}
    closed_loop = True

    def __init__(self, runner: RunPolicyTool) -> None:
        self._runner = runner

    def run(self, env, on_step: Optional[Callable] = None, **kw: Any) -> ToolResult:
        if not self._runner.is_running():
            return ToolResult(feedback="No policy is running. Call run_policy first.")
        return self._runner.resume(env, on_step=on_step)


class AbortPolicyTool(Tool):
    """Stop an open rollout and hold position.

    This is the preemption half of interruptibility, and the "hold position" part is
    load-bearing: VoLoAgent idles the robot while reasoning continues, rather than
    leaving it running into whatever it was getting wrong.
    """

    name = "abort_policy"
    description = ("Stop the running policy now and hold the arm still. Use when the "
                   "monitor observation shows it is going wrong, then choose a "
                   "different sub-instruction or tool.")
    parameters = {"type": "object", "properties": {
        "reason": {"type": "string", "description": "why you are stopping it"}}}

    def __init__(self, runner: RunPolicyTool) -> None:
        self._runner = runner

    def run(self, env, reason: str = "", **kw: Any) -> ToolResult:
        result = self._runner.abort(env)
        if reason:
            result.feedback = f"{result.feedback}\nReason given: {reason}"
        return result
