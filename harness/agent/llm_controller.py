"""The LLM controller: connects an LLM to an environment via a ReAct loop.

Modes:
  json  -- the default. Each step, send the observation and ask for one JSON
           action; parse, step, repeat (closed-loop).
  code  -- Code-as-Policies. The LLM writes a Python snippet that calls a skill
           library to drive the environment.
  plan  -- First ask for a plan, then fall back to the json loop with the plan
           in context.
  tools -- Tool use. The LLM calls named tools (grasp, move_to, list_objects,
           ...) one at a time; the harness executes them and feeds back results.

Visualization: pass a TraceRecorder and/or an on_step callback to capture the
per-step observation, prompt, response, action, reward and rendered frame.
"""
from __future__ import annotations

from typing import Callable, Optional

from harness.agent.action_parser import parse_action
from harness.agent.code_executor import execute_code
from harness.agent.prompts import (
    build_observation_message,
    build_system_prompt,
    build_tools_system_prompt,
)
from harness.agent.skills import SkillContext, get_skill_docs
from harness.envs.base import Env
from harness.llm.base import ChatMessage, LLMClient
from harness.llm.capabilities import TokenUsage, estimate_cost
from harness.types import Action, Episode
from harness.utils.logging import get_logger
from harness.viz.recorder import TraceRecorder, TraceStep, action_to_dict

log = get_logger("harness.agent.llm_controller")

_FENCE = chr(96) * 3


def _extract_code(text: str) -> str:
    t = text.strip()
    if t.startswith(_FENCE):
        lines = t.splitlines()
        if lines and lines[0].strip().startswith(_FENCE):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith(_FENCE):
            lines = lines[:-1]
        t = chr(10).join(lines).strip()
    return t


def _serialize_messages(messages: list[ChatMessage]) -> list[dict]:
    out = []
    for m in messages:
        if isinstance(m.content, str):
            preview = m.content
        else:
            parts = []
            for blk in m.content:
                t = blk.get("type")
                if t == "text":
                    parts.append(blk.get("text", ""))
                elif t == "image_url":
                    parts.append("[image]")
            preview = " ".join(parts)
        if len(preview) > 500:
            preview = preview[:500] + "..."
        out.append({"role": m.role, "content": preview})
    return out


class LLMController:
    name = "llm_controller"

    def __init__(
        self,
        llm: LLMClient,
        *,
        mode: str = "json",
        max_steps: int = 50,
        use_vision: bool = False,
        #: Read a bounded monitor prompt each turn instead of replaying the whole
        #: transcript, consulting the full history only at planning points. Off by
        #: default so existing runs are byte-identical; see harness/agent/context.py.
        two_clock: bool = False,
        monitor_decisions: int = 4,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        task_description: str = "",
        recorder: Optional[TraceRecorder] = None,
        on_step: Optional[Callable[[TraceStep], None]] = None,
        tools: Optional[list] = None,
        policy=None,
        policy_options: Optional[dict] = None,
        **kwargs,
    ) -> None:
        self._llm = llm
        self._mode = mode
        self._max_steps = max_steps
        self._use_vision = use_vision
        self._warned_no_image = False
        self._two_clock = bool(two_clock)
        self._monitor_decisions = int(monitor_decisions)
        self._transcript = None
        self._system_prompt_override = system_prompt
        self._temperature = temperature
        self._task_description = task_description
        self._recorder = recorder
        self._on_step = on_step
        # use_vision belongs here too: without it no frame is rendered, so asking
        # for vision produced a text-only prompt with nothing to attach.
        self._capture_frames = ((recorder is not None and recorder.capture_frames)
                               or on_step is not None or use_vision)

        # token / cost accounting, reset per episode in run()
        self._usage = TokenUsage()
        self._cost_usd = 0.0
        self._cost_known = True
        self._llm_calls = 0

        self._policy = None
        if policy is not None:
            from harness.policies import get_policy

            self._policy = get_policy(policy, llm=llm)

        self._tools = None
        self._tool_registry = None
        self._tools_full = None
        if mode == "tools":
            from harness.tools import ToolRegistry, get_default_tools, get_policy_tools

            if tools:
                self._tools_full = list(tools)
            elif self._policy is not None:
                # policy-as-tool: the policy owns motion, the LLM plans/verifies
                opts = {"use_vision": use_vision, **(policy_options or {})}
                self._tools_full = get_policy_tools(self._policy, **opts)
            else:
                self._tools_full = get_default_tools()
            self._tools = self._tools_full
            self._tool_registry = ToolRegistry(self._tools)

    # -- llm accounting -------------------------------------------------- #
    def _complete(self, messages, **kw):
        """Call the LLM, recording tokens and cost.

        Every provider already returns usage; nothing was reading it. Cost is
        left unknown (rather than zero) when the model has no verified price,
        so a blank cell never masquerades as "free".
        """
        resp = self._llm.complete(messages, **kw)
        self._llm_calls += 1
        usage = TokenUsage.from_raw(resp.usage)
        self._usage = self._usage + usage
        cost = estimate_cost(resp.model, usage)
        if cost is None:
            self._cost_known = False
        else:
            self._cost_usd += cost
        return resp

    def _record_usage(self, ep: Episode) -> None:
        ep.metadata["llm_calls"] = self._llm_calls
        # The budget is in LLM turns, not environment steps -- a query tool
        # consumes a turn without stepping the env, and one skill can step it
        # many times. Recording it lets classify_failure tell "spent the whole
        # budget" apart from "acted and got it wrong"; without it that branch
        # never fired and every exhausted budget was filed as task_failed.
        ep.metadata["max_steps"] = self._max_steps
        ep.metadata["usage"] = self._usage.to_dict()
        ep.metadata["cost_usd"] = round(self._cost_usd, 6) if self._cost_known else None

    # -- main loop ------------------------------------------------------- #
    def run(self, env: Env, *, seed: Optional[int] = None) -> Episode:
        self._usage = TokenUsage()
        self._cost_usd = 0.0
        self._cost_known = True
        self._llm_calls = 0

        ep = Episode(metadata={"mode": self._mode, "llm": self._llm.name, "env": env.name})
        obs = env.reset(seed=seed)
        ep.observations.append(obs)

        if self._mode == "skills":
            self._run_skills_mode(env, ep)
            ep.success = env.is_success() or bool(ep.infos and ep.infos[-1].get("success", False))
            ep.total_reward = sum(ep.rewards)
            ep.metadata["steps"] = ep.steps
            self._record_usage(ep)
            if self._recorder is not None:
                self._recorder.finish(success=ep.success, total_reward=ep.total_reward)
            return ep

        if self._mode == "tools":
            from harness.tools import ToolRegistry

            self._tools = self._filter_tools_for_env(env)
            self._tool_registry = ToolRegistry(self._tools)

        task = self._task_description or self._default_task(env)
        if self._mode == "tools":
            system = self._system_prompt_override or build_tools_system_prompt(task=task, tools=self._tools)
        else:
            system = self._system_prompt_override or build_system_prompt(
                task=task,
                action_space=env.action_space,
                mode=self._mode,
                skill_docs=get_skill_docs() if self._mode == "code" else "",
            )
        messages: list[ChatMessage] = [ChatMessage.system(system)]
        if self._two_clock:
            from harness.agent.context import DeliberationContext, MonitorContext, Transcript

            self._transcript = Transcript(system=system)
            self._monitor = MonitorContext(decisions=self._monitor_decisions,
                                           include_image=self._use_vision)
            self._deliberation = DeliberationContext()

        if self._mode == "plan":
            plan = self._complete(
                [
                    ChatMessage.system(system),
                    ChatMessage.user("Write a short step-by-step plan to accomplish the task. Be concise."),
                ],
                temperature=self._temperature,
            )
            messages.append(ChatMessage.assistant(plan.content))
            messages.append(ChatMessage.user("Now execute the plan, one action at a time."))

        def record(action: Action, result) -> None:
            ep.actions.append(action)
            ep.rewards.append(result.reward)
            ep.infos.append(result.info)
            ep.observations.append(result.obs)

        for step in range(self._max_steps):
            obs_text = env.get_text_state() or obs.to_text(env.observation_space.state_names) or "(no observation)"
            frame = env.render() if self._capture_frames else None

            if self._mode == "tools":
                self._tools_step(env, ep, messages, obs, obs_text, step, frame)
                if ep.observations:
                    obs = ep.observations[-1]  # tools may have stepped the env
                if ep.infos and ep.infos[-1].get("success", False):
                    break
                if ep.actions and ep.actions[-1].kind == "stop":
                    break
                continue

            user_text = build_observation_message(obs_text=obs_text, step=step, max_steps=self._max_steps)
            messages.append(self._observation_message(user_text, obs, frame))

            prompt = _serialize_messages(messages)
            resp = self._complete(messages, temperature=self._temperature)
            messages.append(ChatMessage.assistant(resp.content))

            if self._mode == "code":
                before = len(ep.actions)
                ctx = SkillContext(env, on_step=record)
                result = execute_code(_extract_code(resp.content), namespace=ctx.namespace())
                if not result["ok"]:
                    log.warning("code execution failed: %s", result["error"][:400])
                if ep.observations:
                    obs = ep.observations[-1]
                done = ctx.finished or env.is_success() or bool(ep.infos and ep.infos[-1].get("success", False))
                self._emit(
                    step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content,
                    action_dict={"kind": "code", "code": _extract_code(resp.content), "skill_steps": len(ep.actions) - before},
                    reward=sum(ep.rewards), success=done, info=dict(ep.infos[-1]) if ep.infos else {}, frame=frame,
                )
                if done:
                    break
                continue

            action = parse_action(resp.content, env.action_space)
            ep.actions.append(action)

            if action.kind == "stop":
                self._emit(step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content, action_dict=action_to_dict(action), reward=None, success=None, info={}, frame=frame)
                break

            result = env.step(action)
            ep.rewards.append(result.reward)
            ep.infos.append(result.info)
            ep.observations.append(result.obs)
            obs = result.obs

            self._emit(step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content, action_dict=action_to_dict(action), reward=result.reward, success=result.success, info=dict(result.info), frame=frame)

            if result.success or result.truncated:
                break

        ep.success = bool(ep.infos and ep.infos[-1].get("success", False))
        ep.total_reward = sum(ep.rewards)
        ep.metadata["steps"] = ep.steps
        self._record_usage(ep)
        if self._recorder is not None:
            self._recorder.finish(success=ep.success, total_reward=ep.total_reward)
        return ep

    def _frame_for_vision(self, obs, frame):
        if not self._use_vision:
            return None
        image = getattr(obs, "image", None)
        return frame if image is None else image

    def _is_planning_point(self) -> bool:
        """First turn, a subgoal boundary, or after a recovery."""
        t = self._transcript
        if t is None or not t.turns:
            return True
        return bool(getattr(self, "_needs_deliberation", False))

    def _feedback(self, messages, text: str) -> None:
        """Record tool feedback in whichever store the run is using."""
        if self._transcript is not None:
            self._transcript.record_feedback(text)
        else:
            messages.append(ChatMessage.user("Tool result: " + text))

    def _observation_message(self, obs_text: str, obs, frame) -> ChatMessage:
        """The per-turn observation, with the camera frame attached when asked for.

        Tools mode used to append text unconditionally, so `use_vision: true` with
        `mode: tools` was a silent no-op -- the flag was accepted, forwarded into
        the policy options, and the model never received a pixel. Silent
        degradation is worse than an error: every run looked like a vision run.
        """
        if not self._use_vision:
            return ChatMessage.user(obs_text)
        image = getattr(obs, "image", None)
        if image is None:
            image = frame
        if image is None:
            if not self._warned_no_image:
                self._warned_no_image = True
                log.warning("use_vision is set but the environment produced no frame; "
                            "sending text only -- this is NOT a vision evaluation")
            return ChatMessage.user(obs_text)
        from harness.perception.vision import encode_image

        return ChatMessage.user_vision(obs_text, encode_image(image))

    # -- tools mode ------------------------------------------------------ #
    def _tools_step(self, env, ep, messages, obs, obs_text, step, frame) -> None:
        from harness.tools import parse_tool_call

        image = self._frame_for_vision(obs, frame)
        if self._transcript is not None:
            # Fast clock by default; the slow one only at planning points. A planning
            # point is the first turn, a subgoal boundary, or a recovery -- anywhere a
            # decision needs the whole history rather than the last few moves.
            planning = self._is_planning_point()
            self._transcript.begin_turn(obs_text, deliberated=planning)
            view = self._deliberation if planning else self._monitor
            turn_messages = view.messages(self._transcript, obs_text, image=image)
        else:
            messages.append(self._observation_message(obs_text, obs, frame))
            turn_messages = messages
        prompt = _serialize_messages(turn_messages)
        resp = self._complete(turn_messages, temperature=self._temperature)
        if self._transcript is None:
            messages.append(ChatMessage.assistant(resp.content))

        name, args = parse_tool_call(resp.content)
        if self._transcript is not None:
            self._transcript.record_reply(
                resp.content,
                decision=f"{name}({', '.join(f'{k}={v}' for k, v in (args or {}).items())})"
                if name else "unparsed")

        if name is None or name not in self._tool_registry:
            hint = "Invalid tool call. Available tools: " + ", ".join(t.name for t in self._tools)
            self._feedback(messages, hint)
            self._emit(step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content, action_dict={"tool": name, "args": args}, reward=None, success=None, info={}, frame=frame)
            return

        if name == "done":
            self._emit(step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content, action_dict={"tool": "done", "args": args}, reward=None, success=None, info={}, frame=frame)
            ep.actions.append(Action(kind="stop", comment="done"))
            return

        tool = self._tool_registry.get(name)
        args = {k: v for k, v in (args or {}).items() if k != "on_step"}

        if getattr(tool, "closed_loop", False):
            self._closed_loop_tool_step(env, ep, messages, tool, name, args, obs_text, prompt, resp, step, frame)
            return

        try:
            result = tool.run(env, **args)
        except Exception as e:  # noqa: BLE001 - a bad tool call must not kill the episode
            log.warning("tool '%s' failed: %s", name, e)
            self._feedback(messages, f"Tool '{name}' failed: {e}. Check the arguments and try again.")
            self._emit(step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content, action_dict={"tool": name, "args": args, "error": str(e)}, reward=None, success=None, info={}, frame=frame)
            return

        if result.action is not None:
            r = env.step(result.action)
            ep.actions.append(result.action)
            ep.rewards.append(r.reward)
            ep.infos.append(r.info)
            ep.observations.append(r.obs)
            obs = r.obs
            self._emit(step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content, action_dict={"tool": name, "args": args}, reward=r.reward, success=r.success, info=dict(r.info), frame=frame)
            self._feedback(messages, result.feedback)
            return

        # pure perception / control tool (no env step)
        self._emit(step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content, action_dict={"tool": name, "args": args}, reward=None, success=None, info={}, frame=frame)
        self._feedback(messages, result.feedback)

    def _closed_loop_tool_step(self, env, ep, messages, tool, name, args, obs_text, prompt, resp, step, frame) -> None:
        """Run a tool that drives the env itself (e.g. run_policy) and record it.

        The tool consumes many env steps per call, so its inner steps are folded
        into the Episode via on_step; the trace gets one entry for the tool call.
        """
        before = len(ep.actions)

        def record(action: Action, result) -> None:
            ep.actions.append(action)
            ep.rewards.append(result.reward)
            ep.infos.append(result.info)
            ep.observations.append(result.obs)

        try:
            result = tool.run(env, on_step=record, **args)
        except Exception as e:  # noqa: BLE001 - a bad tool call must not kill the episode
            log.warning("closed-loop tool '%s' failed: %s", name, e)
            self._feedback(messages, f"Tool '{name}' failed: {e}. Check the arguments and try again.")
            self._emit(step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content, action_dict={"tool": name, "args": args, "error": str(e)}, reward=None, success=None, info={}, frame=frame)
            return

        inner = ep.actions[before:]
        reward = sum(ep.rewards[before:]) if inner else None
        self._emit(
            step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content,
            action_dict={"tool": name, "args": args, "steps": result.steps},
            reward=reward, success=result.success,
            info=dict(ep.infos[-1]) if ep.infos else {}, frame=frame,
        )
        messages.append(ChatMessage.user("Tool result: " + result.feedback))
        if result.done:
            ep.actions.append(Action(kind="stop", comment=f"{name} declared done"))

    # -- skills mode (plan -> execute with subgoal verification) -------- #
    def _run_skills_mode(self, env, ep) -> None:
        from harness.skills.executor import run_skill
        from harness.skills.planning import build_plan_prompt, parse_plan
        from harness.skills.registry import make_skill, skill_catalog

        task_spec = getattr(env, "task_spec", None)
        task_desc = (task_spec.description if task_spec is not None else "") or self._task_description or self._default_task(env)
        scene = env.get_text_state()

        def record(action, result) -> None:
            ep.actions.append(action)
            ep.rewards.append(result.reward)
            ep.infos.append(result.info)
            ep.observations.append(result.obs)

        # 1. understand -> plan
        msgs = build_plan_prompt(task=task_desc, scene=scene, catalog=skill_catalog())
        plan_resp = self._complete(msgs, temperature=self._temperature)
        plan = parse_plan(plan_resp.content)

        # fall back to the task spec's gold plan if the LLM plan is unusable
        if not plan and task_spec is not None and task_spec.steps:
            plan = [{"skill": s["skill"], "args": s.get("args", {})} for s in task_spec.steps]
        ep.metadata["plan"] = plan

        # 2. act -> run each skill closed-loop with subgoal verification
        for i, step in enumerate(plan):
            name = step.get("skill")
            args = step.get("args") or {}
            if name is None:
                continue
            skill = make_skill(name, **args)
            res = run_skill(env, skill, budget=self._max_steps, on_step=record)
            if not res.success:
                res = run_skill(env, skill, budget=self._max_steps, on_step=record)  # retry once
            ep.metadata[f"skill_{i}_{name}"] = {"success": res.success, "feedback": res.feedback, "steps": res.steps}
            if env.is_success():
                break

    # -- helpers --------------------------------------------------------- #
    def _emit(self, *, step, observation_text, prompt_messages, llm_response, action_dict, reward, success, info, frame) -> None:
        if self._recorder is None and self._on_step is None:
            return
        ts = TraceStep(
            step=step,
            observation_text=observation_text,
            prompt_messages=prompt_messages,
            llm_response=llm_response,
            action=action_dict,
            reward=reward,
            success=success,
            info=info,
            frame=frame if self._capture_frames else None,
        )
        if self._recorder is not None:
            self._recorder.steps.append(ts)
        if self._on_step is not None:
            self._on_step(ts)

    def _filter_tools_for_env(self, env) -> list:
        kind = getattr(env.action_space, "kind", "")
        if kind == "joint_position":
            # joint-space control: ee-space tools don't apply
            return [t for t in self._tools_full if t.name not in ("move_to", "move_delta")]
        # ee-space control (default): joint-space set_joints doesn't apply
        return [t for t in self._tools_full if t.name != "set_joints"]

    def _default_task(self, env: Env) -> str:
        return f"Control the robot to accomplish the task in the {env.name} environment."
