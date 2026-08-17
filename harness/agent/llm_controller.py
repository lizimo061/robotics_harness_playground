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
        system_prompt: str = "",
        temperature: Optional[float] = None,
        task_description: str = "",
        recorder: Optional[TraceRecorder] = None,
        on_step: Optional[Callable[[TraceStep], None]] = None,
        tools: Optional[list] = None,
        **kwargs,
    ) -> None:
        self._llm = llm
        self._mode = mode
        self._max_steps = max_steps
        self._use_vision = use_vision
        self._system_prompt_override = system_prompt
        self._temperature = temperature
        self._task_description = task_description
        self._recorder = recorder
        self._on_step = on_step
        self._capture_frames = (recorder is not None and recorder.capture_frames) or on_step is not None

        self._tools = None
        self._tool_registry = None
        self._tools_full = None
        if mode == "tools":
            from harness.tools import ToolRegistry, get_default_tools

            self._tools_full = list(tools) if tools else get_default_tools()
            self._tools = self._tools_full
            self._tool_registry = ToolRegistry(self._tools)

    # -- main loop ------------------------------------------------------- #
    def run(self, env: Env, *, seed: Optional[int] = None) -> Episode:
        ep = Episode(metadata={"mode": self._mode, "llm": self._llm.name, "env": env.name})
        obs = env.reset(seed=seed)
        ep.observations.append(obs)

        if self._mode == "skills":
            self._run_skills_mode(env, ep)
            ep.success = env.is_success() or bool(ep.infos and ep.infos[-1].get("success", False))
            ep.total_reward = sum(ep.rewards)
            ep.metadata["steps"] = ep.steps
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

        if self._mode == "plan":
            plan = self._llm.complete(
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
                if ep.infos and ep.infos[-1].get("success", False):
                    break
                if ep.actions and ep.actions[-1].kind == "stop":
                    break
                continue

            user_text = build_observation_message(obs_text=obs_text, step=step, max_steps=self._max_steps)
            if self._use_vision and obs.image is not None:
                from harness.perception.vision import encode_image

                b64 = encode_image(obs.image)
                messages.append(ChatMessage.user_vision(user_text, b64))
            else:
                messages.append(ChatMessage.user(user_text))

            prompt = _serialize_messages(messages)
            resp = self._llm.complete(messages, temperature=self._temperature)
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
        if self._recorder is not None:
            self._recorder.finish(success=ep.success, total_reward=ep.total_reward)
        return ep

    # -- tools mode ------------------------------------------------------ #
    def _tools_step(self, env, ep, messages, obs, obs_text, step, frame) -> None:
        from harness.tools import parse_tool_call

        messages.append(ChatMessage.user(obs_text))
        prompt = _serialize_messages(messages)
        resp = self._llm.complete(messages, temperature=self._temperature)
        messages.append(ChatMessage.assistant(resp.content))

        name, args = parse_tool_call(resp.content)

        if name is None or name not in self._tool_registry:
            hint = "Invalid tool call. Available tools: " + ", ".join(t.name for t in self._tools)
            messages.append(ChatMessage.user(hint))
            self._emit(step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content, action_dict={"tool": name, "args": args}, reward=None, success=None, info={}, frame=frame)
            return

        if name == "done":
            self._emit(step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content, action_dict={"tool": "done", "args": args}, reward=None, success=None, info={}, frame=frame)
            ep.actions.append(Action(kind="stop", comment="done"))
            return

        tool = self._tool_registry.get(name)
        result = tool.run(env, **args)

        if result.action is not None:
            r = env.step(result.action)
            ep.actions.append(result.action)
            ep.rewards.append(r.reward)
            ep.infos.append(r.info)
            ep.observations.append(r.obs)
            obs = r.obs
            self._emit(step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content, action_dict={"tool": name, "args": args}, reward=r.reward, success=r.success, info=dict(r.info), frame=frame)
            messages.append(ChatMessage.user("Tool result: " + result.feedback))
            return

        # pure perception / control tool (no env step)
        self._emit(step=step, observation_text=obs_text, prompt_messages=prompt, llm_response=resp.content, action_dict={"tool": name, "args": args}, reward=None, success=None, info={}, frame=frame)
        messages.append(ChatMessage.user("Tool result: " + result.feedback))

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
        plan_resp = self._llm.complete(msgs, temperature=self._temperature)
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
