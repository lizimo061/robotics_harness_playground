"""Planning: decompose a natural-language task into an ordered skill plan.

This is the "understanding" half of long-horizon control: the LLM reads the task
description + scene and produces a list of skill calls. The "action" half runs
each skill closed-loop (harness/skills/executor.py) with subgoal verification.
"""
from __future__ import annotations

from harness.agent.action_parser import extract_json
from harness.llm.base import ChatMessage

_NL = chr(10)


def build_plan_prompt(*, task: str, scene: str, catalog: list[dict]) -> list[ChatMessage]:
    skill_lines = [f"- {c['signature']}: {c['description']}" for c in catalog]
    system = (
        "You are a task planner for a robot arm. Given a task description and the "
        "current scene, decompose the task into an ordered list of skills."
    )
    user = (
        f"Task: {task}" + _NL + _NL
        + "Scene:" + _NL + scene + _NL + _NL
        + "Available skills:" + _NL + _NL.join(skill_lines) + _NL + _NL
        + 'Return a JSON list of steps: [{"skill": "<name>", "args": {...}}, ...]'
    )
    return [ChatMessage.system(system), ChatMessage.user(user)]


def parse_plan(text: str) -> list[dict]:
    """Parse a plan from an LLM response. Returns a list of {"skill","args"}."""
    data = extract_json(text)
    if data is None:
        return []
    if isinstance(data, dict) and "plan" in data:
        data = data["plan"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    plan = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("skill") or item.get("name") or item.get("action")
        args = item.get("args") or item.get("arguments") or item.get("parameters") or {}
        if name:
            plan.append({"skill": str(name), "args": dict(args) if isinstance(args, dict) else {}})
    return plan
