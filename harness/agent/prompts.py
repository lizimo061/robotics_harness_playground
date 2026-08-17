"""Prompt templates for the LLM controller."""
from __future__ import annotations

from harness.types import ActionSpace

_NL = chr(10)


def _action_instructions(action_space: ActionSpace) -> str:
    kind = action_space.kind
    if kind == "ee_delta":
        return (
            "The arm accepts incremental end-effector motion. Each action is a JSON object:"
            + _NL
            + '  {"action": "move", "delta": [dx, dy, dz], "gripper": <0 or 1>}'
            + _NL
            + "Use small deltas within the bounds above. gripper=1 closes/grasps, 0 opens."
            + _NL
            + 'You may also move to an absolute pose: {"action": "move_to", "pose": [x, y, z]}.'
        )
    if kind == "joint_position":
        names = ", ".join(action_space.joint_names) if action_space.joint_names else "joints"
        return (
            "The arm accepts joint-position control. Each action is a JSON object:"
            + _NL
            + '  {"action": "joints", "joint_positions": [<one value per joint>]}'
            + _NL
            + f"Joint order: {names}. Values are radians within the bounds above."
        )
    if kind == "discrete":
        return (
            "The action space is discrete. Each action is a JSON object:"
            + _NL
            + '  {"action": "value", "value": <integer index>}'
        )
    return 'Each action is a JSON object with "action" and "value" keys.'


def build_system_prompt(*, task: str, action_space: ActionSpace, mode: str = "json", skill_docs: str = "") -> str:
    lines = [
        "You are a robot control policy. You control a simulated robot arm to complete a task.",
        "You receive the current state as text each step and reply with ONE action.",
        "",
        "## Task",
        str(task),
        "",
        "## Action space",
        action_space.description or action_space.to_text(),
        "",
    ]
    if mode == "code":
        lines += [
            "## Mode: Code-as-Policies",
            "Write a short Python snippet that calls the available robot skills to make progress.",
            "You may call several skills in sequence. Call state() to read the current state and",
            "done() when finished. Return only the Python code (no markdown, no explanation).",
            "",
            "## Available skills",
            skill_docs,
        ]
    else:
        lines += [
            "## Output format",
            "Reply with exactly one JSON object and nothing else. Recognised forms:",
            _action_instructions(action_space),
            '  {"action": "stop"}   -- declare the task finished',
        ]
    return _NL.join(lines)


def build_tools_system_prompt(*, task: str, tools) -> str:
    lines = [
        "You are a robot control policy. Complete the task by calling tools, one at a time.",
        "You observe the state as text each turn and reply with ONE tool call.",
        "",
        "## Task",
        str(task),
        "",
        "## Tools",
    ]
    for t in tools:
        lines.append(f"- {t.signature()}: {t.description}")
    lines += [
        "",
        "## Output format",
        'Reply with exactly one JSON object: {"tool": "<name>", "args": {<arguments>}}',
        "Use small moves. Call done() when the task is finished.",
    ]
    return _NL.join(lines)


def build_observation_message(*, obs_text: str, step: int, max_steps: int) -> str:
    return (
        f"Step {step + 1}/{max_steps}."
        + _NL
        + "Current state:"
        + _NL
        + obs_text
        + _NL
        + "What is your next action?"
    )
