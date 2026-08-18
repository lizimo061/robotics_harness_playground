"""Tool registry and tool-call parsing."""
from __future__ import annotations

from typing import Optional

from harness.agent.action_parser import extract_json
from harness.tools.base import Tool
from harness.tools.builtin import (
    DoneTool,
    GetEEPoseTool,
    GetObjectPosTool,
    GraspTool,
    IsGraspedTool,
    ListGoalsTool,
    ListObjectsTool,
    ListObstaclesTool,
    MoveDeltaTool,
    MoveToTool,
    ReleaseTool,
    SetJointsTool,
)


class ToolRegistry:
    def __init__(self, tools: Optional[list[Tool]] = None) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in (tools or [])}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'. Available: {sorted(self._tools)}")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def tools(self) -> list[Tool]:
        return list(self._tools.values())


#: Which query tools each scaffolding tier exposes. The motion tools are the same in
#: every tier -- the variable under study is what the agent can KNOW, not what it can
#: do. Naming the tiers after CaP-X's ladder keeps the comparison legible.
TIERS = ("privileged", "perception")

#: tools that read simulator ground truth
_PRIVILEGED_QUERIES = ("get_object_position", "list_objects", "list_goals",
                       "list_obstacles")


def tools_for_tier(tools, tier: str, detector=None) -> list:
    """Filter a toolset to a scaffolding tier, adding perception tools as needed.

    ``privileged`` keeps the ground-truth queries (the default, and what every result
    in this repo before now was measured with). ``perception`` removes them and offers
    ``detect``/``point_at`` instead, so object locations must be *looked up by
    looking*. Motion, gripper and `done` tools are untouched in both.

    A gap between the two tiers on the same task is the CaP-X measurement: how much of
    a score was the designer's scaffolding rather than the agent.
    """
    tier = str(tier or "privileged").lower()
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {list(TIERS)}")
    if tier == "privileged":
        return list(tools)
    if detector is None:
        raise ValueError("tier 'perception' needs a detector; pass detector=... "
                         "(see harness.perception.detect.get_detector)")
    from harness.tools.perception_tools import DetectTool, PointAtTool

    kept = [t for t in tools if t.name not in _PRIVILEGED_QUERIES]
    return [*kept, DetectTool(detector), PointAtTool(detector)]


def get_default_tools(policy=None, **policy_kwargs) -> list[Tool]:
    """The standard toolset; prepends run_policy when a policy is supplied."""
    tools: list[Tool] = []
    if policy is not None:
        from harness.tools.policy_tool import RunPolicyTool

        tools.append(RunPolicyTool(policy, **policy_kwargs))
    tools += [
        MoveToTool(),
        MoveDeltaTool(),
        GraspTool(),
        ReleaseTool(),
        SetJointsTool(),
        GetEEPoseTool(),
        GetObjectPosTool(),
        ListObjectsTool(),
        ListGoalsTool(),
        ListObstaclesTool(),
        IsGraspedTool(),
        DoneTool(),
    ]
    return tools


def get_policy_tools(policy, **policy_kwargs) -> list[Tool]:
    """Policy-centric toolset for continuous-control benchmarks (RoboLab).

    All motion goes through run_policy, so the hand-written ee/joint action tools
    are dropped -- their 2D/3D coordinate model does not match a RoboLab action
    vector, and mixing the two invites the LLM to fight the policy for control.
    What is left: delegate motion, perceive, declare done.
    """
    from harness.tools.policy_tool import (
        AbortPolicyTool,
        ContinuePolicyTool,
        RunPolicyTool,
    )

    runner = RunPolicyTool(policy, **policy_kwargs)
    # continue/abort are only meaningful when the rollout can pause; offering them
    # without a monitor cadence would advertise a choice the agent cannot make.
    interruptible = [ContinuePolicyTool(runner), AbortPolicyTool(runner)] \
        if getattr(runner, "_monitor_every", 0) else []

    return [
        runner,
        *interruptible,
        GetEEPoseTool(),
        GetObjectPosTool(),
        ListObjectsTool(),
        ListGoalsTool(),
        IsGraspedTool(),
        DoneTool(),
    ]


def parse_tool_call(text: str):
    """Parse an LLM response into (tool_name, args). Returns (None, None) on failure.

    Accepts: {"tool": name, "args": {...}} | {"name":..., "arguments": {...}}
    """
    data = extract_json(text)
    if not isinstance(data, dict):
        return None, None
    name = data.get("tool") or data.get("name") or data.get("action")
    args = data.get("args") or data.get("arguments") or data.get("parameters")
    if args is None and "args" not in data and "arguments" not in data:
        # some models inline the args alongside the tool name
        args = {k: v for k, v in data.items() if k not in ("tool", "name", "action", "args", "arguments", "parameters")}
    if not isinstance(args, dict):
        args = {}
    return name, args
