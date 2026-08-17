"""Built-in tools for robot-arm manipulation.

Action tools (map to a low-level Action), perception tools (query the env), and
control tools (signal completion). They use the object-aware query API on the
environment, so they work for TabletopEnv today and a Franka backend later.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from harness.tools.base import Tool, ToolResult
from harness.types import Action

_OBJ = {"type": "object", "properties": {}, "required": []}


def _fmt(pos) -> str:
    if pos is None:
        return "unknown"
    return "(" + ", ".join(f"{v:.3f}" for v in np.asarray(pos).ravel()) + ")"


class MoveToTool(Tool):
    name = "move_to"
    description = "Move the end-effector to an absolute position."
    parameters = {
        "type": "object",
        "properties": {
            "x": {"type": "number", "description": "x coordinate"},
            "y": {"type": "number", "description": "y coordinate"},
            "z": {"type": "number", "description": "z coordinate (default 0)"},
        },
        "required": ["x", "y"],
    }

    def run(self, env, x: float, y: float, z: float = 0.0, **kw: Any) -> ToolResult:
        return ToolResult(
            feedback=f"moving to ({x}, {y}, {z})",
            action=Action(kind="ee_pose", value=np.array([x, y, z], dtype=np.float32)),
        )


class MoveDeltaTool(Tool):
    name = "move_delta"
    description = "Move the end-effector by a small delta."
    parameters = {
        "type": "object",
        "properties": {
            "dx": {"type": "number"},
            "dy": {"type": "number"},
            "dz": {"type": "number", "description": "optional z delta"},
        },
        "required": ["dx", "dy"],
    }

    def run(self, env, dx: float, dy: float, dz: float = 0.0, **kw: Any) -> ToolResult:
        return ToolResult(
            feedback=f"moving by ({dx}, {dy}, {dz})",
            action=Action(kind="ee_delta", value=np.array([dx, dy, dz], dtype=np.float32)),
        )


class GraspTool(Tool):
    name = "grasp"
    description = "Close the gripper to grasp the object under it."
    parameters = _OBJ

    def run(self, env, **kw: Any) -> ToolResult:
        return ToolResult(feedback="closing gripper", action=Action(kind="noop", gripper=1.0))


class ReleaseTool(Tool):
    name = "release"
    description = "Open the gripper to release the grasped object."
    parameters = _OBJ

    def run(self, env, **kw: Any) -> ToolResult:
        return ToolResult(feedback="opening gripper", action=Action(kind="noop", gripper=0.0))


class SetJointsTool(Tool):
    name = "set_joints"
    description = "Set robot arm joint positions (radians)."
    parameters = {
        "type": "object",
        "properties": {"positions": {"type": "array", "items": {"type": "number"}}},
        "required": ["positions"],
    }

    def run(self, env, positions, **kw: Any) -> ToolResult:
        return ToolResult(
            feedback=f"setting joints {np.asarray(positions).tolist()}",
            action=Action(kind="joint_position", value=np.asarray(positions, dtype=np.float32)),
        )


class GetEEPoseTool(Tool):
    name = "get_end_effector_position"
    description = "Report the current end-effector position."
    parameters = _OBJ

    def run(self, env, **kw: Any) -> ToolResult:
        return ToolResult(feedback=f"end-effector at {_fmt(env.get_ee_pos())}")


class GetObjectPosTool(Tool):
    name = "get_object_position"
    description = "Report the position of a named object."
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "object name"}},
        "required": ["name"],
    }

    def run(self, env, name: str, **kw: Any) -> ToolResult:
        pos = env.get_object_pos(name)
        if pos is None:
            return ToolResult(feedback=f"object '{name}' not found; available: {env.list_objects()}")
        return ToolResult(feedback=f"object '{name}' at {_fmt(pos)}")


class ListObjectsTool(Tool):
    name = "list_objects"
    description = "List all objects in the scene with their positions."
    parameters = _OBJ

    def run(self, env, **kw: Any) -> ToolResult:
        objs = env.list_objects()
        if not objs:
            return ToolResult(feedback="no objects in the scene")
        parts = [f"{n} {_fmt(env.get_object_pos(n))}" for n in objs]
        return ToolResult(feedback="objects: " + ", ".join(parts))


class ListGoalsTool(Tool):
    name = "list_goals"
    description = "List all goal locations with their positions."
    parameters = _OBJ

    def run(self, env, **kw: Any) -> ToolResult:
        goals = env.list_goals()
        if not goals:
            return ToolResult(feedback="no goals in the scene")
        parts = [f"{n} {_fmt(env.get_goal_pos(n))}" for n in goals]
        return ToolResult(feedback="goals: " + ", ".join(parts))


class ListObstaclesTool(Tool):
    name = "list_obstacles"
    description = "List all obstacles with their positions."
    parameters = _OBJ

    def run(self, env, **kw: Any) -> ToolResult:
        obs = env.list_obstacles()
        if not obs:
            return ToolResult(feedback="no obstacles in the scene")
        return ToolResult(feedback="obstacles: " + ", ".join(obs))


class IsGraspedTool(Tool):
    name = "is_grasped"
    description = "Report whether the gripper is currently holding an object."
    parameters = _OBJ

    def run(self, env, **kw: Any) -> ToolResult:
        g = env.grasped_object()
        if g:
            return ToolResult(feedback=f"grasping object '{g}'")
        return ToolResult(feedback="not grasping anything")


class DoneTool(Tool):
    name = "done"
    description = "Declare the task finished."
    parameters = _OBJ

    def run(self, env, **kw: Any) -> ToolResult:
        return ToolResult(feedback="done", done=True)
