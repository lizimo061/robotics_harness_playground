"""Parse an LLM response into a structured Action.

LLMs often wrap their output in markdown code fences or add prose, so the
parser is deliberately lenient: it extracts the first balanced JSON object and
accepts both a canonical form and friendly shorthands.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from harness.types import Action, ActionSpace

_FENCE = chr(96) * 3  # three backticks


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith(_FENCE):
        lines = t.splitlines()
        if lines and lines[0].strip().startswith(_FENCE):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith(_FENCE):
            lines = lines[:-1]
        t = chr(10).join(lines).strip()
    return t


def extract_json(text: str) -> Optional[dict]:
    """Return the first balanced JSON object in the text, or None."""
    if not text:
        return None
    text = _strip_fence(text)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    bs = chr(92)  # backslash
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == bs:
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _to_array(value: Any) -> Optional[np.ndarray]:
    if value is None or isinstance(value, str):
        return None
    try:
        return np.asarray(value, dtype=np.float32).ravel()
    except (ValueError, TypeError):
        return None


def parse_action(text: str, action_space: Optional[ActionSpace] = None) -> Action:
    """Parse an LLM response into an Action (never raises; noop on garbage)."""
    data = extract_json(text)
    if data is None:
        return Action(kind="noop", comment=(text or "").strip()[:80])

    comment = str(data.get("comment") or "")
    kind = data.get("kind")
    value = data.get("value")
    gripper = data.get("gripper")

    # friendly top-level action name
    act = data.get("action")
    if act is not None:
        a = str(act).strip().lower()
        if a in ("stop", "done", "finish", "success", "hold", "terminate"):
            return Action(kind="stop", comment=comment or a)
        if a in ("noop", "wait", "stay", "none"):
            return Action(kind="noop", gripper=_as_float(gripper), comment=comment or a)
        if a in ("open", "release"):
            return Action(kind="noop", gripper=0.0, comment=comment or a)
        if a in ("close", "grasp"):
            return Action(kind="noop", gripper=1.0, comment=comment or a)
        if a in ("move", "delta", "move_delta"):
            kind = "ee_delta"
            value = data.get("delta", value)
        elif a in ("move_to", "goto", "set_pose", "pose"):
            kind = "ee_pose"
            value = data.get("pose", data.get("position", value))
        elif a == "gripper":
            g = value if value is not None else gripper
            return Action(kind="noop", gripper=_as_float(g), comment=comment)
        elif a in ("joints", "joint_positions", "set_joints"):
            kind = "joint_position"
            value = data.get("joint_positions", data.get("joints", value))

    # canonical kind mapping
    if kind is not None:
        k = str(kind).strip().lower()
        if k in ("stop", "done", "finish", "terminate"):
            return Action(kind="stop", comment=comment)
        if k in ("noop", "wait"):
            return Action(kind="noop", gripper=_as_float(gripper), comment=comment)
        if k in ("ee_delta", "delta", "move"):
            kind = "ee_delta"
            value = value if value is not None else data.get("delta")
        elif k in ("ee_pose", "pose", "move_to"):
            kind = "ee_pose"
            value = value if value is not None else data.get("pose", data.get("position"))
        elif k in ("joint_position", "joint_positions", "joints"):
            kind = "joint_position"
            value = value if value is not None else data.get("joint_positions", data.get("joints"))
        elif k == "gripper":
            g = value if value is not None else gripper
            return Action(kind="noop", gripper=_as_float(g), comment=comment)
        else:
            kind = "noop"

    arr = _to_array(value)

    # clip to the action-space bounds when known
    if arr is not None and action_space is not None and action_space.low.size and action_space.high.size:
        dim = min(action_space.dim, arr.size)
        arr = arr.copy()
        arr[:dim] = np.clip(arr[:dim], action_space.low[:dim], action_space.high[:dim])

    return Action(kind=kind or "noop", value=arr, gripper=_as_float(gripper), comment=comment)


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None
