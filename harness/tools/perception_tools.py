"""Perception tools: what the agent can ask about the world, and at what cost.

These are the non-privileged counterparts to ``get_object_position``. Instead of
reading simulator state, they route through a :class:`~harness.perception.detect
.Detector` and answer from an image -- so the agent has to look, and a miss is a real
miss rather than an impossibility.

Both are query tools: they consume a turn and no environment step (see
harness/tools/base.py). Framed that way, the perception tier is not "harder because
the numbers are noisier" but "harder because looking costs turns".
"""
from __future__ import annotations

from typing import Any, Optional

from harness.tools.base import Tool, ToolResult
from harness.utils.logging import get_logger

log = get_logger("harness.tools.perception")

_QUERY = {
    "type": "object",
    "properties": {"query": {"type": "string",
                             "description": "what to look for, in plain words"}},
    "required": ["query"],
}


class DetectTool(Tool):
    """Open-vocabulary detection over the current camera frame."""

    name = "detect"
    description = ("Look for something in the current camera view and report where it "
                   "is. Returns nothing if it is not visible.")
    parameters = _QUERY

    def __init__(self, detector, *, max_results: int = 5) -> None:
        self._detector = detector
        self._max = int(max_results)

    def run(self, env, query: str = "", **kw: Any) -> ToolResult:
        query = str(query or "").strip()
        if not query:
            return ToolResult(feedback="detect needs a 'query' describing what to look for.")
        image = _frame(env)
        binder = getattr(self._detector, "bind", None)
        if binder is not None:
            binder(env)
        found = self._detector.detect(image, query) or []
        if not found:
            # A negative is information, not an error: it may be occluded, out of
            # frame, or absent. Saying which is the agent's job.
            return ToolResult(feedback=f"'{query}' was not detected in the current view.")
        lines = [d.describe() for d in found[:self._max]]
        return ToolResult(feedback=f"detected {len(found)} for '{query}': " + "; ".join(lines))


class PointAtTool(Tool):
    """Molmo-style pointing: one pixel for the thing named."""

    name = "point_at"
    description = ("Point at the named thing in the current view and report its pixel "
                   "location and, when available, its 3-D position.")
    parameters = _QUERY

    def __init__(self, detector) -> None:
        self._detector = detector

    def run(self, env, query: str = "", **kw: Any) -> ToolResult:
        query = str(query or "").strip()
        if not query:
            return ToolResult(feedback="point_at needs a 'query'.")
        binder = getattr(self._detector, "bind", None)
        if binder is not None:
            binder(env)
        found = self._detector.detect(_frame(env), query) or []
        if not found:
            return ToolResult(feedback=f"cannot point at '{query}': not detected.")
        best = max(found, key=lambda d: d.confidence)
        return ToolResult(feedback=f"'{query}': {best.describe()}")


def _frame(env):
    """The current camera frame, or None. A detector with no image must cope."""
    try:
        return env.render()
    except Exception as e:  # noqa: BLE001 - a render fault is not a task failure
        log.debug("render() failed while detecting: %s", e)
        return None
