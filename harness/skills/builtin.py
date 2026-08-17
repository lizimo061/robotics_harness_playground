"""Built-in composite skills, built on primitive actions + env queries.

Each skill is a tiny closed-loop state machine: satisfied(env) tells the
executor when to stop, plan_action(env) emits the next primitive Action.

These are 2D-first (xy distance); a 3D Franka backend uses the same interface
with an approach-height offset (see docs/guides/long-horizon.md).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from harness.skills.base import Skill, dist2d
from harness.types import Action


class PickSkill(Skill):
    name = "pick"
    description = "Pick up a named object with the gripper."
    parameters = {"type": "object", "properties": {"object": {"type": "string"}}, "required": ["object"]}

    def satisfied(self, env) -> bool:
        return env.grasped_object() == self.args.get("object")

    def plan_action(self, env) -> Optional[Action]:
        if self.satisfied(env):
            return None
        obj = env.get_object_pos(self.args["object"])
        ee = env.get_ee_pos()
        if obj is None or ee is None:
            return None
        if dist2d(env, ee, obj) < self.tolerance:
            return Action(kind="noop", gripper=1.0)
        return Action(kind="ee_pose", value=np.asarray(obj, dtype=np.float32))


class PlaceSkill(Skill):
    name = "place"
    description = "Carry the grasped object to a target position and release it."
    parameters = {
        "type": "object",
        "properties": {"object": {"type": "string"}, "target": {"type": "array", "items": {"type": "number"}}},
        "required": ["object", "target"],
    }

    def satisfied(self, env) -> bool:
        obj = env.get_object_pos(self.args["object"])
        tgt = np.asarray(self.args["target"], dtype=float)
        if obj is None:
            return False
        return dist2d(env, obj, tgt) < self.tolerance and env.grasped_object() != self.args["object"]

    def plan_action(self, env) -> Optional[Action]:
        oname = self.args["object"]
        tgt = np.asarray(self.args["target"], dtype=float)
        if env.grasped_object() == oname:
            if dist2d(env, env.get_ee_pos(), tgt) < self.tolerance:
                return Action(kind="noop", gripper=0.0)
            return Action(kind="ee_pose", value=tgt.astype(np.float32))
        if env.grasped_object() is not None:
            return Action(kind="noop", gripper=0.0)  # drop whatever else we hold
        obj = env.get_object_pos(oname)
        if obj is None:
            return None
        if dist2d(env, env.get_ee_pos(), obj) < self.tolerance:
            return Action(kind="noop", gripper=1.0)
        return Action(kind="ee_pose", value=np.asarray(obj, dtype=np.float32))


class PutInSkill(Skill):
    name = "put_in"
    description = "Put an object into a container (opening the container door if needed)."
    parameters = {
        "type": "object",
        "properties": {"object": {"type": "string"}, "container": {"type": "string"}},
        "required": ["object", "container"],
    }

    def satisfied(self, env) -> bool:
        obj = env.get_object_pos(self.args["object"])
        interior = env.get_container_interior(self.args["container"])
        if obj is None or interior is None:
            return False
        return dist2d(env, obj, interior) < self.tolerance and env.grasped_object() != self.args["object"]

    def plan_action(self, env) -> Optional[Action]:
        cname = self.args["container"]
        oname = self.args["object"]
        interior = env.get_container_interior(cname)

        # 1. ensure the container is open
        door = env.get_door_pos(cname)
        if door is not None and not env.is_container_open(cname):
            if dist2d(env, env.get_ee_pos(), door) < self.tolerance:
                return Action(kind="noop", gripper=1.0)
            return Action(kind="ee_pose", value=np.asarray(door, dtype=np.float32))

        # 2. pick the object
        if env.grasped_object() != oname:
            if env.grasped_object() is not None:
                return Action(kind="noop", gripper=0.0)
            obj = env.get_object_pos(oname)
            if obj is None:
                return None
            if dist2d(env, env.get_ee_pos(), obj) < self.tolerance:
                return Action(kind="noop", gripper=1.0)
            return Action(kind="ee_pose", value=np.asarray(obj, dtype=np.float32))

        # 3. carry to the interior and release
        if interior is None:
            return None
        if dist2d(env, env.get_ee_pos(), interior) < self.tolerance:
            return Action(kind="noop", gripper=0.0)
        return Action(kind="ee_pose", value=np.asarray(interior, dtype=np.float32))


class OpenSkill(Skill):
    name = "open"
    description = "Open a container door."
    parameters = {"type": "object", "properties": {"container": {"type": "string"}}, "required": ["container"]}

    def satisfied(self, env) -> bool:
        return env.is_container_open(self.args["container"])

    def plan_action(self, env) -> Optional[Action]:
        door = env.get_door_pos(self.args["container"])
        if door is None:
            return None
        if dist2d(env, env.get_ee_pos(), door) < self.tolerance:
            return Action(kind="noop", gripper=1.0)
        return Action(kind="ee_pose", value=np.asarray(door, dtype=np.float32))


class PressSkill(Skill):
    name = "press"
    description = "Press a button."
    parameters = {"type": "object", "properties": {"button": {"type": "string"}}, "required": ["button"]}

    def satisfied(self, env) -> bool:
        return env.is_button_pressed(self.args["button"])

    def plan_action(self, env) -> Optional[Action]:
        pos = env.get_button_pos(self.args["button"])
        if pos is None:
            return None
        if dist2d(env, env.get_ee_pos(), pos) < self.tolerance:
            return Action(kind="noop", gripper=1.0)
        return Action(kind="ee_pose", value=np.asarray(pos, dtype=np.float32))
