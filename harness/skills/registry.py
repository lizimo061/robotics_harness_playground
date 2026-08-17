"""Skill catalog and instantiation."""
from __future__ import annotations

from typing import Optional

from harness.skills.base import Skill
from harness.skills.builtin import OpenSkill, PickSkill, PlaceSkill, PressSkill, PutInSkill

_SKILL_CLASSES: dict[str, type] = {
    "pick": PickSkill,
    "place": PlaceSkill,
    "put_in": PutInSkill,
    "open": OpenSkill,
    "press": PressSkill,
}


def get_skill_classes() -> dict:
    return dict(_SKILL_CLASSES)


def make_skill(name: str, **args) -> Skill:
    if name not in _SKILL_CLASSES:
        raise KeyError(f"Unknown skill '{name}'. Available: {sorted(_SKILL_CLASSES)}")
    return _SKILL_CLASSES[name](**args)


def skill_catalog() -> list[dict]:
    """Return [{name, description, signature, parameters}] for prompts."""
    out = []
    for name, cls in _SKILL_CLASSES.items():
        inst = cls()
        out.append({
            "name": name,
            "description": inst.description,
            "signature": inst.signature(),
            "parameters": inst.parameters,
        })
    return out
