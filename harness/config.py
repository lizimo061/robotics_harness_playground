"""Typed configuration for the harness.

Configuration is expressed as YAML (or JSON) and loaded into dataclasses.
Every subsystem reads its settings from these dataclasses, so swapping an LLM
provider, environment, or agent is a one-line change in a YAML file.

Example::

    from harness.config import load_config
    cfg = load_config("configs/toy_pick_place.yaml")
"""
from __future__ import annotations

import dataclasses
import json
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _load_yaml_or_json(text: str) -> dict:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ImportError:
        data = json.loads(text)
    return data if isinstance(data, dict) else {}


def _instantiate(dc_cls, data: Optional[dict], defaults: Optional[dict] = None):
    data = data or {}
    defaults = defaults or {}
    try:
        hints = typing.get_type_hints(dc_cls)
    except Exception:
        hints = {}
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(dc_cls):
        if f.name in data:
            val = data[f.name]
        elif f.name in defaults:
            val = defaults[f.name]
        else:
            continue  # keep the declared default
        target = hints.get(f.name)
        if target is not None and dataclasses.is_dataclass(target) and isinstance(val, dict):
            val = _instantiate(target, val)
        kwargs[f.name] = val
    return dc_cls(**kwargs)


@dataclass
class LLMConfig:
    """Which LLM to call and how."""

    provider: str = "deepseek"  # deepseek | kimi | openai | claude | custom | ollama | vllm | mock
    model: str = ""  # empty => use the provider default (set explicitly to override)
    api_key: str = ""  # literal key; prefer api_key_env
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: float = 60.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvConfig:
    """Which simulated environment/task to run."""

    name: str = "toy_tabletop"  # toy_tabletop | gymnasium:<id> | genesis:<task> | robosuite:<task>
    task: str = "pick_and_place"
    seed: int = 0
    max_episode_steps: int = 200
    render: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Which agent/control strategy to use."""

    name: str = "llm_controller"
    mode: str = "json"  # json (emit actions) | code (Code-as-Policies) | plan (plan then act)
    max_steps: int = 50
    use_vision: bool = False
    system_prompt: str = ""  # override the default system prompt
    temperature: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    episodes: int = 1
    log_dir: str = "logs"
    save_trajectories: bool = True
    verbose: bool = True


@dataclass
class VizConfig:
    """Visualization settings: animation frames + LLM trace shown together."""

    enabled: bool = True
    backend: str = "html"  # html (replay file) | console (live print) | live (matplotlib) | none
    output: str = "logs/viz.html"  # path for the html backend
    #: when set, also write one video file per episode. The episode index, seed
    #: and outcome are appended to the stem, because a directory of videos is
    #: only useful if you can tell which one is the failure without opening them.
    video: str = ""
    #: capture every Nth environment step for the video. One tool call can drive
    #: many env steps, so frames are taken at the env boundary rather than once
    #: per LLM turn; raise this for long episodes.
    capture_every: int = 1
    fps: int = 8
    capture_frames: bool = True
    title: str = ""


@dataclass
class HarnessConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    seed: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "HarnessConfig":
        return _instantiate(cls, data)

    @classmethod
    def from_file(cls, path) -> "HarnessConfig":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = _load_yaml_or_json(text)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_config(path) -> HarnessConfig:
    return HarnessConfig.from_file(path)
