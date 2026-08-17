"""Robotics Harness.

A modular framework for controlling simulated robots with large language models.

Layout:
    harness.config      -- typed configuration (YAML/JSON -> dataclasses)
    harness.types       -- shared dataclasses (Obs, Action, StepResult, Episode)
    harness.llm         -- LLM clients (DeepSeek, Kimi, OpenAI, Claude, mock, ...)
    harness.envs        -- environment adapters (toy, gymnasium, genesis, robosuite)
    harness.robot       -- robot models & action-space helpers
    harness.agent       -- the control loop that connects LLM <-> environment
    harness.perception  -- vision helpers (image encoding)
    harness.eval        -- metrics & trajectory logging
"""

from harness.config import (
    HarnessConfig,
    LLMConfig,
    EnvConfig,
    AgentConfig,
    EvalConfig,
    load_config,
)
from harness.types import Obs, Action, StepResult, Episode, ActionSpace, ObservationSpace
from harness.runner import run_eval

__version__ = "0.1.0"

__all__ = [
    "HarnessConfig",
    "LLMConfig",
    "EnvConfig",
    "AgentConfig",
    "EvalConfig",
    "load_config",
    "Obs",
    "Action",
    "StepResult",
    "Episode",
    "ActionSpace",
    "ObservationSpace",
    "run_eval",
    "__version__",
]
