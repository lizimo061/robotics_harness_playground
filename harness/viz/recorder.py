"""Trace recording: capture per-step observations, prompts, responses, actions and frames."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from harness.types import Action


@dataclass
class TraceStep:
    step: int
    observation_text: str = ""
    prompt_messages: list = field(default_factory=list)  # serialized messages (OpenAI style)
    llm_response: str = ""
    action: dict = field(default_factory=dict)
    reward: Optional[float] = None
    success: Optional[bool] = None
    info: dict = field(default_factory=dict)
    frame: Optional[np.ndarray] = None


def action_to_dict(a: Action) -> dict:
    v = a.value
    if v is not None:
        v = np.asarray(v).tolist()
    return {
        "kind": a.kind,
        "value": v,
        "gripper": a.gripper,
        "duration": a.duration,
        "comment": a.comment,
    }


class TraceRecorder:
    """Accumulates per-step trace data during an episode."""

    def __init__(self, *, capture_frames: bool = True, metadata: Optional[dict] = None) -> None:
        self.capture_frames = capture_frames
        self.metadata = metadata or {}
        self.steps: list[TraceStep] = []
        self.final_success: Optional[bool] = None
        self.total_reward: float = 0.0

    def record(self, **kw: Any) -> TraceStep:
        ts = TraceStep(
            step=kw.get("step", len(self.steps)),
            observation_text=kw.get("observation_text", ""),
            prompt_messages=list(kw.get("prompt_messages") or []),
            llm_response=kw.get("llm_response", ""),
            action=dict(kw.get("action") or {}),
            reward=kw.get("reward"),
            success=kw.get("success"),
            info=dict(kw.get("info") or {}),
            frame=kw.get("frame") if self.capture_frames else None,
        )
        self.steps.append(ts)
        return ts

    def finish(self, *, success: Optional[bool] = None, total_reward: float = 0.0) -> None:
        self.final_success = success
        self.total_reward = total_reward

    @staticmethod
    def step_to_dict(s: TraceStep) -> dict:
        return {
            "step": s.step,
            "observation_text": s.observation_text,
            "prompt_messages": s.prompt_messages,
            "llm_response": s.llm_response,
            "action": s.action,
            "reward": s.reward,
            "success": s.success,
            "info": s.info,
        }

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "final_success": self.final_success,
            "total_reward": self.total_reward,
            "steps": [self.step_to_dict(s) for s in self.steps],
        }
