"""Evaluation metrics."""
from __future__ import annotations

from dataclasses import dataclass, field

from harness.types import Episode


@dataclass
class EpisodeMetrics:
    success: bool = False
    steps: int = 0
    total_reward: float = 0.0
    metadata: dict = field(default_factory=dict)


def episode_metrics(ep: Episode) -> EpisodeMetrics:
    return EpisodeMetrics(
        success=ep.success, steps=ep.steps, total_reward=ep.total_reward, metadata=ep.metadata
    )


def summarize(episodes: list[Episode]) -> dict:
    n = len(episodes)
    if n == 0:
        return {}
    successes = sum(1 for e in episodes if e.success)
    return {
        "episodes": n,
        "success_rate": successes / n,
        "mean_steps": sum(e.steps for e in episodes) / n,
        "mean_reward": sum(e.total_reward for e in episodes) / n,
    }
