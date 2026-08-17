"""Trajectory logging to JSONL."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harness.types import Action, Episode


def _action_to_dict(a: Action) -> dict:
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


def episode_to_dict(ep: Episode) -> dict:
    return {
        "success": ep.success,
        "steps": ep.steps,
        "total_reward": ep.total_reward,
        "rewards": ep.rewards,
        "actions": [_action_to_dict(a) for a in ep.actions],
        "infos": ep.infos,
        "metadata": ep.metadata,
    }


class TrajectoryLogger:
    """Appends episodes to a JSONL file and saves the run config alongside."""

    def __init__(self, log_dir: str = "logs", run_name: str = "run") -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._run_name = run_name
        self._path = self._dir / f"{run_name}.jsonl"
        self._count = 0

    def log(self, ep: Episode) -> None:
        rec = episode_to_dict(ep)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self._count += 1

    def log_config(self, cfg: dict) -> None:
        cfg_path = self._dir / f"{self._run_name}.config.json"
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return self._count
