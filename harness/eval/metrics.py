"""Evaluation metrics.

``summarize`` keeps its old signature (a list of Episodes) so existing callers
and tests are unaffected, and gains a richer path: ``summarize_records`` works
from the flat per-episode log, which is what makes every number recomputable
without re-running inference.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from harness.eval.results import FailureMode
from harness.eval.stats import aggregate_over_tasks, beta_ci
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
    """Summarise a list of Episodes.

    Reports an interval alongside the point estimate: a success rate from a
    handful of episodes is nearly uninformative on its own, and printing it
    bare invites over-reading. Token and cost totals are included when the
    agent recorded them.
    """
    n = len(episodes)
    if n == 0:
        return {}

    successes = sum(1 for e in episodes if e.success)
    lo, hi = beta_ci(successes, n)

    out = {
        "episodes": n,
        "success_rate": successes / n,
        "success_ci_95": [round(lo, 4), round(hi, 4)],
        "ci_width_pp": round(100 * (hi - lo), 1),
        "mean_steps": sum(e.steps for e in episodes) / n,
        "mean_reward": sum(e.total_reward for e in episodes) / n,
    }

    calls = [e.metadata.get("llm_calls") for e in episodes]
    if any(c is not None for c in calls):
        out["llm_calls"] = sum(c or 0 for c in calls)

    tok: dict[str, int] = defaultdict(int)
    for e in episodes:
        for k, v in (e.metadata.get("usage") or {}).items():
            tok[k] += int(v or 0)
    if tok:
        out["usage"] = dict(tok)

    costs = [e.metadata.get("cost_usd") for e in episodes]
    if costs and all(c is not None for c in costs):
        total = sum(costs)
        out["cost_usd"] = round(total, 6)
        if successes:
            out["cost_per_success_usd"] = round(total / successes, 6)
    elif any(c is not None for c in costs):
        # partial pricing is worse than none -- say so rather than under-report
        out["cost_usd"] = None
        out["cost_note"] = "some episodes used a model with no verified price"

    return out


def oracle_steps_by_task(records: Sequence[dict]) -> dict:
    """Median oracle step count per task, for use as an efficiency reference.

    Manipulation has no geodesic optimum, so SPL's shortest-path denominator
    does not exist. A scripted oracle is the closest available stand-in -- the
    same substitution BEHAVIOR-1K makes when it normalises efficiency against
    human demonstrations.
    """
    per_task: dict[str, list[int]] = defaultdict(list)
    for r in records:
        if str(r.get("policy") or "").startswith("oracle") and r.get("success"):
            per_task[str(r.get("env_name") or "?")].append(int(r.get("episode_step") or 0))
    out = {}
    for task, steps in per_task.items():
        steps = sorted(s for s in steps if s > 0)
        if steps:
            out[task] = steps[len(steps) // 2]
    return out


def summarize_records(
    records: Sequence[dict],
    *,
    reliability_k: Iterable[int] = (2, 5, 10),
    oracle_steps: Optional[dict] = None,
) -> dict:
    """Summarise a flat per-episode log, grouped by model then task.

    This is the leaderboard view. Per-task grouping matters for more than
    presentation: pass^k must be averaged per task, and interval width is
    dominated by task count rather than by repeats per task.
    """
    if not records:
        return {}

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_model[str(r.get("policy") or "unknown")].append(r)

    models = {}
    for model, rows in sorted(by_model.items()):
        by_task: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_task[str(r.get("env_name") or r.get("task_name") or "?")].append(r)

        n = len(rows)
        k = sum(1 for r in rows if r.get("success"))
        lo, hi = beta_ci(k, n)

        per_task_counts = [
            (sum(1 for r in rs if r.get("success")), len(rs)) for rs in by_task.values()
        ]

        # score: successes count 1.0, failures contribute fractional progress
        scores = [
            1.0 if r.get("success") else (float(r["score"]) if r.get("score") is not None else 0.0)
            for r in rows
        ]

        fails: dict[str, int] = defaultdict(int)
        for r in rows:
            fm = r.get("failure_mode") or FailureMode.NONE
            if fm != FailureMode.NONE:
                fails[fm] += 1

        tok: dict[str, int] = defaultdict(int)
        for r in rows:
            for kk, vv in (r.get("usage") or {}).items():
                tok[kk] += int(vv or 0)

        costs = [r.get("cost_usd") for r in rows]
        cost = round(sum(c for c in costs if c is not None), 6) if any(
            c is not None for c in costs
        ) else None
        cost_known = all(c is not None for c in costs)

        entry = {
            "episodes": n,
            "tasks": len(by_task),
            "success_rate": round(k / n, 4),
            "success_ci_95": [round(lo, 4), round(hi, 4)],
            "ci_width_pp": round(100 * (hi - lo), 1),
            "score_mean": round(sum(scores) / n, 4),
            "mean_steps": round(sum(int(r.get("episode_step") or 0) for r in rows) / n, 2),
            "usage": dict(tok) or None,
            "cost_usd": cost if cost_known else None,
            "failure_modes": dict(fails) or None,
        }
        if cost_known and cost and k:
            entry["cost_per_success_usd"] = round(cost / k, 6)
        if not cost_known and cost is not None:
            entry["cost_note"] = "partial: some episodes used an unpriced model"

        for kk in reliability_k:
            v = aggregate_over_tasks(per_task_counts, kk, metric="pass_hat_k")
            if v is not None:
                entry[f"pass_hat_{kk}"] = round(v, 4)

        # harness-fault episodes are reported separately: they say nothing
        # about the model, and silently counting them as failures is a lie.
        not_model = sum(v for kk, v in fails.items() if kk in FailureMode.NOT_MODEL_FAULT)
        if not_model:
            entry["not_model_fault"] = not_model
            entry["not_model_fault_pct"] = round(100 * not_model / n, 1)

        # Efficiency, reported beside success and never folded into it: one
        # SPL number cannot distinguish "half the episodes failed" from "all
        # succeeded at twice the optimal path".
        if oracle_steps:
            ratios, spl = [], []
            for r in rows:
                task = str(r.get("env_name") or "?")
                ref = oracle_steps.get(task)
                took = int(r.get("episode_step") or 0)
                if not ref:
                    continue
                if r.get("success") and took > 0:
                    ratios.append(took / ref)
                # SoftSPL-style: graded score weighted by step efficiency
                grade = 1.0 if r.get("success") else (
                    float(r["score"]) if r.get("score") is not None else 0.0
                )
                spl.append(grade * (ref / max(took, ref)) if took else 0.0)
            if ratios:
                entry["steps_vs_oracle"] = round(sum(ratios) / len(ratios), 3)
            if spl:
                entry["soft_spl"] = round(sum(spl) / len(spl), 4)

        entry["per_task"] = {
            t: {
                "successes": sum(1 for r in rs if r.get("success")),
                "trials": len(rs),
            }
            for t, rs in sorted(by_task.items())
        }
        models[model] = entry

    return {"models": models, "episodes": len(records)}
