"""Command-line entry point.

Two surfaces:

    harness <config.yaml>                  # one (env, model) evaluation
    harness job <job.yaml> [--resume]      # the tasks x agents x seeds grid
    harness job <job.yaml> --summary-only  # re-aggregate from disk, no runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from harness.config import load_config
from harness.runner import run_eval


def _load_job_config(path: str):
    """Read a job config, tolerating YAML or JSON."""
    from harness.config import _load_yaml_or_json
    from harness.eval.job import AgentSpec, JobConfig

    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text) if path.endswith(".json") else _load_yaml_or_json(text)

    agents = [AgentSpec(**a) for a in (data.pop("agents", None) or [])]
    known = set(JobConfig.__dataclass_fields__)
    cfg = JobConfig(**{k: v for k, v in data.items() if k in known})
    cfg.agents = agents
    return cfg


def _print_leaderboard(summary: dict) -> None:
    from harness.eval.stats import rank_interval

    models = (summary.get("leaderboard") or {}).get("models") or {}
    if not models:
        print("no results")
        return
    names = list(models)
    scores = [models[m]["success_rate"] for m in names]
    ivs = [tuple(models[m]["success_ci_95"]) for m in names]
    ranks = rank_interval(scores, ivs)

    order = sorted(range(len(names)), key=lambda i: (-scores[i], names[i]))
    head = (f"{'rank':>6}  {'agent':20s} {'success':>8} {'95% CI':>16} "
            f"{'pass^2':>7} {'score':>6} {'st/orc':>7} {'cost':>10}")
    print(head)
    print("-" * len(head))
    for i in order:
        m, e = names[i], models[names[i]]
        lo, hi = e["success_ci_95"]
        r = f"{ranks[i][0]}-{ranks[i][1]}" if ranks[i][0] != ranks[i][1] else str(ranks[i][0])
        cost = f"${e['cost_usd']:.4f}" if e.get("cost_usd") else "-"
        print(f"{r:>6}  {m:20s} {100 * e['success_rate']:7.1f}% "
              f"[{100 * lo:5.1f},{100 * hi:5.1f}] {e.get('pass_hat_2', '-'):>7} "
              f"{e['score_mean']:6.2f} {e.get('steps_vs_oracle', '-'):>7} {cost:>10}")
    if summary.get("infra_failures"):
        print(f"\ninfra failures (denominator unchanged): {summary['infra_failures']}")
    print(f"\nresults: {summary.get('job_dir') or summary.get('run_dir')}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run a robotics harness task or job.")
    sub = parser.add_subparsers(dest="command")

    job = sub.add_parser("job", help="run a tasks x agents x seeds grid")
    job.add_argument("job_config", help="path to a job YAML/JSON config")
    job.add_argument("--no-resume", action="store_true",
                     help="start fresh instead of resuming an existing run")
    job.add_argument("--concurrency", type=int, default=None)
    job.add_argument("--summary-only", action="store_true",
                     help="re-aggregate an existing job from disk without running trials")

    # default (no subcommand): single-config evaluation
    parser.add_argument("config", nargs="?", help="path to a YAML or JSON config file")
    parser.add_argument("--episodes", type=int, default=None, help="override the episode count")

    args = parser.parse_args(argv)

    if args.command == "job":
        from harness.eval.job import build_summary, load_job, run_job

        cfg = _load_job_config(args.job_config)
        if args.concurrency is not None:
            cfg.concurrency = args.concurrency
        if args.summary_only:
            summary = build_summary(cfg, load_job(cfg.dir))
        else:
            summary = run_job(cfg, resume=not args.no_resume)
        _print_leaderboard(summary)
        return

    if not args.config:
        parser.error("a config path is required (or use: harness job <job.yaml>)")

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg.eval.episodes = args.episodes
    run_eval(cfg)


if __name__ == "__main__":
    main()
