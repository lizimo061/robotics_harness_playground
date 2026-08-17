"""Command-line entry point: run a task from a YAML/JSON config."""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from harness.config import load_config
from harness.runner import run_eval


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run a robotics harness task.")
    parser.add_argument("config", help="path to a YAML or JSON config file")
    parser.add_argument("--episodes", type=int, default=None, help="override the episode count")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg.eval.episodes = args.episodes
    run_eval(cfg)


if __name__ == "__main__":
    main()
