"""Run a task from a YAML/JSON config file.

    python examples/run_task.py configs/deepseek_pick_place.yaml
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.config import load_config
from harness.runner import run_eval


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python examples/run_task.py <config.yaml> [episodes]")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    if len(sys.argv) > 2:
        cfg.eval.episodes = int(sys.argv[2])
    summary = run_eval(cfg)
    print("summary:", summary)


if __name__ == "__main__":
    main()
