"""Run a task and write a synced animation + LLM-trace viewer.

This uses the html backend by default (works anywhere). Open the printed path
in a browser to watch the animation replay in sync with the LLM trace.

    python examples/visualize.py                       # offline toy demo
    python examples/visualize.py configs/deepseek_pick_place.yaml

To watch live in the terminal instead, set viz.backend to "console" in the
config, or pass --console here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.config import load_config
from harness.runner import run_eval


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/toy_pick_place.yaml"
    console = "--console" in sys.argv

    cfg = load_config(cfg_path)
    cfg.viz.enabled = True
    if console:
        cfg.viz.backend = "console"
    elif cfg.viz.backend not in ("html", "live"):
        cfg.viz.backend = "html"

    summary = run_eval(cfg)
    print("summary:", summary)
    if cfg.viz.backend == "html":
        print("open:", os.path.abspath(cfg.viz.output))


if __name__ == "__main__":
    main()
