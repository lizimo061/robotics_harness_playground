"""Serve the harness PolicyAgent over HTTP (for RoboLab / policy-server benchmarks).

    DEEPSEEK_API_KEY=sk-... python examples/serve_robolab.py --port 8000

Run this in any process that has the harness deps (it does NOT need RoboLab).
The RoboLab side uses examples/robolab_inference_client.py to talk to it.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.config import LLMConfig
from harness.llm import get_llm
from harness.serving import PolicySessionManager, serve
from harness.types import ActionSpace


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the harness policy over HTTP")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--action-kind", default="joint_position", choices=["joint_position", "ee_delta"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--provider", default=None, help="deepseek | kimi | openai | claude | mock")
    args = parser.parse_args()

    provider = args.provider or ("deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "mock")
    llm = get_llm(LLMConfig(provider=provider, temperature=args.temperature))

    bound = np.pi if args.action_kind == "joint_position" else 0.1
    space = ActionSpace(
        kind=args.action_kind,
        dim=args.action_dim,
        low=-bound * np.ones(args.action_dim, dtype=np.float32),
        high=bound * np.ones(args.action_dim, dtype=np.float32),
        description=f"{args.action_kind} action of dim {args.action_dim}; last dim is the gripper (0 open, 1 closed).",
    )

    manager = PolicySessionManager(llm, action_space=space, action_dim=args.action_dim, temperature=args.temperature)
    serve(manager, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
