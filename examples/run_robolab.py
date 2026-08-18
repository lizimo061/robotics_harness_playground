"""Run a RoboLab task with the harness agent (in-process).

Run INSIDE the RoboLab venv (Linux + CUDA). The Isaac Sim AppLauncher must be
created BEFORE importing RoboLab/IsaacLab, so this script follows RoboLab's own
example pattern.

    python examples/run_robolab.py --task RubiksCubeTask --headless          # real DeepSeek (tools mode)
    python examples/run_robolab.py --task BananaInBowlTask --mode tools --headless
    python examples/run_robolab.py --task RubiksCubeTask --scripted --headless   # offline, no API key

Hierarchical mode (policy-as-tool): the LLM plans and a trained VLA executes.
Point --policy-url at a policy server speaking the harness protocol (see
examples/serve_robolab.py, or wrap pi0.5/GR00T behind it):

    python examples/run_robolab.py --task BananaInBowlTask --headless \
        --policy-url http://localhost:8000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Isaac Sim AppLauncher must run BEFORE importing robolab ---
import cv2  # noqa: F401  (required before isaaclab)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a RoboLab task with the harness agent")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", default="RubiksCubeTask", help="RoboLab task class name")
parser.add_argument("--mode", default="tools", choices=["json", "tools", "code", "plan", "skills"])
parser.add_argument("--max-steps", type=int, default=100)
parser.add_argument("--scripted", action="store_true", help="run a deterministic no-op loop (no LLM)")
parser.add_argument("--action-mode", default="ee_delta", choices=["ee_delta", "joint_position"])
parser.add_argument(
    "--policy-url",
    default="",
    help="policy-server URL; enables hierarchical policy-as-tool mode (LLM plans, policy executes)",
)
parser.add_argument("--policy-steps", type=int, default=60, help="default env steps per run_policy call")
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from harness.envs.robolab import RoboLabEnv  # noqa: E402


def main() -> None:
    env = RoboLabEnv(args_cli.task, action_mode=args_cli.action_mode, device=args_cli.device)
    print("task:", env.task)
    print("instruction:", env.get_text_state())
    print("action space:", env.action_space.to_text())

    try:
        if args_cli.scripted:
            import numpy as np

            from harness.types import Action

            env.reset()
            for i in range(args_cli.max_steps):
                a = np.zeros(env.action_space.dim, dtype=np.float32)
                a[-1] = 1.0 if i % 2 == 0 else 0.0  # toggle gripper (sanity check)
                r = env.step(Action(kind=env.action_space.kind, value=a))
                if r.success:
                    print("scripted success at step", i)
                    break
        else:
            key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
            if not key:
                print("Set DEEPSEEK_API_KEY (or MOONSHOT_API_KEY), or use --scripted.")
                return

            from harness.agent import LLMController
            from harness.config import LLMConfig
            from harness.llm import get_llm
            from harness.viz.html import save_html
            from harness.viz.recorder import TraceRecorder

            provider = "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "kimi"
            llm = get_llm(LLMConfig(provider=provider))
            recorder = TraceRecorder(capture_frames=False, metadata={"env": env.name, "mode": args_cli.mode})

            policy = None
            if args_cli.policy_url:
                # hierarchical: the LLM delegates motion to the policy via run_policy
                policy = {
                    "type": "remote",
                    "base_url": args_cli.policy_url,
                    "action_dim": env.action_space.dim,
                }
                print(f"policy-as-tool enabled -> {args_cli.policy_url}")

            ctrl = LLMController(
                llm, mode=args_cli.mode, max_steps=args_cli.max_steps,
                task_description=env.get_text_state(), recorder=recorder,
                policy=policy,
                policy_options={"default_steps": args_cli.policy_steps},
            )
            ep = ctrl.run(env)
            print(f"result: success={ep.success} steps={ep.steps} reward={ep.total_reward:.3f}")
            save_html(recorder, "logs/robolab.html", title=env.get_text_state(), fps=8)
            print("LLM trace saved to logs/robolab.html")
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
