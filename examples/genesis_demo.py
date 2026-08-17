"""Run a Franka task in Genesis (genesis-world).

    python examples/genesis_demo.py --task pick_place                # real DeepSeek (tools mode)
    python examples/genesis_demo.py --task pick_place --show-viewer  # native Genesis GUI (real-time)
    python examples/genesis_demo.py --task pick_place --scripted --video demo.mp4   # offline, export video
    python examples/genesis_demo.py --task pick_place --viz run.html # HTML replay (animation + LLM trace)

Visualization options:
  --show-viewer  open the native Genesis interactive GUI
  --video PATH   record frames and export an mp4/gif video
  --viz PATH     write the harness HTML replay (animation + LLM trace)
  --scripted     run a deterministic pick-place (no LLM / API key needed)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.tasks import available_tasks, generate_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Franka task in Genesis")
    parser.add_argument("--task", default="pick_place", choices=available_tasks(3))
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--control-mode", default="ee_delta", choices=["ee_delta", "joint_position"])
    parser.add_argument("--mode", default="tools", choices=["json", "tools", "code", "plan", "skills"])
    parser.add_argument("--show-viewer", action="store_true", help="open the native Genesis GUI")
    parser.add_argument("--video", default=None, help="export an mp4/gif video to this path")
    parser.add_argument("--viz", default=None, help="write the HTML replay (animation + LLM trace) to this path")
    parser.add_argument("--scripted", action="store_true", help="run a deterministic pick-place (no LLM)")
    args = parser.parse_args()

    print("3D tasks:", available_tasks(3))
    spec = generate_task(args.task, seed=args.seed, difficulty=args.difficulty, dims=3)
    print("Task:", spec.description)
    print("objects:", spec.objects)
    print("goals:", spec.goals)

    try:
        from harness.envs.genesis import GenesisFrankaEnv

        env = GenesisFrankaEnv(
            task_spec=spec,
            backend=args.backend,
            control_mode=args.control_mode,
            show_viewer=args.show_viewer,
            record_video=args.video is not None,
            video_path=args.video or "video.mp4",
        )
    except ImportError:
        print("Genesis is not installed. Run: pip install genesis-world")
        print("(showing the generated 3D TaskSpec instead)")
        return

    try:
        if args.scripted:
            from harness.types import Action

            cube = env.get_object_pos("cube")
            goal = env.get_goal_pos("goal")
            env.step(Action(kind="ee_pose", value=cube.copy()))
            env.step(Action(kind="noop", gripper=1.0))
            env.step(Action(kind="ee_pose", value=goal.copy()))
            env.step(Action(kind="noop", gripper=0.0))
            print("scripted success:", env.is_success())
        else:
            key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
            if not key:
                print("No API key set. Use --scripted for an offline run, or set DEEPSEEK_API_KEY.")
                env.close()
                return

            from harness.agent import LLMController
            from harness.config import LLMConfig
            from harness.llm import get_llm
            from harness.viz.html import save_html
            from harness.viz.recorder import TraceRecorder

            provider = "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else "kimi"
            llm = get_llm(LLMConfig(provider=provider))
            recorder = TraceRecorder(capture_frames=True, metadata={"env": env.name, "mode": args.mode})
            ctrl = LLMController(llm, mode=args.mode, max_steps=80, task_description=spec.description, recorder=recorder)
            ep = ctrl.run(env)
            print(f"result: success={ep.success} steps={ep.steps} reward={ep.total_reward:.3f}")
            if args.viz and recorder.steps:
                save_html(recorder, args.viz, title=spec.description, fps=8)
                print("viz saved to", args.viz)
    finally:
        env.close()  # auto-saves the video when --video was set

    if args.video:
        print("video saved to", args.video)


if __name__ == "__main__":
    main()
