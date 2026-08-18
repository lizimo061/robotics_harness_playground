"""Evaluate an LLM across several RoboLab tasks, with video and a report.

Run INSIDE the RoboLab venv (Linux + CUDA). Isaac Sim's AppLauncher must be
created BEFORE importing RoboLab/IsaacLab, so this follows RoboLab's own pattern.

    OMNI_KIT_ACCEPT_EULA=Y python examples/eval_robolab.py \
        --tasks RubiksCubeTask,BananaInBowlTask --episodes 2 --headless \
        --out runs/robolab-live

Why one process for every task: Isaac Sim startup dominates the wall clock
(minutes), so paying it once and looping tasks in-process is the difference
between a usable sweep and an overnight one. Each task's env is created and
closed in turn; if one fails to build, the sweep records that and continues
rather than losing the tasks behind it.

Output layout matches a harness job, so the existing tooling just works:

    <out>/<model>/episode_results.jsonl     -> harness report <out>
    <out>/videos/<task>_ep<i>_<outcome>.mp4
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Isaac Sim AppLauncher must run BEFORE importing robolab ---
import cv2  # noqa: F401,E402  (required before isaaclab)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Evaluate an LLM across RoboLab tasks")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--tasks", default="RubiksCubeTask",
                    help="comma-separated RoboLab task class names")
parser.add_argument("--episodes", type=int, default=2, help="episodes per task")
parser.add_argument("--mode", default="tools", choices=["json", "tools", "code", "plan", "skills"])
parser.add_argument("--max-steps", type=int, default=40, help="LLM turn budget per episode")
# The mode selects RoboLab's action flavour, so the list must match the adapter's
# registrar table -- omitting one here silently makes it unreachable.
parser.add_argument("--action-mode", default="ee_pose",
                    choices=["ee_pose", "ee_delta", "joint_position"],
                    help="ee_pose = absolute EE pose via IK (what move_to means); "
                         "ee_delta = relative IK; joint_position = raw joint targets")
parser.add_argument("--provider", default="deepseek")
parser.add_argument("--model", default="deepseek-chat")
parser.add_argument("--out", default="runs/robolab-live", help="output directory")
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--every", type=int, default=2, help="capture every Nth env step")
parser.add_argument("--scripted", action="store_true",
                    help="deterministic no-op loop instead of an LLM (no API key needed)")
parser.add_argument("--with-null", action="store_true",
                    help="also run the null baseline on each task, in the SAME Isaac "
                         "process. A task the null agent passes has a vacuous success "
                         "check, and Isaac startup is too expensive to pay twice for it.")
parser.add_argument("--null-episodes", type=int, default=1)
parser.add_argument("--scripted-map", default="",
                    help="solvability probe, as 'Task:source>target,Task2:src>dst'. "
                         "The probe is TOLD what to move where, which is the point: a "
                         "task it cannot solve is not evidence about any model, and "
                         "RoboLab states goals in language so a generic oracle has "
                         "nothing to aim at.")
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True  # required for camera observations
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pathlib import Path  # noqa: E402

from harness.agent import LLMController  # noqa: E402
from harness.agent.baselines import get_baseline_agent  # noqa: E402
from harness.config import LLMConfig  # noqa: E402
from harness.envs.robolab import RoboLabEnv  # noqa: E402
from harness.eval.results import ResultsWriter, record_from_episode  # noqa: E402
from harness.llm import get_llm  # noqa: E402
from harness.viz.capture import FrameCapture  # noqa: E402
from harness.viz.video import write_video  # noqa: E402


def _run_episode(env, raw, task, agent_id, agent_writer, i, kind, *,
                 llm, instruction, vid_dir, move=None) -> list:
    """Run one episode, append its record, write its video. Returns manifest rows."""
    env.clear()
    t0 = time.time()
    err = None
    if kind == "scripted" and move:
        agent = get_baseline_agent("scripted_pick_place", source=move[0],
                                   target=move[1], max_steps=260)
    elif kind == "null" or args_cli.scripted:
        agent = get_baseline_agent("null", max_steps=args_cli.max_steps)
    else:
        agent = LLMController(llm, mode=args_cli.mode, max_steps=args_cli.max_steps,
                              task_description=instruction)
    try:
        ep = agent.run(env, seed=i)
    except Exception as e:  # noqa: BLE001 - one bad episode must not end the sweep
        err = e
        print(f"!! episode {i} raised {type(e).__name__}: {e}", flush=True)
        from harness.types import Episode
        ep = Episode(metadata={"mode": args_cli.mode, "env": task})

    rec = record_from_episode(
        ep, raw, policy=agent_id, seed=i, episode_index=i, mode=args_cli.mode,
        wall_clock_s=round(time.time() - t0, 3), error=err)
    d = rec.to_dict()
    d["env_name"] = task  # so the report's per-task grid keys on the RoboLab task
    agent_writer.append_dict(d)

    outcome = "success" if ep.success else "fail"
    print(f"  [{agent_id}] ep{i}: success={ep.success} steps={ep.steps} "
          f"frames={len(env.frames)} {time.time() - t0:.0f}s", flush=True)
    if ep.metadata.get("error"):
        print(f"  !! {ep.metadata['error']}", flush=True)
    if not env.frames:
        print("  !! no frames captured (camera off, or render() is None)", flush=True)
        return []
    stem = (f"{task}_ep{i}_{outcome}" if kind == "llm"
            else f"{task}_{kind}_{outcome}")
    path = vid_dir / f"{stem}.mp4"
    try:
        write_video(env.frames, path, fps=args_cli.fps)
    except Exception as e:  # noqa: BLE001 - a missing codec must not lose the run
        print(f"  !! video failed: {type(e).__name__}: {e}", flush=True)
        return []
    print(f"  video: {path}", flush=True)
    return [{"task": task, "agent": agent_id, "episode": i,
             "success": bool(ep.success), "steps": ep.steps, "video": str(path)}]


def main() -> None:
    tasks = [t.strip() for t in args_cli.tasks.split(",") if t.strip()]
    out = Path(args_cli.out)
    vid_dir = out / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    model_id = "scripted" if args_cli.scripted else args_cli.model.replace("/", "_")
    writer = ResultsWriter(out / model_id)
    writer.write_config(vars(args_cli) | {"tasks": tasks})
    null_writer = ResultsWriter(out / "null") if args_cli.with_null else None

    # "Task:source>target,..." -> {task: (source, target)}
    scripted_map = {}
    for entry in (e.strip() for e in args_cli.scripted_map.split(",") if e.strip()):
        task_part, _, move = entry.partition(":")
        source, _, target = move.partition(">")
        if task_part and source and target:
            scripted_map[task_part.strip()] = (source.strip(), target.strip())
    scripted_writer = ResultsWriter(out / "scripted") if scripted_map else None

    llm = None if args_cli.scripted else get_llm(
        LLMConfig(provider=args_cli.provider, model=args_cli.model, max_tokens=1024))

    manifest, failures = [], []
    for task in tasks:
        print(f"\n=== {task} ", flush=True)
        try:
            raw = RoboLabEnv(task, action_mode=args_cli.action_mode, device=args_cli.device)
        except Exception as e:  # noqa: BLE001 - one bad task must not end the sweep
            print(f"!! could not build {task}: {type(e).__name__}: {e}", flush=True)
            failures.append({"task": task, "error": f"{type(e).__name__}: {e}"})
            continue

        # skip_blank: Isaac's first frame after a reset is solid black until the
        # renderer settles, which otherwise opens every video on a blank frame.
        env = FrameCapture(raw, every=args_cli.every, skip_blank=True)
        instruction = env.get_text_state()
        print(f"instruction: {instruction[:160]}", flush=True)
        # Perception check: without scene objects a text-only model is blind and
        # any score it gets is meaningless, so say so loudly before spending the
        # episodes.
        objs = env.list_objects()
        print(f"scene objects ({len(objs)}): {objs}", flush=True)
        if not objs:
            print("!! no scene objects exposed -- a text model cannot locate anything",
                  flush=True)
        else:
            print("state given to the agent:\n    "
                  + env.get_text_state().replace("\n", "\n    "), flush=True)
        # (agent_id, writer, episodes, is_null) -- the null baseline shares this
        # task's env, so it costs no extra Isaac startup.
        plans = [(model_id, writer, args_cli.episodes, "llm")]
        if null_writer is not None:
            plans.append(("null", null_writer, args_cli.null_episodes, "null"))
        if scripted_writer is not None and task in scripted_map:
            plans.append(("scripted", scripted_writer, 1, "scripted"))

        try:
            for agent_id, agent_writer, n_eps, kind in plans:
                for i in range(n_eps):
                    manifest.extend(_run_episode(
                        env, raw, task, agent_id, agent_writer, i, kind,
                        llm=llm, instruction=instruction, vid_dir=vid_dir,
                        move=scripted_map.get(task)))
        finally:
            env.close()

    (out / "video_manifest.json").write_text(
        json.dumps({"videos": manifest, "build_failures": failures}, indent=2))
    print(f"\nresults: {writer.path}\nvideos:  {vid_dir}", flush=True)
    if failures:
        print(f"tasks that would not build: {[f['task'] for f in failures]}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
