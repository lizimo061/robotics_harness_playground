"""High-level runner: config -> environment + LLM + agent -> evaluation."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from harness.agent.llm_controller import LLMController, is_offline_llm, model_id_of
from harness.config import HarnessConfig, VizConfig
from harness.envs.registry import get_env
from harness.eval.logger import TrajectoryLogger
from harness.eval.metrics import summarize, summarize_records
from harness.eval.infra import summarize_infra
from harness.eval.results import (
    REPORTING_RULE,
    ResultsWriter,
    record_from_episode,
    write_per_instance_details,
)
from harness.llm.capabilities import check_vision_config
from harness.llm.registry import get_llm
from harness.types import Episode
from harness.utils.logging import get_logger
from harness.utils.seeds import set_seed
from harness.viz.html import save_html
from harness.viz.recorder import TraceRecorder

log = get_logger("harness.runner")


def _make_live_viewer(viz: VizConfig):
    """Build an optional live on_step viewer for the requested backend."""
    if viz.backend == "console":
        from harness.viz.live import ConsoleTracer

        return ConsoleTracer()
    if viz.backend == "live":
        try:
            from harness.viz.live import MatplotlibViewer

            return MatplotlibViewer(fps=viz.fps, title=viz.title or "robot view")
        except Exception as e:  # no display / no matplotlib
            log.warning("matplotlib live viewer unavailable (%s); falling back to html", e)
            return None
    return None


def _write_episode_video(frames: list, cfg, index: int, seed: int,
                         success: Optional[bool]) -> Optional[Path]:
    """Write one episode's frames to `<stem>_ep<i>_seed<n>_<outcome><ext>`.

    Naming the outcome into the filename is the point: hunting for the failure
    case in a directory of look-alike videos otherwise means opening each one.
    """
    from harness.viz.video import write_video

    if not frames:
        log.warning("episode %d captured no frames; is the env's render() returning None?", index)
        return None
    outcome = "success" if success else "fail"
    base = Path(cfg.viz.video)
    out = base.with_name(f"{base.stem}_ep{index}_seed{seed}_{outcome}{base.suffix or '.mp4'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_video(frames, out, fps=cfg.viz.fps)
    except Exception as e:  # noqa: BLE001 - a missing codec must not lose the run
        log.warning("could not write %s (%s: %s)", out, type(e).__name__, e)
        return None
    log.info("video: %s (%d frames, %s)", out, len(frames), outcome)
    return out


def _check_vision_config(cfg: HarnessConfig, llm) -> None:
    """Refuse a config that needs pixels but names a model that cannot see them.

    The controller checks itself too, but only against what it is handed, and
    `agent.tier` is a config field the controller does not receive here -- so a
    `tier: perception` run against deepseek-chat would otherwise sail straight
    past the gate and report a success rate for an agent whose only route to an
    object's location (detect/point_at over the frame) it could not use.

    Escape hatches, in order: the mock provider (offline, scripted), and
    `agent.extra.allow_blind_vision: true` for a deliberate blind run.
    """
    tier = str(cfg.agent.extra.get("tier", cfg.agent.tier) or "privileged")
    if not (cfg.agent.use_vision or tier != "privileged"):
        return
    if bool(cfg.agent.extra.get("allow_blind_vision")) or is_offline_llm(llm):
        return
    # Prefer what the client will actually send: cfg.llm.model is often blank and
    # the provider fills in its own default (deepseek -> deepseek-chat).
    model = model_id_of(llm) or cfg.llm.model
    if not model:
        log.warning("cannot verify image support: provider %r reports no model name, "
                    "so this vision run is unchecked", cfg.llm.provider)
        return
    check_vision_config(model, use_vision=cfg.agent.use_vision, tier=tier)


def run_eval(cfg: HarnessConfig) -> dict:
    """Run the configured episodes and return a summary dict."""
    set_seed(cfg.seed)

    llm = get_llm(cfg.llm)
    env = get_env(cfg.env)

    # Capture at the env boundary, not once per LLM turn: a single tool call can
    # drive many env steps, and a per-turn video skips exactly the motion a
    # reviewer wants to see.
    capture = None
    if cfg.viz.enabled and cfg.viz.video and cfg.viz.capture_frames:
        from harness.viz.capture import FrameCapture

        capture = FrameCapture(env, every=cfg.viz.capture_every)
        env = capture

    viz_enabled = cfg.viz.enabled and cfg.viz.backend != "none"
    recorder = TraceRecorder(
        capture_frames=viz_enabled and cfg.viz.capture_frames,
        metadata={"llm": llm.name, "env": env.name, "mode": cfg.agent.mode, "task": cfg.env.task},
    ) if viz_enabled else None
    viewer = _make_live_viewer(cfg.viz) if viz_enabled else None

    # Reference baselines bypass the LLM entirely: the oracle proves a task is
    # solvable (and supplies the efficiency denominator), the null agent proves
    # the success check is not vacuous.
    baseline = cfg.agent.name.strip().lower()
    if baseline in ("oracle", "oracle_agent", "null", "null_agent", "nop", "noop"):
        from harness.agent.baselines import get_baseline_agent

        agent = get_baseline_agent(baseline, max_steps=cfg.agent.max_steps, **cfg.agent.extra)
        model_id = "oracle" if baseline.startswith("oracle") else "null"
    else:
        # Only the LLM path can be fooled by a blind model; the baselines above
        # never look at a frame.
        _check_vision_config(cfg, llm)
        # The scaffolding tier and its detector have to be forwarded explicitly.
        # An earlier attempt to wire these went in as a string patch whose pattern
        # never matched, so it silently changed nothing: `tier: perception` in a
        # YAML reached AgentConfig, was validated, and was then dropped on the
        # floor here, leaving the agent with the privileged toolset while the run
        # was labelled as perception. Assert the plumbing instead of assuming it --
        # tests/test_perception_tier.py covers the switch, and the runner-level
        # test in tests/test_vision_gating.py covers this hand-off.
        tier = str(cfg.agent.extra.get("tier", cfg.agent.tier) or "privileged")
        detector = None
        if tier != "privileged":
            from harness.perception.detect import get_detector

            detector = get_detector(cfg.agent.detector or "oracle", env)
        extra = {k: v for k, v in cfg.agent.extra.items()
                 if k not in ("tier", "detector")}
        agent = LLMController(
            llm,
            mode=cfg.agent.mode,
            max_steps=cfg.agent.max_steps,
            use_vision=cfg.agent.use_vision,
            tier=tier,
            detector=detector,
            system_prompt=cfg.agent.system_prompt,
            temperature=cfg.agent.temperature,
            task_description=cfg.env.task,
            recorder=recorder,
            on_step=viewer.on_step if viewer is not None else None,
            **extra,
        )
        # One directory per (model, env) so two models under comparison cannot
        # append into the same file -- run_name=cfg.env.name collided by design.
        model_id = (cfg.llm.model or cfg.llm.provider or "model").replace("/", "_")

    run_name = f"{cfg.env.name.replace(':', '_')}__{model_id}"
    logger = TrajectoryLogger(log_dir=cfg.eval.log_dir, run_name=run_name)
    writer = ResultsWriter(Path(cfg.eval.log_dir) / run_name)
    writer.write_config(cfg.to_dict())
    # must match the record's env_name, which is how readers find this file
    _task_id = getattr(env, "task", None) or getattr(getattr(env, "task_spec", None), "kind", None) or env.name
    writer.write_task_config(str(_task_id), cfg.to_dict())
    if cfg.eval.save_trajectories:
        logger.log_config(cfg.to_dict())

    episodes: list[Episode] = []
    records: list[dict] = []
    ep_start = 0
    try:
        for i in range(cfg.eval.episodes):
            seed = cfg.seed + i
            if capture is not None:
                capture.clear()
            t0 = time.time()
            err: Optional[BaseException] = None
            try:
                ep = agent.run(env, seed=seed)
            except Exception as e:  # noqa: BLE001 - one bad episode must not kill the sweep
                err = e
                ep = Episode(metadata={"mode": cfg.agent.mode, "llm": llm.name, "env": env.name})
                log.warning("episode %d raised %s: %s", i, type(e).__name__, e)
            episodes.append(ep)

            # Write this episode's frames before the next one overwrites them:
            # both the capture wrapper and the recorder accumulate, so a single
            # write at the end would splice every episode into one video.
            if cfg.viz.video:
                frames = (capture.frames if capture is not None
                          else [s.frame for s in recorder.steps[ep_start:]
                                if s.frame is not None] if recorder is not None else [])
                _write_episode_video(list(frames), cfg, i, seed, ep.success)
                if recorder is not None:
                    ep_start = len(recorder.steps)

            rec = record_from_episode(
                ep, env,
                policy=model_id, seed=seed, episode_index=i,
                mode=cfg.agent.mode, wall_clock_s=round(time.time() - t0, 3),
                error=err,
            )
            writer.append(rec)
            records.append(rec.to_dict())

            if cfg.eval.save_trajectories:
                logger.log(ep)
            if cfg.eval.verbose:
                log.info(
                    "episode %d: success=%s steps=%d reward=%.3f mode=%s cost=%s",
                    i, ep.success, ep.steps, ep.total_reward, rec.failure_mode, rec.cost_usd,
                )
    finally:
        env.close()
        if viewer is not None and hasattr(viewer, "close"):
            viewer.close()

    if recorder is not None and recorder.steps:
        title = cfg.viz.title or f"{env.name} - {cfg.agent.mode} ({llm.name})"
        if cfg.viz.backend in ("html", "live"):
            path = save_html(recorder, cfg.viz.output, title=title, fps=cfg.viz.fps)
            log.info("visualization saved to %s (open in a browser)", path)

    summary = summarize(episodes)
    summary["run_dir"] = str(writer.dir)
    summary["results_file"] = str(writer.path)
    summary["reporting_rule"] = REPORTING_RULE
    if records:
        summary["leaderboard"] = summarize_records(records)
        infra = summarize_infra(records)
        if infra:
            summary["infra_failures"] = infra
        summary["per_instance_details"] = str(write_per_instance_details(writer.dir, records))
    if cfg.eval.verbose:
        log.info("summary: success_rate=%.3f ci95=%s width=%.1fpp cost=%s",
                 summary.get("success_rate", 0.0), summary.get("success_ci_95"),
                 summary.get("ci_width_pp", 0.0), summary.get("cost_usd"))
        log.info("results: %s", writer.path)
    return summary
