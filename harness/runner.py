"""High-level runner: config -> environment + LLM + agent -> evaluation."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from harness.agent.llm_controller import LLMController
from harness.config import HarnessConfig, VizConfig
from harness.envs.registry import get_env
from harness.eval.logger import TrajectoryLogger
from harness.eval.metrics import summarize, summarize_records
from harness.eval.results import ResultsWriter, record_from_episode
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


def run_eval(cfg: HarnessConfig) -> dict:
    """Run the configured episodes and return a summary dict."""
    set_seed(cfg.seed)

    llm = get_llm(cfg.llm)
    env = get_env(cfg.env)

    viz_enabled = cfg.viz.enabled and cfg.viz.backend != "none"
    recorder = TraceRecorder(
        capture_frames=viz_enabled and cfg.viz.capture_frames,
        metadata={"llm": llm.name, "env": env.name, "mode": cfg.agent.mode, "task": cfg.env.task},
    ) if viz_enabled else None
    viewer = _make_live_viewer(cfg.viz) if viz_enabled else None

    agent = LLMController(
        llm,
        mode=cfg.agent.mode,
        max_steps=cfg.agent.max_steps,
        use_vision=cfg.agent.use_vision,
        system_prompt=cfg.agent.system_prompt,
        temperature=cfg.agent.temperature,
        task_description=cfg.env.task,
        recorder=recorder,
        on_step=viewer.on_step if viewer is not None else None,
        **cfg.agent.extra,
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
    try:
        for i in range(cfg.eval.episodes):
            seed = cfg.seed + i
            t0 = time.time()
            err: Optional[BaseException] = None
            try:
                ep = agent.run(env, seed=seed)
            except Exception as e:  # noqa: BLE001 - one bad episode must not kill the sweep
                err = e
                ep = Episode(metadata={"mode": cfg.agent.mode, "llm": llm.name, "env": env.name})
                log.warning("episode %d raised %s: %s", i, type(e).__name__, e)
            episodes.append(ep)

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
    if records:
        summary["leaderboard"] = summarize_records(records)
    if cfg.eval.verbose:
        log.info("summary: success_rate=%.3f ci95=%s width=%.1fpp cost=%s",
                 summary.get("success_rate", 0.0), summary.get("success_ci_95"),
                 summary.get("ci_width_pp", 0.0), summary.get("cost_usd"))
        log.info("results: %s", writer.path)
    return summary
