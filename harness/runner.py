"""High-level runner: config -> environment + LLM + agent -> evaluation."""
from __future__ import annotations

from harness.agent.llm_controller import LLMController
from harness.config import HarnessConfig, VizConfig
from harness.envs.registry import get_env
from harness.eval.logger import TrajectoryLogger
from harness.eval.metrics import summarize
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

    logger = TrajectoryLogger(log_dir=cfg.eval.log_dir, run_name=cfg.env.name)
    if cfg.eval.save_trajectories:
        logger.log_config(cfg.to_dict())

    episodes: list[Episode] = []
    try:
        for i in range(cfg.eval.episodes):
            ep = agent.run(env, seed=cfg.seed + i)
            episodes.append(ep)
            if cfg.eval.save_trajectories:
                logger.log(ep)
            if cfg.eval.verbose:
                log.info("episode %d: success=%s steps=%d reward=%.3f", i, ep.success, ep.steps, ep.total_reward)
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
    if cfg.eval.verbose:
        log.info("summary: %s", summary)
    return summary
