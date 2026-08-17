# eval (harness/eval/)
> Purpose: evaluation metrics + JSONL trajectory logging.
> Read when: benchmarking, or changing what gets logged.
> Key files: metrics.py, logger.py

## Public API

- summarize(episodes) -> {episodes, success_rate, mean_steps, mean_reward}
- EpisodeMetrics / episode_metrics(ep)
- TrajectoryLogger(log_dir, run_name).log(ep) / log_config(cfg)
- episode_to_dict(ep)

## Notes

- Trajectories append to logs/<run_name>.jsonl; the config is saved alongside.
- harness.runner.run_eval ties config -> env + llm + agent -> these metrics.

## Related
- modules/viz.md, modules/config.md, concepts.md (Episode).
