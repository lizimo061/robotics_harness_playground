# eval (harness/eval/)
> Purpose: run the tasks x agents x seeds grid, score it defensibly, publish it.
> Read when: benchmarking models, changing what gets logged, or reading a board.
> Key files: job.py, results.py, stats.py, metrics.py, infra.py, lock.py, report.py

## Public API

**Running a grid**
- `JobConfig` / `AgentSpec` — the declared grid; `cfg.dir` is the job directory
- `run_job(cfg, resume=True)` -> summary dict (per-trial env + per-trial LLM)
- `load_job(job_dir)` -> flat record list; `build_summary(cfg, records)` -> summary
- `regrade(job_dir, scorer)` — re-score existing records into a marked copy;
  never mutates the source
- `RunLock.capture(...)` / `LockMismatch` — a resume may not silently change
  the task set, agent set, seeds, or code that produced the numbers

**Scoring**
- `summarize(episodes)`; `summarize_records(records, reliability_k, oracle_steps)`
- `oracle_steps_by_task(records)` — the efficiency denominator
- `beta_ci` / `wilson_ci`, `pass_at_k`, `pass_hat_k`, `aggregate_over_tasks`,
  `mcnemar`, `rank_interval`, `resolution_ratio`
- `TrialRecord` / `record_from_episode`, `classify_failure`, `FailureMode`
- `classify_infra_failure` / `summarize_infra`
- `ResultsWriter`, `load_results`, `write_per_instance_details`, `REPORTING_RULE`

**Publishing**
- `render_report(summary, records, title="")` -> HTML string
- `write_report(job_dir, out=None, title="")` -> path

## Notes

- **Baselines are not decoration.** The oracle proves each task is solvable and
  supplies the step denominator; the null agent proves the success check is not
  vacuous. A task the oracle fails, or null passes, makes every model's score on
  it uninterpretable, so the report raises an audit callout instead of ranking it
  quietly. A configured task that produced no records at all is also called out —
  otherwise the grid looks complete with a column missing.
- **Ranks are ranges.** `rank_interval` ties agents whose intervals overlap.
  Printing 1, 2, 3 over overlapping intervals invents an ordering the sample size
  does not support. Overlap is still not a significance test — use `mcnemar` on
  the paired task set for that.
- **Intervals are one-sided at the boundaries.** An equal-tailed Beta interval at
  0/30 returns roughly [0.1%, 11.2%], which excludes its own point estimate;
  spending all of alpha on the tail where the uncertainty lives gives [0, 9.2%].
- **The reporting rule is published with the numbers.** Whether errored trials
  count or are excluded moves reported gaps by whole percentage points, so
  `REPORTING_RULE` ships inside the summary and inside the HTML.
- **Infra failures are reported separately and never change the denominator.**
  Two tiers: clearly environmental, and ambiguous.
- Trials are keyed `(task, agent_id, seed)`; a resume completes the grid rather
  than re-running it. Every agent writes its own append-only
  `episode_results.jsonl`, so a summary is always recomputable from disk
  (`harness job <cfg> --summary-only`).
- Envs are stateful, so every trial constructs its own. Sharing one is the
  episode-state leak that silently made solved tasks stay solved.

## CLI

```
harness job configs/leaderboard_tabletop.yaml              # run the grid
harness job configs/leaderboard_tabletop.yaml --report     # ...and write HTML
harness job configs/leaderboard_tabletop.yaml --summary-only
harness report logs/tabletop-v1 -o board.html
```

## The report

Self-contained HTML: no network requests, so it renders from `file://` and
survives being emailed. KPI tiles, per-agent success bars with the 95% interval
drawn beneath each bar, and a task x agent grid.

Constraints in `report.py` that are correctness rather than taste:

- Every heatmap cell prints its `n/N`, so colour is never the only encoding and a
  genuine zero is distinguishable from no-data.
- Cell ink is chosen per ramp step against the fill behind it, not inherited from
  the page. `tests/test_report.py` recomputes the contrast of every pair and
  fails below 4.5:1 — white is *not* always right on a saturated fill.
- Baselines are marked by form (a `reference` chip, a muted bar), never by a
  status hue, which would imply good/bad about a reference row.
- Both theme states are defined at token level, including the un-stamped
  system-default one.

## Related
- modules/viz.md, modules/config.md, modules/agent.md (baselines), concepts.md
