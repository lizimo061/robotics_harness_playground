"""Jobs: the cross product of tasks x agents x seeds, run concurrently.

``run_eval`` evaluates one (env, model) pair serially. A leaderboard needs the
whole grid, and the grid is where the wall-clock cost lives -- so this module
fans trials out, resumes an interrupted sweep, and can re-score a finished one
without paying for inference again.

Three properties are deliberate:

**A trial owns its environment.** Envs are stateful and not thread-safe, and a
shared one is exactly the bug that made episodes leak into each other. Each
trial constructs its own env and its own LLM client, so nothing mutable is
shared between workers.

**Concurrency is capped per provider as well as globally.** One provider's rate
limit should not be able to fail trials belonging to another.

**Only infrastructure is retried.** Retrying an agent timeout or a refusal
would launder a real failure into a pass, so the retry set is an explicit
allow-list of transport faults.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from harness.eval.infra import classify_infra_failure, summarize_infra
from harness.eval.lock import LockMismatch, RunLock
from harness.eval.metrics import oracle_steps_by_task, summarize_records
from harness.eval.results import (
    REPORTING_RULE,
    ResultsWriter,
    load_results,
    record_from_episode,
    write_per_instance_details,
)
from harness.types import Episode
from harness.utils.logging import get_logger

log = get_logger("harness.eval.job")

#: Exception type names worth retrying: transport faults only. An agent
#: timeout, a refusal, or a missing reward are *results*, not errors.
RETRYABLE = (
    "TransientLLMError",
    "TransientPolicyServerError",
    "ConnectError",
    "ReadTimeout",
    "ConnectTimeout",
    "PoolTimeout",
    "RemoteProtocolError",
)


@dataclass
class AgentSpec:
    """One column of the leaderboard: an agent, and the model behind it."""

    name: str = "llm_controller"        # llm_controller | oracle | null_agent
    model: str = ""                     # model id; blank for baselines
    provider: str = "mock"
    mode: str = "tools"
    max_steps: int = 20
    temperature: Optional[float] = 0.2
    max_tokens: int = 512
    extra: dict = field(default_factory=dict)
    #: cap on this agent's concurrent trials, e.g. to respect a rate limit
    concurrency: Optional[int] = None

    @property
    def id(self) -> str:
        if self.name in ("oracle", "oracle_agent"):
            return "oracle"
        if self.name in ("null", "null_agent", "nop", "noop"):
            return "null"
        return (self.model or self.provider or "model").replace("/", "_")

    def to_dict(self) -> dict:
        return {
            "name": self.name, "model": self.model, "provider": self.provider,
            "mode": self.mode, "max_steps": self.max_steps,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
            "extra": self.extra,
        }


@dataclass
class JobConfig:
    job_name: str = "job"
    log_dir: str = "logs"
    env_name: str = "tabletop"
    tasks: list = field(default_factory=lambda: ["pick_place"])
    agents: list = field(default_factory=list)
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    concurrency: int = 4
    max_retries: int = 2
    env_params: dict = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return Path(self.log_dir) / self.job_name

    def to_dict(self) -> dict:
        return {
            "job_name": self.job_name, "log_dir": self.log_dir,
            "env_name": self.env_name, "tasks": list(self.tasks),
            "agents": [a.to_dict() for a in self.agents], "seeds": list(self.seeds),
            "concurrency": self.concurrency, "max_retries": self.max_retries,
            "env_params": self.env_params,
        }


@dataclass(frozen=True)
class TrialKey:
    task: str
    agent_id: str
    seed: int

    def matches(self, record: dict) -> bool:
        return (
            str(record.get("env_name")) == self.task
            and str(record.get("policy")) == self.agent_id
            and record.get("seed") == self.seed
        )


def _build_env(cfg: JobConfig, task: str):
    from harness.config import EnvConfig
    from harness.envs.registry import get_env

    return get_env(EnvConfig(name=cfg.env_name, task=task, params=dict(cfg.env_params)))


def _build_agent(spec: AgentSpec):
    if spec.name.strip().lower() in ("oracle", "oracle_agent", "null", "null_agent", "nop", "noop"):
        from harness.agent.baselines import get_baseline_agent

        return get_baseline_agent(spec.name, max_steps=spec.max_steps, **spec.extra)

    from harness.agent.llm_controller import LLMController
    from harness.config import LLMConfig
    from harness.llm import get_llm

    llm = get_llm(LLMConfig(
        provider=spec.provider, model=spec.model,
        temperature=spec.temperature if spec.temperature is not None else 0.2,
        max_tokens=spec.max_tokens,
        extra=dict(spec.extra.get("llm_extra", {})),
    ))
    return LLMController(
        llm, mode=spec.mode, max_steps=spec.max_steps,
        temperature=spec.temperature,
        **{k: v for k, v in spec.extra.items() if k != "llm_extra"},
    )


def _run_trial(cfg: JobConfig, spec: AgentSpec, key: TrialKey) -> dict:
    """Run one cell. Builds its own env and agent -- nothing shared."""
    last_error: Optional[BaseException] = None
    for attempt in range(cfg.max_retries + 1):
        env = None
        t0 = time.time()
        try:
            env = _build_env(cfg, key.task)
            agent = _build_agent(spec)
            ep = agent.run(env, seed=key.seed)
            rec = record_from_episode(
                ep, env, policy=key.agent_id, seed=key.seed,
                episode_index=key.seed, mode=spec.mode,
                wall_clock_s=round(time.time() - t0, 3),
            )
            return rec.to_dict()
        except Exception as e:  # noqa: BLE001
            last_error = e
            name = type(e).__name__
            retryable = name in RETRYABLE or classify_infra_failure(f"{name}: {e}") is not None
            if retryable and attempt < cfg.max_retries:
                backoff = min(8.0, 0.5 * (2 ** attempt))
                log.warning("trial %s/%s seed=%s attempt %d failed (%s); retrying in %.1fs",
                            key.task, key.agent_id, key.seed, attempt + 1, name, backoff)
                time.sleep(backoff)
                continue
            break
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass

    # Record the failure rather than dropping the cell: an excluded trial
    # changes the denominator, which is the one thing the reporting rule
    # promises not to do.
    ep = Episode(metadata={"mode": spec.mode, "agent": spec.name})
    rec = record_from_episode(
        ep, _StubEnv(key.task), policy=key.agent_id, seed=key.seed,
        episode_index=key.seed, mode=spec.mode, error=last_error,
    )
    return rec.to_dict()


class _StubEnv:
    """Stand-in so a trial that died before env construction still records."""

    def __init__(self, task: str) -> None:
        self.name = task
        self.task = task

    def get_text_state(self) -> str:
        return ""

    def is_success(self) -> bool:
        return False


def run_job(cfg: JobConfig, *, resume: bool = True) -> dict:
    """Run the grid, writing one results file per agent. Returns the summary."""
    if not cfg.agents:
        raise ValueError("JobConfig.agents is empty: nothing to run")

    cfg.dir.mkdir(parents=True, exist_ok=True)
    lock_path = cfg.dir / "lock.json"
    lock = RunLock.capture(
        tasks=list(cfg.tasks),
        agents=[a.to_dict() for a in cfg.agents],
        seeds=list(cfg.seeds),
        episodes_per_cell=1,
        concurrency=cfg.concurrency,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    existing = RunLock.read(lock_path)
    if existing is not None and resume:
        if existing != lock:
            raise LockMismatch(
                "cannot resume: this run was configured differently.\n  "
                + "\n  ".join(lock.describe_mismatch(existing))
            )
    lock.write(lock_path)

    writers = {a.id: ResultsWriter(cfg.dir / a.id) for a in cfg.agents}
    for a in cfg.agents:
        writers[a.id].write_config({"job": cfg.to_dict(), "agent": a.to_dict()})

    done: list[dict] = []
    if resume:
        for a in cfg.agents:
            done += load_results(writers[a.id].dir)
    if done:
        log.info("resuming: %d trial(s) already recorded", len(done))

    pending: list[tuple[AgentSpec, TrialKey]] = []
    for spec in cfg.agents:
        for task in cfg.tasks:
            for seed in cfg.seeds:
                key = TrialKey(task=task, agent_id=spec.id, seed=int(seed))
                if any(key.matches(r) for r in done):
                    continue
                pending.append((spec, key))

    log.info("job %s: %d trial(s) to run (%d already done), concurrency=%d",
             cfg.job_name, len(pending), len(done), cfg.concurrency)

    # per-agent gates so one provider's rate limit cannot starve another
    gates = {
        a.id: threading.Semaphore(a.concurrency or cfg.concurrency)
        for a in cfg.agents
    }
    write_locks = {a.id: threading.Lock() for a in cfg.agents}
    records: list[dict] = list(done)

    def work(spec: AgentSpec, key: TrialKey) -> dict:
        with gates[key.agent_id]:
            rec = _run_trial(cfg, spec, key)
        with write_locks[key.agent_id]:
            writers[key.agent_id].append_dict(rec)
        return rec

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
            futures = {pool.submit(work, s, k): k for s, k in pending}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    records.append(fut.result())
                except Exception as e:  # noqa: BLE001 - should not happen; never lose the sweep
                    log.error("trial %s/%s seed=%s crashed the worker: %s",
                              key.task, key.agent_id, key.seed, e)

    return build_summary(cfg, records)


def build_summary(cfg: JobConfig, records: list[dict]) -> dict:
    """Aggregate records into the leaderboard view; recomputable from disk."""
    oracle = oracle_steps_by_task(records)
    summary = {
        "job_name": cfg.job_name,
        "job_dir": str(cfg.dir),
        "trials": len(records),
        # The *configured* grid, not the grid that happened to produce records.
        # A task whose every trial died must still appear downstream, otherwise
        # a report renders a complete-looking board with a column missing.
        "tasks": list(cfg.tasks),
        "seeds": list(cfg.seeds),
        "reporting_rule": REPORTING_RULE,
        "oracle_steps": oracle or None,
        "leaderboard": summarize_records(records, oracle_steps=oracle),
    }
    infra = summarize_infra(records)
    if infra:
        summary["infra_failures"] = infra
    for agent_id in {str(r.get("policy")) for r in records}:
        rows = [r for r in records if str(r.get("policy")) == agent_id]
        write_per_instance_details(cfg.dir / agent_id, rows)
    return summary


def load_job(job_dir) -> list[dict]:
    """Read every agent's results back from a job directory."""
    d = Path(job_dir)
    out: list[dict] = []
    for sub in sorted(p for p in d.iterdir() if p.is_dir()):
        out += load_results(sub)
    return out


def regrade(
    job_dir,
    scorer: Callable[[dict], dict],
    *,
    out_dir=None,
) -> dict:
    """Re-score a finished job with a new grader, without re-running inference.

    ``scorer`` receives a trial record and returns the fields to overwrite
    (typically ``success`` and ``score``). Source records are never modified:
    the regraded copy is written elsewhere, so a grader change can always be
    compared against the original rather than replacing it.

    At frontier-model prices a full sweep is the expensive part and the grader
    is the part most likely to be wrong, so re-scoring has to be possible
    without paying for the sweep twice.
    """
    src = Path(job_dir)
    rows = load_job(src)
    if not rows:
        raise FileNotFoundError(f"no results under {src}")

    dst = Path(out_dir) if out_dir else src.parent / f"{src.name}__regraded"
    changed = up = down = 0
    per_agent: dict[str, ResultsWriter] = {}
    new_rows: list[dict] = []

    for r in rows:
        before = bool(r.get("success"))
        patch = scorer(dict(r)) or {}
        merged = {**r, **patch}
        merged["regraded"] = True
        after = bool(merged.get("success"))
        if after != before:
            changed += 1
            up += int(after and not before)
            down += int(before and not after)
        agent_id = str(merged.get("policy") or "unknown")
        per_agent.setdefault(agent_id, ResultsWriter(dst / agent_id)).append_dict(merged)
        new_rows.append(merged)

    mean_before = sum(1 for r in rows if r.get("success")) / len(rows)
    mean_after = sum(1 for r in new_rows if r.get("success")) / len(new_rows)
    log.info("regrade delta over %d trial(s): %d changed (%d up, %d down), "
             "success %.3f -> %.3f", len(rows), changed, up, down, mean_before, mean_after)

    return {
        "source": str(src),
        "regraded_dir": str(dst),
        "trials": len(rows),
        "changed": changed,
        "up": up,
        "down": down,
        "success_before": round(mean_before, 4),
        "success_after": round(mean_after, 4),
        "leaderboard": summarize_records(new_rows, oracle_steps=oracle_steps_by_task(new_rows)),
    }
