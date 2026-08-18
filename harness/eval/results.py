"""Per-episode results: the reward dict and the on-disk record.

Two design choices, both taken from what Harbor and RoboLab learned:

1. **The task emits the score, not the harness.** Harbor deleted its
   harness-side parser layer because it "created an awkward dependency between
   task logic and harness logic". Here an Env may implement ``reward_dict()``
   and return whatever floats describe its own outcome; the harness reads them
   without knowing what a robot is. That is what lets partial credit and
   robotics-native efficiency coexist without touching harness code.

2. **One flat append-only file is the record of truth.** RoboLab keeps a single
   ``episode_results.jsonl`` and recomputes every task- and run-level number on
   read. Emitting the same shape means ``robolab-dashboard`` and
   ``analysis/read_results.py`` work on our runs unchanged; the fields they do
   not know about are simply ignored.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from harness.eval.infra import classify_infra_failure
from harness.types import Episode


class FailureMode:
    """Why an episode ended, so the report can separate kinds of zero.

    A model scoring 0 because its output could not be parsed is telling you
    about the prompt contract, not about its manipulation ability. Folding
    those into one success rate is the most common way a comparison lies.
    """

    NONE = "none"                      # succeeded
    TASK_FAILED = "task_failed"         # ran to completion, success check false
    AGENT_TIMEOUT = "agent_timeout"     # step budget exhausted
    PARSE_FAILURE = "parse_failure"     # output unreadable -- format, not capability
    CONTEXT_EXCEEDED = "context_exceeded"
    REFUSAL = "refusal"
    PROVIDER_ERROR = "provider_error"   # 5xx / rate limit after retries
    HARNESS_ERROR = "harness_error"     # our bug, not the model's

    #: modes that indicate the harness or provider failed, not the model.
    NOT_MODEL_FAULT = (PARSE_FAILURE, CONTEXT_EXCEEDED, PROVIDER_ERROR, HARNESS_ERROR)


def _num(v: Any) -> Any:
    """Make numpy scalars/arrays JSON-safe; NaN/Inf become None."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, np.ndarray):
        return [_num(x) for x in v.tolist()]
    if isinstance(v, float):
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(v, dict):
        return {k: _num(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_num(x) for x in v]
    return v


def default_reward_dict(env, ep: Episode) -> dict:
    """Synthesise a reward dict for an env that does not provide one.

    Envs are encouraged to implement ``reward_dict()`` themselves. This keeps
    every env scoreable in the meantime, and guarantees the ``success`` key
    the leaderboard gates on always exists.
    """
    own = None
    fn = getattr(env, "reward_dict", None)
    if callable(fn):
        try:
            own = fn()
        except Exception:  # noqa: BLE001 - a broken scorer must not lose the run
            own = None

    out: dict[str, Any] = {"success": 1 if ep.success else 0}
    if isinstance(own, dict):
        out.update({k: _num(v) for k, v in own.items()})
        out["success"] = 1 if (own.get("success", ep.success)) else 0
    out.setdefault("sim_steps", ep.steps)
    return out


def classify_failure(ep: Episode, *, error: Optional[BaseException] = None) -> str:
    """Assign a failure mode from what the episode actually recorded."""
    if error is not None:
        name = type(error).__name__.lower()
        if "ratelimit" in name or "transient" in name or "llmerror" in name:
            return FailureMode.PROVIDER_ERROR
        return FailureMode.HARNESS_ERROR
    if ep.success:
        return FailureMode.NONE

    # a run made entirely of unparseable replies is a format failure, not a
    # capability one: parse_action degrades to a commentless noop.
    acts = [a for a in ep.actions if a.kind != "stop"]
    if acts:
        noops = sum(1 for a in acts if a.kind == "noop" and a.value is None and a.gripper is None)
        if noops == len(acts):
            return FailureMode.PARSE_FAILURE

    if ep.infos and ep.infos[-1].get("truncated"):
        return FailureMode.AGENT_TIMEOUT
    # Both keys must actually be present: `None == None` would otherwise make
    # every metadata-less failure look like a timeout.
    steps, budget = ep.metadata.get("steps"), ep.metadata.get("max_steps")
    if steps is not None and budget is not None and steps >= budget:
        return FailureMode.AGENT_TIMEOUT
    return FailureMode.TASK_FAILED


@dataclass
class TrialRecord:
    """One episode. Keys mirror RoboLab's episode_results.jsonl where they overlap."""

    # --- RoboLab-compatible core ---
    env_name: str = ""            # per-task identifier -- what you group by
    task_name: str = ""
    backend: str = ""             # which simulator produced it (tabletop, robolab, ...)
    run: int = 0
    episode: int = 0
    env_id: int = 0
    policy: str = ""              # the model identifier; RoboLab groups on this
    instruction: str = ""
    success: bool = False
    score: Optional[float] = None
    reason: Optional[str] = None
    episode_step: int = 0
    duration: Optional[float] = None
    attributes: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    instruction_type: str = "default"
    dt: Optional[float] = None      # seconds per env step; dashboards need it
    timing: dict = field(default_factory=dict)

    # --- our additions (ignored by RoboLab's readers) ---
    seed: Optional[int] = None
    mode: str = ""
    rewards: dict = field(default_factory=dict)
    failure_mode: str = FailureMode.NONE
    llm_calls: Optional[int] = None
    usage: dict = field(default_factory=dict)
    cost_usd: Optional[float] = None
    wall_clock_s: Optional[float] = None
    harness_version: str = ""
    infra_failure: Optional[dict] = None   # {"reason","tier"} -- advisory only

    def to_dict(self) -> dict:
        return {k: _num(v) for k, v in asdict(self).items()}


def record_from_episode(
    ep: Episode,
    env,
    *,
    policy: str,
    seed: Optional[int] = None,
    episode_index: int = 0,
    mode: str = "",
    wall_clock_s: Optional[float] = None,
    error: Optional[BaseException] = None,
) -> TrialRecord:
    """Build a trial record from a finished (or failed) episode."""
    rewards = default_reward_dict(env, ep)
    score = rewards.get("score")
    if score is None:
        # successes count 1.0; failures fall back to fractional progress when
        # the env reports it. Matches RoboLab's get_avg_score() override.
        sub_done, sub_total = rewards.get("subtasks_completed"), rewards.get("subtasks_total")
        if ep.success:
            score = 1.0
        elif isinstance(sub_done, (int, float)) and sub_total:
            score = float(sub_done) / float(sub_total)

    task_spec = getattr(env, "task_spec", None)
    dt = getattr(env, "dt", None)
    if dt is None:
        sim = getattr(getattr(env, "_env_cfg", None), "sim", None)
        base, dec = getattr(sim, "dt", None), getattr(getattr(env, "_env_cfg", None), "decimation", None)
        dt = float(base) * float(dec) if base and dec else None
    # RoboLab's readers treat env_name as the per-task id and locate
    # <run>/<env_name>/env_cfg.json from it, so env_name must discriminate the
    # task; the simulator goes in `backend`.
    task_id = (
        getattr(env, "task", None)
        or getattr(task_spec, "kind", None)
        or getattr(env, "name", "")
        or "task"
    )
    return TrialRecord(
        env_name=str(task_id),
        task_name=str(task_id),
        backend=getattr(env, "name", "") or "",
        run=0,
        episode=episode_index,
        env_id=episode_index,
        policy=policy,
        instruction=(getattr(task_spec, "description", "") or env.get_text_state() or "").split("\n")[0][:200],
        success=bool(ep.success),
        score=_num(score),
        reason=(ep.actions[-1].comment[:160] if ep.actions and ep.actions[-1].comment else None),
        episode_step=ep.steps,
        duration=wall_clock_s,
        attributes=list(getattr(task_spec, "params", {}).keys()) if task_spec else [],
        metrics={k: v for k, v in rewards.items() if k not in ("success", "score")},
        seed=seed,
        mode=mode or str(ep.metadata.get("mode", "")),
        rewards=rewards,
        failure_mode=classify_failure(ep, error=error),
        infra_failure=classify_infra_failure(
            f"{type(error).__name__}: {error}" if error is not None else None
        ),
        llm_calls=ep.metadata.get("llm_calls"),
        usage=ep.metadata.get("usage") or {},
        cost_usd=ep.metadata.get("cost_usd"),
        wall_clock_s=wall_clock_s,
        dt=dt,
        timing={"wall_total_s": wall_clock_s} if wall_clock_s is not None else {},
    )


class ResultsWriter:
    """Append-only writer for episode_results.jsonl.

    Append-only and flushed per record so a killed run loses at most the
    episode in flight, and a torn final line is skipped on read.
    """

    FILENAME = "episode_results.jsonl"

    def __init__(self, run_dir) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / self.FILENAME
        self.count = 0

    def append(self, record: TrialRecord) -> None:
        self.append_dict(record.to_dict())

    def append_dict(self, record: dict) -> None:
        """Append an already-serialised record (the job runner's path)."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_num(record)) + "\n")
            f.flush()
        self.count += 1

    def write_config(self, cfg: dict, name: str = "config.json") -> Path:
        p = self.dir / name
        p.write_text(json.dumps(_num(cfg), indent=2, default=str), encoding="utf-8")
        return p

    def write_task_config(self, task_name: str, cfg: dict) -> Path:
        """Write <run>/<task>/env_cfg.json.

        RoboLab's dashboard only recognises a directory as a task dir when it
        contains env_cfg.json (or an hdf5, or log_*.json), so without this the
        run is readable for metrics but not for per-task drill-down.
        """
        d = self.dir / (task_name or "task")
        d.mkdir(parents=True, exist_ok=True)
        p = d / "env_cfg.json"
        p.write_text(json.dumps(_num(cfg), indent=2, default=str), encoding="utf-8")
        return p


def load_results(run_dir) -> list[dict]:
    """Read episode_results.jsonl, tolerating a torn final line."""
    p = Path(run_dir) / ResultsWriter.FILENAME
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # partial write from a hard kill
    return out


#: How this harness turns episodes into a reported score. Published alongside
#: every run because the rule itself moves numbers: on a comparison of two
#: public SWE-bench submissions, counting missing submissions as failures
#: rather than excluding them accounted for 2.3pp of a 15.6pp reported gap.
#: A score without its rule is not interpretable.
REPORTING_RULE = {
    "version": 1,
    "success_gate": "reward_dict['success'] == 1, as emitted by the task",
    "score": "successes count 1.0; failures contribute fractional subtask progress",
    "errored_episodes": "counted as failures, never excluded from the denominator",
    "timeouts": "counted as failures (failure_mode=agent_timeout)",
    "infra_failures": "flagged by tier and reported separately; denominator unchanged",
    "harness_faults": "parse/context/provider errors reported separately as "
                      "not_model_fault; still counted as failures in the rate",
    "attempts_per_task": "single attempt per (task, seed); no best-of-k selection",
    "interval": "Beta(k+1, n-k+1) 95% credible interval",
    "reliability": "pass^k averaged per task, tasks with fewer than k trials skipped",
}


def write_per_instance_details(run_dir, records) -> "Path":
    """Emit per_instance_details.json using SWE-bench's field names.

    Their schema is {instance: {cost, api_calls, resolved}} -- the only
    battle-tested cost/efficiency leaderboard shape among the benchmarks
    surveyed, so the names are matched exactly for tooling compatibility.
    """
    out: dict[str, dict] = {}
    for r in records:
        key = f"{r.get('env_name') or 'task'}::seed{r.get('seed')}"
        out[key] = {
            "cost": r.get("cost_usd"),
            "api_calls": r.get("llm_calls"),
            "resolved": bool(r.get("success")),
        }
    p = Path(run_dir) / "per_instance_details.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return p
