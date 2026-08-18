import tempfile
import unittest
from pathlib import Path

from harness.eval.job import (
    AgentSpec,
    JobConfig,
    TrialKey,
    build_summary,
    load_job,
    regrade,
    run_job,
)
from harness.eval.lock import LockMismatch, RunLock, digest_of
from harness.eval.results import load_results


def _cfg(log_dir, tasks=("pick_place", "stack"), seeds=(0, 1), agents=None, name="j"):
    return JobConfig(
        job_name=name, log_dir=log_dir, env_name="tabletop",
        tasks=list(tasks), seeds=list(seeds), concurrency=4,
        agents=agents if agents is not None else [AgentSpec(name="oracle", max_steps=300)],
    )


class TestGrid(unittest.TestCase):
    def test_runs_the_full_cross_product(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, tasks=("pick_place", "stack", "sort"), seeds=(0, 1, 2),
                       agents=[AgentSpec(name="oracle", max_steps=300),
                               AgentSpec(name="null_agent", max_steps=300)])
            s = run_job(cfg)
            self.assertEqual(s["trials"], 3 * 3 * 2)

    def test_each_agent_gets_its_own_results_file(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, agents=[AgentSpec(name="oracle", max_steps=300),
                                  AgentSpec(name="null_agent", max_steps=300)])
            run_job(cfg)
            self.assertEqual(len(load_results(cfg.dir / "oracle")), 4)
            self.assertEqual(len(load_results(cfg.dir / "null")), 4)

    def test_baselines_bracket_the_field(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, tasks=("pick_place", "stack", "sort", "reach_avoid"), seeds=(0, 1),
                       agents=[AgentSpec(name="oracle", max_steps=300),
                               AgentSpec(name="null_agent", max_steps=300)])
            models = run_job(cfg)["leaderboard"]["models"]
            self.assertEqual(models["oracle"]["success_rate"], 1.0)
            self.assertEqual(models["null"]["success_rate"], 0.0)

    def test_concurrent_trials_do_not_share_env_state(self):
        """Envs are stateful; a shared one is the episode-leak bug again.

        With per-trial envs the oracle takes the same step count at every seed;
        a leaked layout shows up as a trial finishing in ~1 step.
        """
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d, tasks=("pick_place",), seeds=tuple(range(8)))
            cfg.concurrency = 8
            run_job(cfg)
            steps = [r["episode_step"] for r in load_results(cfg.dir / "oracle")]
            self.assertTrue(all(s > 3 for s in steps), f"state leaked: {steps}")
            self.assertEqual(len(set(steps)), 1, f"same task+seed should agree: {steps}")

    def test_empty_agent_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                run_job(_cfg(d, agents=[]))

    def test_agent_id_collapses_baseline_aliases(self):
        self.assertEqual(AgentSpec(name="oracle_agent").id, "oracle")
        self.assertEqual(AgentSpec(name="nop").id, "null")
        self.assertEqual(AgentSpec(name="llm_controller", model="a/b").id, "a_b")


class TestResume(unittest.TestCase):
    def test_resume_skips_completed_trials(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d)
            first = run_job(cfg)
            again = run_job(cfg)
            self.assertEqual(first["trials"], again["trials"])
            # nothing duplicated on disk
            self.assertEqual(len(load_job(cfg.dir)), first["trials"])

    def test_partial_results_are_completed_not_restarted(self):
        with tempfile.TemporaryDirectory() as d:
            small = _cfg(d, tasks=("pick_place",), seeds=(0,))
            run_job(small)
            # widen the seed set: the lock guard must refuse an inconsistent resume
            wider = _cfg(d, tasks=("pick_place",), seeds=(0, 1))
            with self.assertRaises(LockMismatch):
                run_job(wider)
            # ...and starting fresh runs the whole widened grid
            self.assertEqual(run_job(wider, resume=False)["trials"], 2)

    def test_changed_task_set_is_refused_with_a_reason(self):
        with tempfile.TemporaryDirectory() as d:
            run_job(_cfg(d, tasks=("pick_place",)))
            with self.assertRaises(LockMismatch) as ctx:
                run_job(_cfg(d, tasks=("pick_place", "stack")))
            self.assertIn("tasks", str(ctx.exception))

    def test_summary_is_recomputable_from_disk(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = _cfg(d)
            live = run_job(cfg)
            from_disk = build_summary(cfg, load_job(cfg.dir))
            self.assertEqual(
                live["leaderboard"]["models"]["oracle"]["success_rate"],
                from_disk["leaderboard"]["models"]["oracle"]["success_rate"],
            )
            self.assertEqual(live["trials"], from_disk["trials"])


class TestLock(unittest.TestCase):
    def test_equality_ignores_timestamp_and_invocation(self):
        a = RunLock.capture(tasks=["t"], agents=[{"n": 1}], seeds=[0], created_at="2020")
        b = RunLock.capture(tasks=["t"], agents=[{"n": 1}], seeds=[0], created_at="2026")
        self.assertEqual(a, b)

    def test_equality_catches_a_changed_seed_set(self):
        a = RunLock.capture(tasks=["t"], agents=[], seeds=[0, 1])
        b = RunLock.capture(tasks=["t"], agents=[], seeds=[0, 1, 2])
        self.assertNotEqual(a, b)
        self.assertTrue(any("seeds" in m for m in a.describe_mismatch(b)))

    def test_records_provenance_that_identifies_the_code(self):
        lk = RunLock.capture(tasks=["t"], agents=[], seeds=[0])
        self.assertTrue(lk.harness_version)
        self.assertIsInstance(lk.is_editable_installation, bool)
        self.assertTrue(lk.task_digest.startswith("sha256:"))

    def test_digest_is_order_independent(self):
        self.assertEqual(digest_of({"a": 1, "b": 2}), digest_of({"b": 2, "a": 1}))

    def test_roundtrip_through_disk(self):
        with tempfile.TemporaryDirectory() as d:
            lk = RunLock.capture(tasks=["t"], agents=[], seeds=[0])
            p = lk.write(Path(d) / "lock.json")
            self.assertEqual(RunLock.read(p), lk)

    def test_missing_or_corrupt_lock_reads_as_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(RunLock.read(Path(d) / "nope.json"))
            bad = Path(d) / "lock.json"
            bad.write_text("{not json")
            self.assertIsNone(RunLock.read(bad))


class TestRegrade(unittest.TestCase):
    def _job(self, d):
        cfg = _cfg(d, tasks=("pick_place", "sort"), seeds=(0, 1))
        run_job(cfg)
        return cfg

    def test_regrade_changes_scores_without_running_trials(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._job(d)

            def stricter(rec):
                ok = bool(rec.get("success")) and int(rec.get("episode_step") or 0) <= 4
                return {"success": ok, "score": 1.0 if ok else 0.0}

            out = regrade(cfg.dir, stricter)
            # sort needs 8 oracle steps, so its trials must flip to failure
            self.assertEqual(out["changed"], 2)
            self.assertEqual(out["down"], 2)
            self.assertEqual(out["up"], 0)
            self.assertLess(out["success_after"], out["success_before"])

    def test_source_records_are_never_modified(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._job(d)
            regrade(cfg.dir, lambda r: {"success": False, "score": 0.0})
            rows = load_job(cfg.dir)
            self.assertTrue(all(r["success"] for r in rows))
            self.assertFalse(any(r.get("regraded") for r in rows))

    def test_regraded_copy_is_marked(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._job(d)
            out = regrade(cfg.dir, lambda r: {})
            rows = load_job(out["regraded_dir"])
            self.assertTrue(rows and all(r.get("regraded") for r in rows))

    def test_noop_scorer_changes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._job(d)
            out = regrade(cfg.dir, lambda r: {})
            self.assertEqual(out["changed"], 0)
            self.assertEqual(out["success_before"], out["success_after"])

    def test_regrade_on_an_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "empty").mkdir()
            with self.assertRaises(FileNotFoundError):
                regrade(Path(d) / "empty", lambda r: {})


class TestTrialKey(unittest.TestCase):
    def test_matches_on_all_three_axes(self):
        key = TrialKey(task="t", agent_id="m", seed=3)
        self.assertTrue(key.matches({"env_name": "t", "policy": "m", "seed": 3}))
        for bad in ({"env_name": "x", "policy": "m", "seed": 3},
                    {"env_name": "t", "policy": "z", "seed": 3},
                    {"env_name": "t", "policy": "m", "seed": 9}):
            self.assertFalse(key.matches(bad))


if __name__ == "__main__":
    unittest.main()
