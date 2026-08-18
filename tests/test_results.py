import json
import tempfile
import unittest
from pathlib import Path

from harness.config import HarnessConfig
from harness.envs.tabletop import TabletopEnv
from harness.eval.metrics import summarize, summarize_records
from harness.eval.results import (
    FailureMode,
    ResultsWriter,
    TrialRecord,
    classify_failure,
    default_reward_dict,
    load_results,
    record_from_episode,
)
from harness.eval.stats import (
    aggregate_over_tasks,
    beta_ci,
    mcnemar,
    pass_at_k,
    pass_hat_k,
    rank_interval,
    resolution_ratio,
    wilson_ci,
)
from harness.runner import run_eval
from harness.tasks import generate_task
from harness.types import Action, Episode


class TestStats(unittest.TestCase):
    def test_beta_ci_matches_published_reference_values(self):
        # RoboLab documents these exact intervals for its estimator. Only the
        # interior cases are compared: at k=0 and k=n we deliberately switch to a
        # one-sided interval (see the boundary test below), so the equal-tailed
        # reference values there do not apply.
        for k, n, lo, hi in [(6, 10, 0.308, 0.833), (60, 100, 0.499, 0.694)]:
            got = beta_ci(k, n)
            self.assertAlmostEqual(got[0], lo, places=2, msg=f"{k}/{n} lower")
            self.assertAlmostEqual(got[1], hi, places=2, msg=f"{k}/{n} upper")

    def test_beta_ci_stays_in_unit_interval_at_the_boundary(self):
        # a Wald standard error is simply wrong at 10/10
        lo, hi = beta_ci(10, 10)
        self.assertGreater(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_interval_always_contains_its_own_point_estimate(self):
        """The property that forced the one-sided boundary convention.

        An equal-tailed interval at k=0 returns something like [0.2%, 28.5%] --
        it excludes the observed 0%. Rendered beside a bar of length zero that
        reads as a harness bug, and it blurs "scored zero" into "we are unsure it
        is zero". Both estimators must bracket what was measured.
        """
        for estimator in (beta_ci, wilson_ci):
            for k, n in [(0, 1), (1, 1), (0, 10), (10, 10), (0, 30), (30, 30), (2, 5)]:
                lo, hi = estimator(k, n)
                with self.subTest(estimator=estimator.__name__, k=k, n=n):
                    self.assertLessEqual(lo, k / n)
                    self.assertLessEqual(k / n, hi)

    def test_boundary_intervals_are_one_sided(self):
        self.assertEqual(beta_ci(0, 30)[0], 0.0)
        self.assertEqual(beta_ci(30, 30)[1], 1.0)
        # ...and spending all of alpha on one tail tightens that tail
        self.assertLess(beta_ci(0, 30)[1], 0.10)

    def test_beta_ci_degenerate_n(self):
        self.assertEqual(beta_ci(0, 0), (0.0, 1.0))

    def test_wilson_fallback_is_close_to_beta(self):
        for k, n in [(3, 10), (6, 10), (50, 100)]:
            b, w = beta_ci(k, n), wilson_ci(k, n)
            self.assertLess(abs(b[0] - w[0]), 0.06)
            self.assertLess(abs(b[1] - w[1]), 0.06)

    def test_pass_at_k_and_pass_hat_k_agree_at_k1(self):
        self.assertEqual(pass_at_k(4, 5, 1), pass_hat_k(4, 5, 1))

    def test_pass_hat_k_collapses_where_pass_at_k_saturates(self):
        # the reliability point: 4/5 looks strong until you demand all 5
        self.assertEqual(pass_at_k(4, 5, 5), 1.0)
        self.assertEqual(pass_hat_k(4, 5, 5), 0.0)

    def test_pass_hat_k_is_monotonically_non_increasing(self):
        vals = [pass_hat_k(8, 10, k) for k in range(1, 9)]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_too_few_trials_returns_none(self):
        self.assertIsNone(pass_hat_k(3, 4, 9))
        self.assertIsNone(pass_at_k(3, 4, 9))

    def test_aggregate_skips_tasks_with_too_few_trials(self):
        # second task has only 1 trial, so it cannot contribute to k=2
        v = aggregate_over_tasks([(2, 2), (1, 1)], 2, metric="pass_hat_k")
        self.assertEqual(v, 1.0)

    def test_aggregate_empty_is_none(self):
        self.assertIsNone(aggregate_over_tasks([], 2))

    def test_mcnemar_uses_only_discordant_pairs(self):
        strong = mcnemar(both=40, only_a=12, only_b=3)
        self.assertEqual(strong["discordant"], 15)
        self.assertLess(strong["p_value"], 0.05)
        weak = mcnemar(both=40, only_a=5, only_b=4)
        self.assertGreater(weak["p_value"], 0.05)

    def test_mcnemar_no_discordant_pairs(self):
        self.assertEqual(mcnemar(both=10, only_a=0, only_b=0)["p_value"], 1.0)

    def test_rank_interval_reports_ties_as_ties(self):
        # three overlapping intervals must not be given distinct ranks
        ivs = [(0.67, 0.76), (0.63, 0.73), (0.58, 0.68), (0.29, 0.39)]
        ranks = rank_interval([0.72, 0.68, 0.63, 0.34], ivs)
        self.assertEqual(ranks[0], (1, 3))
        self.assertEqual(ranks[3], (4, 4))  # clearly separated

    def test_resolution_ratio_flags_underpowered_comparison(self):
        self.assertLess(resolution_ratio(50, 8.0), 1.0)
        self.assertGreater(resolution_ratio(5000, 8.0), 1.0)


class TestRewardDict(unittest.TestCase):
    def _env(self):
        env = TabletopEnv(task_spec=generate_task("pick_place", seed=1))
        env.reset()
        return env

    def test_task_supplies_its_own_reward_dict(self):
        rd = self._env().reward_dict()
        for key in ("success", "sim_steps", "score", "subtasks_completed", "subtasks_total"):
            self.assertIn(key, rd)

    def test_env_without_reward_dict_still_scores(self):
        class Bare:
            name = "bare"

            def get_text_state(self):
                return ""

        rd = default_reward_dict(Bare(), Episode(success=True))
        self.assertEqual(rd["success"], 1)
        self.assertIn("sim_steps", rd)

    def test_broken_scorer_does_not_lose_the_run(self):
        class Boom:
            name = "boom"

            def reward_dict(self):
                raise RuntimeError("scorer bug")

        rd = default_reward_dict(Boom(), Episode(success=False))
        self.assertEqual(rd["success"], 0)


class TestFailureClassification(unittest.TestCase):
    def test_success(self):
        self.assertEqual(classify_failure(Episode(success=True)), FailureMode.NONE)

    def test_all_unparseable_replies_is_a_parse_failure_not_a_capability_one(self):
        ep = Episode(actions=[Action(kind="noop"), Action(kind="noop")],
                     metadata={"llm_calls": 2})
        self.assertEqual(classify_failure(ep), FailureMode.PARSE_FAILURE)

    def test_a_baselines_deliberate_inaction_is_not_a_parse_failure(self):
        """The null agent emits noops by design.

        Classifying that as a format failure both slanders the harness and puts
        100% of the baseline's trials into not_model_fault, so a healthy board
        reports a wall of harness faults it did not cause.
        """
        ep = Episode(actions=[Action(kind="noop"), Action(kind="noop")])
        self.assertNotEqual(classify_failure(ep), FailureMode.PARSE_FAILURE)

    def test_real_actions_that_miss_are_task_failures(self):
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.2])])
        self.assertEqual(classify_failure(ep), FailureMode.TASK_FAILED)

    def test_provider_error_is_separated_from_harness_error(self):
        class TransientLLMError(Exception):
            pass

        self.assertEqual(
            classify_failure(Episode(), error=TransientLLMError()), FailureMode.PROVIDER_ERROR
        )
        self.assertEqual(classify_failure(Episode(), error=KeyError()), FailureMode.HARNESS_ERROR)


class TestWriter(unittest.TestCase):
    def test_append_and_reload(self):
        with tempfile.TemporaryDirectory() as d:
            w = ResultsWriter(d)
            w.append(TrialRecord(env_name="t", policy="m", success=True))
            w.append(TrialRecord(env_name="t", policy="m", success=False))
            rows = load_results(d)
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]["success"])

    def test_torn_final_line_is_tolerated(self):
        with tempfile.TemporaryDirectory() as d:
            w = ResultsWriter(d)
            w.append(TrialRecord(env_name="t", policy="m", success=True))
            with w.path.open("a") as f:
                f.write('{"partial": ')  # simulate a hard kill mid-write
            self.assertEqual(len(load_results(d)), 1)

    def test_non_finite_values_are_json_safe(self):
        with tempfile.TemporaryDirectory() as d:
            w = ResultsWriter(d)
            w.append(TrialRecord(env_name="t", policy="m", score=float("nan"),
                                 metrics={"x": float("inf")}))
            raw = w.path.read_text()
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            json.loads(raw.splitlines()[0])  # must parse with a strict reader

    def test_record_from_episode_carries_identity(self):
        env = TabletopEnv(task_spec=generate_task("pick_place", seed=1))
        env.reset()
        rec = record_from_episode(Episode(success=False), env, policy="m1", seed=7,
                                  episode_index=3, mode="tools")
        self.assertEqual((rec.policy, rec.seed, rec.episode), ("m1", 7, 3))
        self.assertEqual(rec.backend, "tabletop")
        self.assertEqual(rec.env_name, "pick_place")  # per-task id, as readers expect


class TestSummaries(unittest.TestCase):
    def test_summarize_keeps_its_old_keys(self):
        s = summarize([Episode(success=True), Episode(success=False)])
        for k in ("episodes", "success_rate", "mean_steps", "mean_reward"):
            self.assertIn(k, s)

    def test_summarize_adds_an_interval(self):
        s = summarize([Episode(success=True)] * 6)
        self.assertIn("success_ci_95", s)
        # 6/6 is compatible with a much lower true rate; the interval must say so
        self.assertLess(s["success_ci_95"][0], 0.7)

    def test_summarize_empty(self):
        self.assertEqual(summarize([]), {})

    def test_partial_pricing_reports_none_not_a_wrong_total(self):
        a = Episode(success=True, metadata={"cost_usd": 0.01})
        b = Episode(success=True, metadata={"cost_usd": None})
        s = summarize([a, b])
        self.assertIsNone(s["cost_usd"])
        self.assertIn("cost_note", s)

    def test_summarize_records_groups_by_model_and_task(self):
        rows = []
        for model, wins in (("m1", [1, 1, 0]), ("m2", [0, 0, 0])):
            for i, w in enumerate(wins):
                rows.append({"policy": model, "env_name": "taskA", "success": bool(w),
                             "score": float(w), "episode_step": 5,
                             "failure_mode": "none" if w else "task_failed"})
        out = summarize_records(rows)
        self.assertAlmostEqual(out["models"]["m1"]["success_rate"], 2 / 3, places=3)
        self.assertEqual(out["models"]["m2"]["success_rate"], 0.0)
        self.assertIn("per_task", out["models"]["m1"])

    def test_summarize_records_separates_harness_faults(self):
        rows = [{"policy": "m", "env_name": "t", "success": False,
                 "failure_mode": "parse_failure"} for _ in range(4)]
        out = summarize_records(rows)["models"]["m"]
        self.assertEqual(out["not_model_fault"], 4)
        self.assertEqual(out["not_model_fault_pct"], 100.0)


class TestRunnerIsolation(unittest.TestCase):
    def _cfg(self, model, log_dir):
        return HarnessConfig.from_dict({
            "seed": 0,
            "llm": {"provider": "mock", "model": model,
                    "extra": {"responses": ['{"tool":"done","args":{}}']}},
            "env": {"name": "tabletop", "task": "pick_place"},
            "agent": {"mode": "tools", "max_steps": 3},
            "eval": {"episodes": 2, "log_dir": log_dir, "verbose": False},
            "viz": {"enabled": False, "backend": "none"},
        })

    def test_two_models_do_not_share_a_results_file(self):
        with tempfile.TemporaryDirectory() as d:
            a = run_eval(self._cfg("model-a", d))
            b = run_eval(self._cfg("model-b", d))
            self.assertNotEqual(a["results_file"], b["results_file"])
            self.assertEqual(len(load_results(Path(a["run_dir"]))), 2)
            self.assertEqual(len(load_results(Path(b["run_dir"]))), 2)

    def test_run_emits_a_task_config_readers_can_find(self):
        with tempfile.TemporaryDirectory() as d:
            s = run_eval(self._cfg("model-a", d))
            rows = load_results(Path(s["run_dir"]))
            task_dir = Path(s["run_dir"]) / rows[0]["env_name"]
            self.assertTrue((task_dir / "env_cfg.json").exists())


if __name__ == "__main__":
    unittest.main()


class TestInfraFailures(unittest.TestCase):
    def test_gpu_oom_is_an_environment_fault(self):
        from harness.eval.infra import TIER_ENVIRONMENT, classify_infra_failure

        got = classify_infra_failure("RuntimeError: CUDA out of memory. Tried to allocate 2 GiB")
        self.assertEqual(got["tier"], TIER_ENVIRONMENT)
        self.assertEqual(got["reason"], "out_of_memory")

    def test_missing_asset_is_ambiguous_not_environment(self):
        from harness.eval.infra import TIER_AMBIGUOUS, classify_infra_failure

        got = classify_infra_failure("Could not open asset @/assets/banana.usd@")
        self.assertEqual(got["tier"], TIER_AMBIGUOUS)

    def test_no_evidence_is_none_not_a_guess(self):
        from harness.eval.infra import classify_infra_failure

        self.assertIsNone(classify_infra_failure(None))
        self.assertIsNone(classify_infra_failure("the gripper missed the banana"))

    def test_infra_counts_never_change_the_denominator(self):
        from harness.eval.infra import summarize_infra

        rows = [
            {"success": False, "infra_failure": {"reason": "out_of_memory", "tier": "environment"}},
            {"success": False, "infra_failure": {"reason": "missing_asset", "tier": "ambiguous"}},
            {"success": True, "infra_failure": None},
        ]
        out = summarize_infra(rows)
        self.assertEqual(out["environment_failures"], 1)
        self.assertEqual(out["ambiguous_failures"], 1)
        self.assertTrue(out["denominator_unchanged"])
        # the success rate is computed independently and is untouched by the above
        self.assertAlmostEqual(summarize_records(
            [{**r, "policy": "m", "env_name": "t"} for r in rows]
        )["models"]["m"]["success_rate"], 1 / 3, places=3)

    def test_no_infra_flags_yields_empty(self):
        from harness.eval.infra import summarize_infra

        self.assertEqual(summarize_infra([{"success": True}]), {})


class TestReportingRule(unittest.TestCase):
    def test_rule_is_published_with_the_run(self):
        import tempfile as _t

        from harness.eval.results import REPORTING_RULE

        with _t.TemporaryDirectory() as d:
            s = run_eval(HarnessConfig.from_dict({
                "seed": 0,
                "llm": {"provider": "mock", "model": "m",
                        "extra": {"responses": ['{"tool":"done","args":{}}']}},
                "env": {"name": "tabletop", "task": "pick_place"},
                "agent": {"mode": "tools", "max_steps": 3},
                "eval": {"episodes": 2, "log_dir": d, "verbose": False},
                "viz": {"enabled": False, "backend": "none"},
            }))
            self.assertEqual(s["reporting_rule"], REPORTING_RULE)
            # the two choices that most move a reported number must be stated
            self.assertIn("never excluded", s["reporting_rule"]["errored_episodes"])
            self.assertIn("denominator unchanged", s["reporting_rule"]["infra_failures"])

    def test_per_instance_details_uses_swebench_field_names(self):
        import json as _j
        import tempfile as _t

        with _t.TemporaryDirectory() as d:
            s = run_eval(HarnessConfig.from_dict({
                "seed": 0,
                "llm": {"provider": "mock", "model": "m",
                        "extra": {"responses": ['{"tool":"done","args":{}}']}},
                "env": {"name": "tabletop", "task": "pick_place"},
                "agent": {"mode": "tools", "max_steps": 3},
                "eval": {"episodes": 2, "log_dir": d, "verbose": False},
                "viz": {"enabled": False, "backend": "none"},
            }))
            data = _j.loads(Path(s["per_instance_details"]).read_text())
            self.assertEqual(len(data), 2)
            entry = next(iter(data.values()))
            self.assertEqual(sorted(entry), ["api_calls", "cost", "resolved"])


class TestBaselines(unittest.TestCase):
    """Reference agents, and the env bug they exposed."""

    def _env(self, task, seed=0):
        return TabletopEnv(task_spec=generate_task(task, seed=seed))

    def test_oracle_solves_every_2d_task(self):
        from harness.agent.baselines import OracleAgent
        from harness.tasks.base import _TASK_GENERATORS_2D as REG

        for task in sorted(REG):
            if task == "cook_bread":
                continue  # kitchen task, needs KitchenEnv
            ep = OracleAgent(max_steps=300).run(self._env(task), seed=0)
            self.assertTrue(ep.success, f"oracle failed {task} -- task may be unsolvable")

    def test_null_agent_solves_nothing(self):
        from harness.agent.baselines import NullAgent
        from harness.tasks.base import _TASK_GENERATORS_2D as REG

        for task in sorted(REG):
            if task == "cook_bread":
                continue
            ep = NullAgent().run(self._env(task), seed=0)
            self.assertFalse(ep.success, f"{task} passes with no actions -- vacuous success check")

    def test_episodes_are_independent(self):
        """reset() must restore object poses, not just the arm.

        step() carries a grasped object by writing self._objects[name]. Without
        restoring them, a solved task stays solved and every later episode
        starts already successful -- silently inflating any multi-episode rate.
        """
        from harness.agent.baselines import OracleAgent

        env = self._env("pick_place")
        steps = [OracleAgent(max_steps=300).run(env, seed=s).steps for s in range(4)]
        self.assertTrue(all(s > 3 for s in steps), f"layout leaked between episodes: {steps}")
        self.assertEqual(len(set(steps)), 1, f"same seed should be deterministic: {steps}")

    def test_null_does_not_inherit_a_solved_layout(self):
        from harness.agent.baselines import NullAgent, OracleAgent

        env = self._env("pick_place")
        self.assertTrue(OracleAgent(max_steps=300).run(env, seed=0).success)
        self.assertFalse(NullAgent().run(env, seed=1).success)

    def test_oracle_supplies_an_efficiency_denominator(self):
        from harness.eval.metrics import oracle_steps_by_task

        rows = [{"policy": "oracle", "env_name": "t", "success": True, "episode_step": 15},
                {"policy": "oracle", "env_name": "t", "success": True, "episode_step": 17},
                {"policy": "oracle", "env_name": "t", "success": False, "episode_step": 3}]
        self.assertEqual(oracle_steps_by_task(rows)["t"], 17)  # median of successes only

    def test_soft_spl_rewards_efficiency_not_just_success(self):
        rows = [{"policy": "fast", "env_name": "t", "success": True, "episode_step": 10},
                {"policy": "slow", "env_name": "t", "success": True, "episode_step": 40}]
        out = summarize_records(rows, oracle_steps={"t": 10})
        self.assertGreater(out["models"]["fast"]["soft_spl"], out["models"]["slow"]["soft_spl"])
        self.assertAlmostEqual(out["models"]["fast"]["steps_vs_oracle"], 1.0, places=3)

    def test_baselines_run_through_the_runner(self):
        with tempfile.TemporaryDirectory() as d:
            for name, expect in (("oracle", True), ("null_agent", False)):
                s = run_eval(HarnessConfig.from_dict({
                    "seed": 0, "llm": {"provider": "mock"},
                    "env": {"name": "tabletop", "task": "pick_place"},
                    "agent": {"name": name, "max_steps": 300},
                    "eval": {"episodes": 2, "log_dir": d, "verbose": False},
                    "viz": {"enabled": False, "backend": "none"},
                }))
                self.assertEqual(s["success_rate"] == 1.0, expect, name)
