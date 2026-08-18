"""Tests for the HTML leaderboard.

Two kinds of assertion here, and the split is deliberate:

- **Computable claims** (contrast of every cell ink, self-containedness, no
  external requests) are recomputed, not eyeballed. A palette that regressed to
  white-on-mid-blue is a real accessibility failure and a screenshot review will
  not reliably catch it.
- **Audit claims** (the oracle failed, null passed, a baseline is missing) must
  be *loud in the output*, because a leaderboard whose tasks are broken is worse
  than no leaderboard -- it looks authoritative while ranking noise.
"""
import re
import tempfile
import unittest
from pathlib import Path

from harness.eval.job import AgentSpec, JobConfig, build_summary, run_job
from harness.eval.report import (
    _RAMP_DARK,
    _RAMP_INK_DARK,
    _RAMP_INK_LIGHT,
    _RAMP_LIGHT,
    _ramp_index,
    render_report,
    write_report,
)

TASKS = ["pick_place", "stack"]


def _rel_luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    def channel(v: float) -> float:
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    la, lb = _rel_luminance(a), _rel_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _rec(task, agent, seed, ok, **kw):
    r = {"env_name": task, "policy": agent, "seed": seed, "success": ok,
         "score": 1.0 if ok else 0.0, "episode_step": 10,
         "failure_mode": None if ok else "timeout"}
    r.update(kw)
    return r


def _summary(records, agents, tasks=TASKS, seeds=(0, 1)):
    with tempfile.TemporaryDirectory() as d:
        cfg = JobConfig(job_name="t", log_dir=d, env_name="tabletop",
                        tasks=list(tasks), seeds=list(seeds),
                        agents=[AgentSpec(name=a) for a in agents])
        return build_summary(cfg, records)


class TestCellContrast(unittest.TestCase):
    """A heatmap label sits on its fill, so it must contrast with the fill."""

    def test_every_ramp_step_ink_passes_aa(self):
        for mode, ramp, inks in (("light", _RAMP_LIGHT, _RAMP_INK_LIGHT),
                                 ("dark", _RAMP_DARK, _RAMP_INK_DARK)):
            self.assertEqual(len(ramp), len(inks), mode)
            for i, (fill, ink) in enumerate(zip(ramp, inks)):
                with self.subTest(mode=mode, step=i):
                    self.assertGreaterEqual(
                        _contrast(fill, ink), 4.5,
                        f"{mode} step {i}: {ink} on {fill} is unreadable")

    def test_each_step_takes_the_better_of_the_two_inks(self):
        """Guards the non-obvious case: white is *not* always right on a dark fill."""
        for mode, ramp, inks in (("light", _RAMP_LIGHT, _RAMP_INK_LIGHT),
                                 ("dark", _RAMP_DARK, _RAMP_INK_DARK)):
            for i, (fill, ink) in enumerate(zip(ramp, inks)):
                other = "#ffffff" if ink == "#0b0b0b" else "#0b0b0b"
                with self.subTest(mode=mode, step=i):
                    self.assertGreaterEqual(_contrast(fill, ink), _contrast(fill, other))

    def test_ramp_is_monotone_in_lightness(self):
        light = [_rel_luminance(c) for c in _RAMP_LIGHT]
        dark = [_rel_luminance(c) for c in _RAMP_DARK]
        self.assertEqual(light, sorted(light, reverse=True), "light ramp not monotone")
        self.assertEqual(dark, sorted(dark), "dark ramp not monotone")

    def test_ramp_index_spans_the_full_range_and_handles_no_data(self):
        n = len(_RAMP_LIGHT)
        self.assertEqual(_ramp_index(0.0, n), 0)
        self.assertEqual(_ramp_index(1.0, n), n - 1)
        self.assertEqual(_ramp_index(None, n), -1)
        self.assertEqual(_ramp_index(2.0, n), n - 1)  # clamped, never out of range
        self.assertEqual(_ramp_index(-1.0, n), 0)


class TestSelfContained(unittest.TestCase):
    def test_page_makes_no_external_requests(self):
        recs = [_rec(t, "oracle", s, True) for t in TASKS for s in (0, 1)]
        html = render_report(_summary(recs, ["oracle"]), recs)
        for pattern in (r"https?://", r"src\s*=\s*[\"']//", r"@import"):
            self.assertIsNone(re.search(pattern, html), f"external reference: {pattern}")

    def test_colours_are_never_defined_only_inside_a_theme_block(self):
        """The classic unreadable-artifact bug: a token with no bare :root value."""
        recs = [_rec(t, "oracle", s, True) for t in TASKS for s in (0, 1)]
        html = render_report(_summary(recs, ["oracle"]), recs)
        base = html.split("@media", 1)[0]
        for token in ("--bg", "--ink", "--panel", "--viz-series", "--viz-track"):
            self.assertIn(token + ":", base, f"{token} has no light-mode default")

    def test_both_theme_states_are_handled(self):
        recs = [_rec(t, "oracle", s, True) for t in TASKS for s in (0, 1)]
        html = render_report(_summary(recs, ["oracle"]), recs)
        self.assertIn("prefers-color-scheme: dark", html)
        self.assertIn('[data-theme="dark"]', html)
        self.assertIn('data-theme="light"', html)  # explicit light beats a dark OS


class TestAuditCallouts(unittest.TestCase):
    def test_oracle_failing_a_task_is_called_out(self):
        recs = ([_rec("pick_place", "oracle", s, True) for s in (0, 1)]
                + [_rec("stack", "oracle", s, False) for s in (0, 1)]
                + [_rec(t, "null", s, False) for t in TASKS for s in (0, 1)])
        html = render_report(_summary(recs, ["oracle", "null_agent"]), recs)
        self.assertIn("oracle does not solve", html)
        self.assertIn("<code>stack</code>", html)

    def test_null_passing_is_called_out_as_vacuous(self):
        recs = ([_rec(t, "oracle", s, True) for t in TASKS for s in (0, 1)]
                + [_rec("pick_place", "null", s, True) for s in (0, 1)]
                + [_rec("stack", "null", s, False) for s in (0, 1)])
        html = render_report(_summary(recs, ["oracle", "null_agent"]), recs)
        self.assertIn("vacuous", html)
        self.assertIn("<code>pick_place</code>", html)

    def test_a_missing_baseline_is_called_out(self):
        recs = [_rec(t, "m", s, True) for t in TASKS for s in (0, 1)]
        html = render_report(_summary(recs, ["llm_controller"]), recs)
        self.assertIn("no oracle", html)

    def test_a_clean_job_raises_no_audit_callout(self):
        recs = ([_rec(t, "oracle", s, True) for t in TASKS for s in (0, 1)]
                + [_rec(t, "null", s, False) for t in TASKS for s in (0, 1)])
        html = render_report(_summary(recs, ["oracle", "null_agent"]), recs)
        self.assertNotIn("audit", html.lower().split("reporting rule")[0].replace(
            "audited", ""))


class TestCells(unittest.TestCase):
    def test_every_cell_prints_its_count_so_colour_is_never_alone(self):
        recs = ([_rec("pick_place", "m", 0, True), _rec("pick_place", "m", 1, False)]
                + [_rec("stack", "m", s, False) for s in (0, 1)])
        html = render_report(_summary(recs, ["llm_controller"]), recs)
        self.assertIn("1/2", html)
        self.assertIn("0/2", html)

    def test_a_task_with_no_trials_is_distinguished_from_a_real_zero(self):
        recs = [_rec("pick_place", "m", s, False) for s in (0, 1)]  # nothing for stack
        html = render_report(_summary(recs, ["llm_controller"]), recs)
        self.assertIn("no trials recorded", html)
        self.assertIn("0/2", html)  # ...and the genuine zero still reads as a zero

    def test_zero_percent_draws_no_bar(self):
        """A sliver at zero implies a value that is not there."""
        from harness.eval.report import _bar_svg
        self.assertIn('width="0.0"', _bar_svg(0.0, 0.0, 0.1))
        self.assertNotIn('width="0.0"', _bar_svg(0.5, 0.3, 0.7))

    def test_interval_end_caps_stay_inside_the_viewbox(self):
        svg = _bar_svg_lines = None
        from harness.eval.report import _bar_svg
        svg = _bar_svg(0.5, 0.0, 1.0, width=100)
        xs = [float(m) for m in re.findall(r'x1="([\d.]+)"', svg)]
        self.assertTrue(all(0 < x < 100 for x in xs), xs)


class TestWriteReport(unittest.TestCase):
    def test_writes_beside_the_job_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = JobConfig(job_name="j", log_dir=d, env_name="tabletop",
                            tasks=["pick_place"], seeds=[0],
                            agents=[AgentSpec(name="oracle", max_steps=300)])
            run_job(cfg)
            out = write_report(cfg.dir)
            self.assertEqual(Path(out).name, "report.html")
            self.assertTrue(Path(out).read_text().startswith("<!doctype html>"))

    def test_honours_an_explicit_output_path(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = JobConfig(job_name="j", log_dir=d, env_name="tabletop",
                            tasks=["pick_place"], seeds=[0],
                            agents=[AgentSpec(name="oracle", max_steps=300)])
            run_job(cfg)
            target = Path(d) / "sub" / "board.html"
            self.assertEqual(Path(write_report(cfg.dir, target)), target)
            self.assertTrue(target.exists())

    def test_missing_job_dir_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                write_report(Path(d) / "nope")

    def test_title_defaults_to_the_job_name(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = JobConfig(job_name="my-job", log_dir=d, env_name="tabletop",
                            tasks=["pick_place"], seeds=[0],
                            agents=[AgentSpec(name="oracle", max_steps=300)])
            run_job(cfg)
            self.assertIn("my-job", Path(write_report(cfg.dir)).read_text())


class TestEscaping(unittest.TestCase):
    def test_task_and_agent_names_are_escaped(self):
        recs = [_rec("<script>x</script>", "m", 0, True)]
        html = render_report(_summary(recs, ["llm_controller"],
                                     tasks=["<script>x</script>"], seeds=[0]), recs)
        self.assertNotIn("<script>x</script>", html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()


class TestCliRouting(unittest.TestCase):
    """`harness <config.yaml>` must survive the addition of subcommands.

    argparse matches a leading positional against the subparser choices and
    rejects anything else, so adding `job` silently broke the original single
    config surface -- with an error that blamed the config path.
    """

    def test_a_config_path_is_not_mistaken_for_a_subcommand(self):
        from unittest import mock

        from harness import cli
        with mock.patch.object(cli, "load_config") as lc, \
             mock.patch.object(cli, "run_eval") as re_:
            cli.main(["configs/toy_tabletop.yaml"])
        lc.assert_called_once_with("configs/toy_tabletop.yaml")
        re_.assert_called_once()

    def test_episodes_override_still_applies(self):
        from unittest import mock

        from harness import cli
        cfg = mock.MagicMock()
        with mock.patch.object(cli, "load_config", return_value=cfg), \
             mock.patch.object(cli, "run_eval"):
            cli.main(["some.yaml", "--episodes", "3"])
        self.assertEqual(cfg.eval.episodes, 3)

    def test_subcommands_still_route_to_their_own_parsers(self):
        from unittest import mock

        from harness import cli
        with mock.patch("harness.eval.report.write_report", return_value="x.html") as wr:
            cli.main(["report", "some/job", "-o", "x.html"])
        wr.assert_called_once()


class TestSeedVariation(unittest.TestCase):
    """A grid of N seeds must be N different instances, not one instance N times.

    reset(seed=) originally reseeded only the RNG while restoring the layout
    captured at construction, so every seed replayed the identical episode. The
    grid still reported n = tasks x seeds x agents, so every confidence interval
    was computed on roughly sqrt(N) times more independence than existed.
    """

    TASKS = ("pick_place", "pick_place_obstacle", "push", "stack", "sort", "reach_avoid")

    def test_distinct_seeds_give_distinct_layouts(self):
        from harness.envs.tabletop import TabletopEnv
        for task in self.TASKS:
            states = set()
            for seed in range(5):
                env = TabletopEnv(task=task)
                env.reset(seed=seed)
                states.add(env.get_text_state())
                env.close()
            with self.subTest(task=task):
                self.assertEqual(len(states), 5, f"{task}: seeds collapsed to one layout")

    def test_a_reused_env_also_varies(self):
        """The runner resets one env in a loop; it must not be stuck on seed 0."""
        from harness.envs.tabletop import TabletopEnv
        env = TabletopEnv(task="pick_place")
        states = set()
        for seed in range(5):
            env.reset(seed=seed)
            states.add(env.get_text_state())
        env.close()
        self.assertEqual(len(states), 5)

    def test_the_same_seed_is_still_reproducible(self):
        from harness.envs.tabletop import TabletopEnv
        env = TabletopEnv(task="stack")
        env.reset(seed=3)
        first = env.get_text_state()
        env.reset(seed=7)
        env.reset(seed=3)
        self.assertEqual(env.get_text_state(), first)
        env.close()

    def test_an_explicit_task_spec_is_never_silently_replaced(self):
        """Honouring a seed must not mean discarding the caller's own layout."""
        from harness.envs.tabletop import TabletopEnv
        from harness.tasks import generate_task
        spec = generate_task("pick_place", seed=99)
        env = TabletopEnv(task_spec=spec)
        env.reset(seed=0)
        before = env.get_text_state()
        env.reset(seed=12345)
        self.assertEqual(env.get_text_state(), before)
        self.assertIs(env.task_spec, spec)
        env.close()


class TestPairedOracleDenominator(unittest.TestCase):
    def test_the_oracle_scores_exactly_one_against_itself(self):
        """Against a per-task median the oracle's own ratio drifts below 1.0,
        which reads as "more efficient than optimal"."""
        from harness.eval.metrics import (
            oracle_steps_by_instance,
            oracle_steps_by_task,
            summarize_records,
        )
        recs = [{"env_name": "t", "policy": "oracle", "seed": s, "success": True,
                 "score": 1.0, "episode_step": steps}
                for s, steps in enumerate([4, 9, 20])]
        lb = summarize_records(recs, oracle_steps=oracle_steps_by_task(recs),
                               oracle_steps_per_instance=oracle_steps_by_instance(recs))
        self.assertEqual(lb["models"]["oracle"]["steps_vs_oracle"], 1.0)

    def test_it_falls_back_to_the_task_median_for_unpaired_instances(self):
        from harness.eval.metrics import (
            oracle_steps_by_instance,
            oracle_steps_by_task,
            summarize_records,
        )
        recs = [{"env_name": "t", "policy": "oracle", "seed": 0, "success": True,
                 "score": 1.0, "episode_step": 10},
                # seed 1 has no oracle run, so the model's ratio uses the median
                {"env_name": "t", "policy": "m", "seed": 1, "success": True,
                 "score": 1.0, "episode_step": 20}]
        lb = summarize_records(recs, oracle_steps=oracle_steps_by_task(recs),
                               oracle_steps_per_instance=oracle_steps_by_instance(recs))
        self.assertEqual(lb["models"]["m"]["steps_vs_oracle"], 2.0)
