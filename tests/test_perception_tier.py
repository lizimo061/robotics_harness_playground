"""P3: perception tools and the scaffolding tier switch.

The point of this phase is to turn a caveat into a number. Every RoboLab and tabletop
result in this repo so far was measured with ground-truth object poses handed to the
agent as text -- CaP-X's most-scaffolded tier. Its finding is that scores depend on
that scaffolding, so the harness needs to be able to run the same task both ways.

Tests here cover: the detector seam, the tools that use it, and the tier filter. What
they deliberately do *not* claim is that OracleDetector is a perception model -- it is
ground truth with optional noise, useful for plumbing and for sensitivity, and the
`source` field says so on every detection.
"""
import unittest

import numpy as np

from harness.envs.tabletop import TabletopEnv
from harness.perception.detect import (
    Detection,
    Detector,
    OracleDetector,
    RemoteDetector,
    get_detector,
)
from harness.tools.perception_tools import DetectTool, PointAtTool
from harness.tools.registry import TIERS, get_default_tools, tools_for_tier


def _env():
    env = TabletopEnv(task="pick_place")
    env.reset(seed=0)
    return env


class TestDetectorSeam(unittest.TestCase):
    def test_the_oracle_detector_satisfies_the_protocol(self):
        self.assertIsInstance(OracleDetector(), Detector)

    def test_the_remote_detector_satisfies_the_protocol(self):
        self.assertIsInstance(RemoteDetector("http://localhost:1"), Detector)

    def test_it_finds_an_object_by_partial_name(self):
        env = _env()
        found = OracleDetector(env).detect(env.render(), "cube")
        env.close()
        self.assertTrue(found)
        self.assertEqual(found[0].name, "cube")
        self.assertIsNotNone(found[0].position)

    def test_a_miss_returns_nothing_rather_than_raising(self):
        env = _env()
        self.assertEqual(OracleDetector(env).detect(env.render(), "unicorn"), [])
        env.close()

    def test_an_empty_query_finds_nothing(self):
        env = _env()
        self.assertEqual(OracleDetector(env).detect(env.render(), ""), [])
        env.close()

    def test_every_detection_says_where_it_came_from(self):
        """A report has to be able to state what produced a number."""
        env = _env()
        found = OracleDetector(env).detect(env.render(), "cube")
        env.close()
        self.assertEqual(found[0].source, "oracle")

    def test_noise_is_labelled_in_the_source(self):
        d = OracleDetector(noise_m=0.02)
        self.assertIn("noise=0.02", d.name)

    def test_no_env_bound_means_no_detections(self):
        self.assertEqual(OracleDetector(None).detect(None, "cube"), [])

    def test_bind_attaches_an_env_later(self):
        env = _env()
        d = OracleDetector()
        self.assertEqual(d.detect(env.render(), "cube"), [])
        d.bind(env)
        self.assertTrue(d.detect(env.render(), "cube"))
        env.close()


class TestDegradation(unittest.TestCase):
    """Noise and dropout exist to measure how much a task leans on exact poses."""

    def test_noise_moves_the_reported_position(self):
        env = _env()
        truth = np.asarray(env.get_object_pos("cube"), dtype=float).ravel()
        noisy = OracleDetector(env, noise_m=0.05, seed=1).detect(env.render(), "cube")
        env.close()
        got = np.asarray(noisy[0].position, dtype=float).ravel()
        self.assertGreater(float(np.linalg.norm(got[:2] - truth[:2])), 1e-6)

    def test_noise_is_reproducible_for_a_seed(self):
        env = _env()
        a = OracleDetector(env, noise_m=0.05, seed=7).detect(env.render(), "cube")
        b = OracleDetector(env, noise_m=0.05, seed=7).detect(env.render(), "cube")
        env.close()
        self.assertTrue(np.allclose(np.asarray(a[0].position, dtype=float),
                                    np.asarray(b[0].position, dtype=float)))

    def test_full_dropout_never_detects(self):
        env = _env()
        d = OracleDetector(env, dropout=1.0, seed=0)
        self.assertEqual(d.detect(env.render(), "cube"), [])
        env.close()

    def test_defaults_are_exactly_the_privileged_tier(self):
        env = _env()
        truth = np.asarray(env.get_object_pos("cube"), dtype=float).ravel()
        got = OracleDetector(env).detect(env.render(), "cube")[0].position
        env.close()
        self.assertTrue(np.allclose(np.asarray(got, dtype=float).ravel(), truth))


class TestPerceptionTools(unittest.TestCase):
    def test_detect_reports_what_it_found(self):
        env = _env()
        r = DetectTool(OracleDetector(env)).run(env, query="cube")
        env.close()
        self.assertIn("cube", r.feedback)
        self.assertIsNone(r.action, "a query tool must not step the env")

    def test_detect_reports_a_miss_as_information(self):
        env = _env()
        r = DetectTool(OracleDetector(env)).run(env, query="teapot")
        env.close()
        self.assertIn("not detected", r.feedback)

    def test_detect_without_a_query_asks_for_one(self):
        env = _env()
        r = DetectTool(OracleDetector(env)).run(env, query="")
        env.close()
        self.assertIn("needs a 'query'", r.feedback)

    def test_point_at_returns_a_pixel(self):
        env = _env()
        r = PointAtTool(OracleDetector(env)).run(env, query="cube")
        env.close()
        self.assertIn("pixel", r.feedback)

    def test_point_at_a_missing_thing_says_so(self):
        env = _env()
        r = PointAtTool(OracleDetector(env)).run(env, query="teapot")
        env.close()
        self.assertIn("cannot point at", r.feedback)

    def test_a_render_failure_does_not_kill_the_call(self):
        class _Blind(TabletopEnv):
            def render(self):
                raise RuntimeError("no camera")

        env = _Blind(task="pick_place")
        env.reset(seed=0)
        r = DetectTool(OracleDetector(env)).run(env, query="cube")
        env.close()
        self.assertTrue(r.feedback)  # answered from state; no exception escaped


class TestTierSwitch(unittest.TestCase):
    def test_privileged_keeps_the_ground_truth_queries(self):
        names = [t.name for t in tools_for_tier(get_default_tools(), "privileged")]
        self.assertIn("get_object_position", names)
        self.assertNotIn("detect", names)

    def test_perception_removes_them_and_adds_looking(self):
        names = [t.name for t in tools_for_tier(get_default_tools(), "perception",
                                                OracleDetector())]
        for privileged in ("get_object_position", "list_objects", "list_goals",
                           "list_obstacles"):
            self.assertNotIn(privileged, names)
        self.assertIn("detect", names)
        self.assertIn("point_at", names)

    def test_motion_is_identical_across_tiers(self):
        """The variable under study is what the agent can know, not what it can do."""
        motion = {"move_to", "move_delta", "grasp", "release", "set_joints", "done"}
        priv = {t.name for t in tools_for_tier(get_default_tools(), "privileged")}
        perc = {t.name for t in tools_for_tier(get_default_tools(), "perception",
                                               OracleDetector())}
        self.assertEqual(motion & priv, motion & perc)
        self.assertTrue(motion <= priv)

    def test_perception_without_a_detector_is_refused_loudly(self):
        with self.assertRaises(ValueError) as ctx:
            tools_for_tier(get_default_tools(), "perception")
        self.assertIn("needs a detector", str(ctx.exception))

    def test_an_unknown_tier_is_refused(self):
        with self.assertRaises(ValueError):
            tools_for_tier(get_default_tools(), "s7")

    def test_the_tier_list_is_the_documented_one(self):
        self.assertEqual(TIERS, ("privileged", "perception"))


class TestDetectorFactory(unittest.TestCase):
    def test_none_means_no_detector(self):
        self.assertIsNone(get_detector(None))

    def test_a_bare_string_builds_the_oracle(self):
        self.assertIsInstance(get_detector("oracle"), OracleDetector)

    def test_a_dict_passes_options_through(self):
        d = get_detector({"type": "oracle", "noise_m": 0.03, "seed": 2})
        self.assertIn("noise=0.03", d.name)

    def test_remote_is_built_from_a_base_url(self):
        d = get_detector({"type": "remote", "base_url": "http://x:8100"})
        self.assertIsInstance(d, RemoteDetector)

    def test_an_unknown_type_is_refused(self):
        with self.assertRaises(ValueError):
            get_detector({"type": "sam9000"})


class TestRemoteDetectorIsSurvivable(unittest.TestCase):
    def test_an_unreachable_detector_returns_nothing_rather_than_raising(self):
        """A detector outage must not be scored as a task failure mid-episode."""
        d = RemoteDetector("http://127.0.0.1:1", timeout=0.05)
        self.assertEqual(d.detect(np.zeros((4, 4, 3), dtype=np.uint8), "cube"), [])

    def test_describe_handles_a_pixel_only_detection(self):
        det = Detection(name="cube", pixel=(10, 20), confidence=0.5, source="remote")
        text = det.describe()
        self.assertIn("pixel", text)
        self.assertIn("cube", text)


if __name__ == "__main__":
    unittest.main()
