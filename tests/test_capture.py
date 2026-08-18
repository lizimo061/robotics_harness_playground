import unittest

import numpy as np

from harness.envs.tabletop import TabletopEnv
from harness.types import Action
from harness.viz.capture import FrameCapture


class _Boom:
    """An env whose render() raises, to prove capture never kills a run."""

    name = "boom"

    def reset(self, **kw):
        return {}

    def step(self, action):
        return type("R", (), {"obs": {}, "reward": 0.0, "success": False,
                              "done": False, "info": {}})()

    def render(self):
        raise RuntimeError("no display")

    def close(self):
        pass


class TestFrameCapture(unittest.TestCase):
    def _env(self):
        return FrameCapture(TabletopEnv(task="pick_place"))

    def test_captures_one_frame_per_env_step_plus_the_initial_state(self):
        env = self._env()
        env.reset(seed=0)
        for _ in range(3):
            env.step(Action(kind="ee_delta", value=[0.05, 0.0]))
        self.assertEqual(len(env.frames), 4)  # initial + 3 steps
        env.close()

    def test_frames_are_images_and_actually_differ(self):
        env = self._env()
        env.reset(seed=0)
        for _ in range(4):
            env.step(Action(kind="ee_delta", value=[0.08, 0.02]))
        self.assertTrue(all(isinstance(f, np.ndarray) and f.ndim == 3 for f in env.frames))
        self.assertGreater(len({f.tobytes() for f in env.frames}), 1,
                           "every frame identical: is the arm actually moving?")
        env.close()

    def test_every_n_subsamples(self):
        env = FrameCapture(TabletopEnv(task="pick_place"), every=3)
        env.reset(seed=0)
        for _ in range(9):
            env.step(Action(kind="ee_delta", value=[0.02, 0.0]))
        self.assertEqual(len(env.frames), 1 + 3)
        env.close()

    def test_clear_resets_between_episodes(self):
        env = self._env()
        env.reset(seed=0)
        env.step(Action(kind="ee_delta", value=[0.05, 0.0]))
        env.clear()
        self.assertEqual(env.frames, [])
        env.reset(seed=1)
        self.assertEqual(len(env.frames), 1)
        env.close()

    def test_the_frame_cap_is_enforced(self):
        env = FrameCapture(TabletopEnv(task="pick_place"), max_frames=3)
        env.reset(seed=0)
        for _ in range(10):
            env.step(Action(kind="ee_delta", value=[0.01, 0.0]))
        self.assertEqual(len(env.frames), 3)
        env.close()

    def test_a_failing_render_is_survivable(self):
        env = FrameCapture(_Boom())
        env.reset()
        env.step(Action(kind="noop"))
        self.assertEqual(env.frames, [])  # no frames, but no exception either

    def test_the_query_api_the_tools_use_still_works_through_the_wrapper(self):
        """Tools call env.list_objects()/get_object_position(); delegation must hold."""
        env = self._env()
        env.reset(seed=0)
        self.assertTrue(env.list_objects())
        self.assertIsNotNone(env.get_object_pos(env.list_objects()[0]))
        self.assertIn("Task:", env.get_text_state())
        self.assertEqual(env.name, "tabletop")
        self.assertIsNotNone(env.action_space)
        env.close()

    def test_an_agent_runs_unchanged_through_the_wrapper(self):
        from harness.agent.baselines import get_baseline_agent
        env = self._env()
        ep = get_baseline_agent("oracle", max_steps=300).run(env, seed=0)
        self.assertTrue(ep.success, "the oracle must still solve a wrapped env")
        self.assertGreater(len(env.frames), 1)
        env.close()


if __name__ == "__main__":
    unittest.main()


class TestBlankFrames(unittest.TestCase):
    """Isaac's first frame after reset is often black before the renderer settles."""

    class _Warmup:
        name = "warmup"

        def __init__(self):
            self.n = 0

        def reset(self, **kw):
            self.n = 0
            return {}

        def step(self, action):
            self.n += 1
            return type("R", (), {"obs": {}, "reward": 0.0, "success": False,
                                  "done": False, "info": {}})()

        def render(self):
            # first frame black, then a real image
            v = 0 if self.n == 0 else 200
            return np.full((4, 4, 3), v, dtype=np.uint8)

        def close(self):
            pass

    def test_a_leading_blank_frame_is_dropped_when_asked(self):
        env = FrameCapture(self._Warmup(), skip_blank=True)
        env.reset()
        self.assertEqual(env.frames, [])
        env.step(Action(kind="noop"))
        self.assertEqual(len(env.frames), 1)
        self.assertEqual(int(env.frames[0].max()), 200)

    def test_it_is_off_by_default(self):
        env = FrameCapture(self._Warmup())
        env.reset()
        self.assertEqual(len(env.frames), 1)

    def test_only_leading_frames_are_dropped(self):
        """A genuinely dark frame mid-episode is data, not an artifact."""
        env = FrameCapture(self._Warmup(), skip_blank=True)
        env.reset()
        env.step(Action(kind="noop"))          # real frame, now frames is non-empty
        env.env.n = 0                          # force render() black again
        env.step(Action(kind="noop"))
        self.assertEqual(len(env.frames), 2)
