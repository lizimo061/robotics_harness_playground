import unittest

import numpy as np

from harness.types import Action, ActionSpace, Episode, Obs, StepResult


class TestActionSpace(unittest.TestCase):
    def test_clip(self):
        sp = ActionSpace(kind="ee_delta", dim=2, low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
        np.testing.assert_allclose(sp.clip([5.0, -5.0]), [1.0, -1.0])

    def test_to_text(self):
        sp = ActionSpace(kind="joint_position", dim=2, joint_names=("j0", "j1"))
        self.assertIn("joint_position", sp.to_text())
        self.assertIn("j0", sp.to_text())


class TestContainers(unittest.TestCase):
    def test_obs_to_text(self):
        obs = Obs(state=np.array([0.1, 0.2]), text="hello")
        self.assertIn("hello", obs.to_text())

    def test_obs_to_text_with_names(self):
        obs = Obs(state=np.array([1.0, 2.0]))
        text = obs.to_text(("a", "b"))
        self.assertIn("a=1", text)
        self.assertIn("b=2", text)

    def test_step_result(self):
        sr = StepResult(obs=Obs(), terminated=True, info={"success": True})
        self.assertTrue(sr.done)
        self.assertTrue(sr.success)

    def test_episode_steps(self):
        ep = Episode()
        self.assertEqual(ep.steps, 0)
        ep.actions.append(Action(kind="move"))
        self.assertEqual(ep.steps, 1)


if __name__ == "__main__":
    unittest.main()
