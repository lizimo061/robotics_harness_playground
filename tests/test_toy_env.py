import unittest

import numpy as np

from harness.envs.toy import ToyTabletopEnv
from harness.types import Action


class TestToyTabletop(unittest.TestCase):
    def test_reset_state(self):
        env = ToyTabletopEnv(seed=0)
        obs = env.reset()
        self.assertEqual(tuple(obs.state.shape), (8,))
        self.assertIn("Object", obs.text)
        self.assertIn("Goal", obs.text)

    def test_action_space(self):
        env = ToyTabletopEnv(seed=0)
        self.assertEqual(env.action_space.kind, "ee_delta")
        self.assertEqual(env.action_space.dim, 2)
        self.assertEqual(env.action_space.gripper_dim, 1)

    def test_delta_is_clipped(self):
        env = ToyTabletopEnv(seed=0)
        env.reset()
        r = env.step(Action(kind="ee_delta", value=np.array([10.0, 0.0])))
        # ee_x: 0.1 + 0.25 (max delta) = 0.35
        self.assertAlmostEqual(float(r.obs.state[0]), 0.35, places=5)

    def test_grasp_release_and_success(self):
        env = ToyTabletopEnv(seed=0, obj_pos=[0.5, 0.5], goal_pos=[0.7, 0.5], ee_pos=[0.5, 0.5])
        env.reset()
        # close gripper while on the object -> grasp
        r1 = env.step(Action(kind="ee_delta", value=np.array([0.0, 0.0]), gripper=1.0))
        self.assertTrue(bool(r1.obs.state[5]))  # grasped flag
        # move the grasped object to the goal
        r2 = env.step(Action(kind="ee_delta", value=np.array([0.2, 0.0]), gripper=1.0))
        self.assertTrue(r2.info.get("success"))
        self.assertTrue(r2.terminated)

    def test_render(self):
        env = ToyTabletopEnv(seed=0, render_size=64)
        env.reset()
        img = env.render()
        self.assertEqual(img.shape, (64, 64, 3))
        self.assertEqual(img.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
