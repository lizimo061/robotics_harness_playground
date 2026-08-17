import unittest

import numpy as np

from harness.envs.kitchen import KitchenEnv
from harness.tasks import generate_task
from harness.types import Action


class TestKitchenEnv(unittest.TestCase):
    def _env(self):
        return KitchenEnv(task_spec=generate_task("cook_bread", seed=1))

    def test_query_api(self):
        env = self._env()
        env.reset()
        self.assertEqual(env.list_objects(), ["bread"])
        self.assertEqual(env.list_containers(), ["oven"])
        self.assertEqual(env.list_buttons(), ["button"])
        self.assertFalse(env.is_container_open("oven"))
        self.assertFalse(env.is_button_pressed("button"))

    def test_subgoal_checks_initial(self):
        env = self._env()
        env.reset()
        self.assertFalse(env.check_subgoal("oven_open"))
        self.assertFalse(env.check_subgoal("bread_in_oven"))
        self.assertFalse(env.check_subgoal("button_pressed"))

    def test_manual_solution(self):
        env = self._env()
        env.reset()
        # open the oven
        door = env.get_door_pos("oven")
        env.step(Action(kind="ee_pose", value=np.asarray(door)))
        env.step(Action(kind="noop", gripper=1.0))
        self.assertTrue(env.is_container_open("oven"))
        # put bread in
        bread = env.get_object_pos("bread")
        env.step(Action(kind="ee_pose", value=np.asarray(bread)))
        env.step(Action(kind="noop", gripper=1.0))
        self.assertEqual(env.grasped_object(), "bread")
        interior = env.get_container_interior("oven")
        env.step(Action(kind="ee_pose", value=np.asarray(interior)))
        env.step(Action(kind="noop", gripper=0.0))
        self.assertTrue(env.check_subgoal("bread_in_oven"))
        # press button
        button = env.get_button_pos("button")
        env.step(Action(kind="ee_pose", value=np.asarray(button)))
        env.step(Action(kind="noop", gripper=1.0))
        self.assertTrue(env.check_subgoal("button_pressed"))
        self.assertTrue(env.is_success())


if __name__ == "__main__":
    unittest.main()
