import unittest

import numpy as np

from harness.envs.tabletop import TabletopEnv
from harness.tasks.base import TaskSpec
from harness.types import Action


def _spec(**kw):
    defaults = dict(kind="pick_place", description="d", objects=[], goals={}, ee_start=[0.1, 0.1], params={})
    defaults.update(kw)
    return TaskSpec(**defaults)


class TestTabletopEnv(unittest.TestCase):
    def test_query_api(self):
        spec = _spec(
            objects=[{"name": "cube", "pos": [0.3, 0.3], "target": "goal"}],
            goals={"goal": [0.8, 0.8]},
        )
        env = TabletopEnv(task_spec=spec)
        env.reset()
        self.assertEqual(env.list_objects(), ["cube"])
        self.assertEqual(env.list_goals(), ["goal"])
        np.testing.assert_allclose(env.get_object_pos("cube"), [0.3, 0.3])
        self.assertIsNone(env.get_object_pos("nope"))
        self.assertFalse(env.is_grasped())

    def test_stack_success(self):
        spec = _spec(
            kind="stack",
            objects=[
                {"name": "base", "pos": [0.4, 0.4], "role": "base"},
                {"name": "top", "pos": [0.4, 0.7], "role": "top"},
            ],
            params={"stack_radius": 0.1, "grasp_radius": 0.15},
        )
        env = TabletopEnv(task_spec=spec)
        env.reset()
        env.step(Action(kind="ee_pose", value=np.array([0.4, 0.7])))
        env.step(Action(kind="noop", gripper=1.0))
        self.assertEqual(env.grasped_object(), "top")
        env.step(Action(kind="ee_pose", value=np.array([0.4, 0.4])))
        r = env.step(Action(kind="noop", gripper=0.0))
        self.assertTrue(r.info["success"])

    def test_obstacle_collision_fails(self):
        spec = _spec(
            kind="pick_place_obstacle",
            objects=[{"name": "cube", "pos": [0.2, 0.2], "target": "goal"}],
            goals={"goal": [0.8, 0.8]},
            obstacles=[{"name": "wall", "pos": [0.5, 0.5], "radius": 0.1}],
            params={"goal_radius": 0.08},
        )
        env = TabletopEnv(task_spec=spec)
        env.reset()
        r = env.step(Action(kind="ee_pose", value=np.array([0.5, 0.5])))
        self.assertTrue(r.info["collided"])
        self.assertFalse(r.info["success"])
        self.assertTrue(r.terminated)

    def test_sort_requires_release(self):
        spec = _spec(
            kind="sort",
            objects=[
                {"name": "red", "pos": [0.2, 0.2], "target": "bin_red"},
                {"name": "green", "pos": [0.4, 0.4], "target": "bin_green"},
            ],
            goals={"bin_red": [0.8, 0.8], "bin_green": [0.6, 0.6]},
            params={"goal_radius": 0.1, "grasp_radius": 0.15, "require_release": True},
        )
        env = TabletopEnv(task_spec=spec)
        env.reset()
        env.step(Action(kind="ee_pose", value=np.array([0.2, 0.2])))
        env.step(Action(kind="noop", gripper=1.0))
        env.step(Action(kind="ee_pose", value=np.array([0.8, 0.8])))
        r = env.step(Action(kind="noop", gripper=1.0))  # still grasping red
        self.assertFalse(r.info["success"])
        env.step(Action(kind="noop", gripper=0.0))  # release red
        self.assertFalse(env.is_grasped())

    def test_render(self):
        env = TabletopEnv(task_spec=_spec(objects=[{"name": "cube", "pos": [0.5, 0.5], "target": "goal"}], goals={"goal": [0.8, 0.8]}), render_size=64)
        env.reset()
        img = env.render()
        self.assertEqual(img.shape, (64, 64, 3))


if __name__ == "__main__":
    unittest.main()
