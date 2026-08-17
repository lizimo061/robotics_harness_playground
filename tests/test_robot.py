import unittest

from harness.robot import get_robot_spec
from harness.robot.actions import move_delta, move_to, set_gripper, set_joint_positions


class TestRobotSpecs(unittest.TestCase):
    def test_panda_spec(self):
        panda = get_robot_spec("panda")
        self.assertEqual(panda.dof, 7)
        self.assertEqual(panda.total_dof, 9)
        self.assertEqual(len(panda.joint_names), 7)

    def test_ur5e_spec(self):
        ur = get_robot_spec("ur5e")
        self.assertEqual(ur.dof, 6)

    def test_unknown_robot(self):
        with self.assertRaises(KeyError):
            get_robot_spec("nope")


class TestActions(unittest.TestCase):
    def test_constructors(self):
        self.assertEqual(move_delta(0.1, 0.2).kind, "ee_delta")
        self.assertEqual(move_to(0.1, 0.2).kind, "ee_pose")
        self.assertEqual(set_gripper(1.0).gripper, 1.0)
        self.assertEqual(len(set_joint_positions([1, 2, 3]).value), 3)


if __name__ == "__main__":
    unittest.main()
