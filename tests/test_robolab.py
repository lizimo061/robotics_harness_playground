import unittest


class TestRoboLabModule(unittest.TestCase):
    def test_lazy_import_without_robolab(self):
        # importing the module must NOT require RoboLab/IsaacLab (heavy imports
        # are deferred to __init__)
        import harness.envs.robolab as r

        self.assertEqual(r.RoboLabEnv.name, "robolab")

    def test_instantiation_raises_clear_error_without_robolab(self):
        import harness.envs.robolab as r

        with self.assertRaises(ImportError):
            r.RoboLabEnv(task="RubiksCubeTask")


if __name__ == "__main__":
    unittest.main()


class _FakeSpace:
    """RelIK: (dx, dy, dz, droll, dpitch, dyaw) + gripper."""

    shape = (7,)
    low = [-0.05] * 6 + [0.0]
    high = [0.05] * 6 + [1.0]


class _FakeAbsSpace:
    """AbsIK: (x, y, z, qw, qx, qy, qz) + gripper."""

    shape = (8,)
    low = [-2.0] * 7 + [0.0]
    high = [2.0] * 7 + [1.0]


class _FakeInner:
    """Stands in for the IsaacLab env: an action space and a device, no sim."""

    device = "cpu"

    def __init__(self, space=None):
        self.action_space = space or _FakeSpace()


def _env_for_action_tests(ee=(0.30, 0.00, 0.20), mode="ee_delta", space=None,
                          quat=(1.0, 0.0, 0.0, 0.0)):
    """A RoboLabEnv with its heavy __init__ bypassed.

    Constructing the real thing needs Isaac Sim, but _to_env_action is pure
    arithmetic on the action space and the current pose, which is exactly the
    part that was wrong.
    """
    import numpy as np

    import harness.envs.robolab as r

    env = r.RoboLabEnv.__new__(r.RoboLabEnv)
    env._env = _FakeInner(space)
    env._num_envs = 1
    env._action_mode = mode
    env._last_proprio = {"ee_pos": np.asarray(ee, dtype=np.float32),
                         "ee_quat": np.asarray(quat, dtype=np.float32)}
    env._instruction = "test instruction"
    env._step_idx = 0
    return env


class TestActionConversion(unittest.TestCase):
    """An absolute target must become a delta, or the agent cannot aim.

    Passing `move_to`'s absolute coordinates straight through made every one a
    command to jump most of a metre; RoboLab's controller saturates, so the arm
    lurched and the agent's failures looked like the model's rather than ours.
    """

    def test_an_absolute_move_becomes_a_clipped_delta(self):
        import numpy as np

        from harness.types import Action
        env = _env_for_action_tests(ee=(0.30, 0.0, 0.20))
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.43, -0.10, 0.03]))).ravel()
        # target - current = (+0.13, -0.10, -0.17), clipped to the +-0.05 limit
        self.assertAlmostEqual(float(out[0]), 0.05, places=5)
        self.assertAlmostEqual(float(out[1]), -0.05, places=5)
        self.assertAlmostEqual(float(out[2]), -0.05, places=5)

    def test_the_delta_points_at_the_target(self):
        import numpy as np

        from harness.types import Action
        env = _env_for_action_tests(ee=(0.30, 0.0, 0.20))
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.32, 0.01, 0.19]))).ravel()
        # inside the limit, so it should be the exact difference
        self.assertAlmostEqual(float(out[0]), 0.02, places=5)
        self.assertAlmostEqual(float(out[1]), 0.01, places=5)
        self.assertAlmostEqual(float(out[2]), -0.01, places=5)

    def test_a_relative_move_is_left_alone(self):
        import numpy as np

        from harness.types import Action
        env = _env_for_action_tests()
        out = np.asarray(env._to_env_action(
            Action(kind="ee_delta", value=[0.01, -0.02, 0.0]))).ravel()
        self.assertAlmostEqual(float(out[0]), 0.01, places=5)
        self.assertAlmostEqual(float(out[1]), -0.02, places=5)

    def test_the_gripper_occupies_the_last_dim(self):
        import numpy as np

        from harness.types import Action
        env = _env_for_action_tests()
        out = np.asarray(env._to_env_action(Action(kind="noop", gripper=1.0))).ravel()
        self.assertEqual(len(out), 7)
        self.assertAlmostEqual(float(out[-1]), 1.0, places=5)

    def test_an_absolute_move_without_a_known_pose_degrades_instead_of_crashing(self):
        import numpy as np

        from harness.types import Action
        env = _env_for_action_tests()
        env._last_proprio = {}
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.4, 0.1, 0.2]))).ravel()
        self.assertEqual(len(out), 7)

    def test_the_delta_limit_comes_from_the_action_space(self):
        env = _env_for_action_tests()
        import numpy as np
        clipped = env._clip_delta(np.asarray([10.0, -10.0, 10.0], dtype=np.float32), 7)
        self.assertTrue(all(abs(float(c)) <= 0.05 + 1e-6 for c in clipped))

    def test_relik_leaves_the_rotation_block_at_zero(self):
        """A RelIK action is (dx,dy,dz,droll,dpitch,dyaw): no unintended rotation."""
        import numpy as np

        from harness.types import Action
        env = _env_for_action_tests()
        out = np.asarray(env._to_env_action(
            Action(kind="ee_delta", value=[0.01, 0.0, 0.0]))).ravel()
        self.assertEqual(len(out), 7)
        self.assertTrue(all(abs(float(x)) < 1e-6 for x in out[3:6]), out)


class TestAbsIKLayout(unittest.TestCase):
    """AbsIK commands an absolute pose, so the quaternion slots must be valid."""

    def _env(self, **kw):
        return _env_for_action_tests(mode="ee_pose", space=_FakeAbsSpace(), **kw)

    def test_an_absolute_target_is_passed_through_not_differenced(self):
        import numpy as np

        from harness.types import Action
        env = self._env(ee=(0.30, 0.0, 0.20))
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.43, -0.10, 0.03]))).ravel()
        self.assertAlmostEqual(float(out[0]), 0.43, places=5)
        self.assertAlmostEqual(float(out[1]), -0.10, places=5)
        self.assertAlmostEqual(float(out[2]), 0.03, places=5)

    def test_the_orientation_block_is_a_unit_quaternion(self):
        import numpy as np

        from harness.types import Action
        env = self._env(quat=(0.0, 1.0, 0.0, 0.0))
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.4, 0.0, 0.2]))).ravel()
        quat = out[3:7]
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0, places=5)

    def test_a_missing_orientation_falls_back_to_identity_not_zeros(self):
        """Zero-filled quaternion slots are not a rotation; the IK diverges."""
        import numpy as np

        from harness.types import Action
        env = self._env()
        env._last_proprio = {"ee_pos": np.asarray([0.3, 0.0, 0.2], dtype=np.float32)}
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.4, 0.0, 0.2]))).ravel()
        self.assertAlmostEqual(float(np.linalg.norm(out[3:7])), 1.0, places=5)

    def test_a_relative_move_is_added_to_the_current_pose(self):
        import numpy as np

        from harness.types import Action
        env = self._env(ee=(0.30, 0.0, 0.20))
        out = np.asarray(env._to_env_action(
            Action(kind="ee_delta", value=[0.02, 0.0, -0.01]))).ravel()
        self.assertAlmostEqual(float(out[0]), 0.32, places=5)
        self.assertAlmostEqual(float(out[2]), 0.19, places=5)

    def test_a_pure_gripper_command_holds_position(self):
        import numpy as np

        from harness.types import Action
        env = self._env(ee=(0.31, 0.02, 0.18))
        out = np.asarray(env._to_env_action(Action(kind="noop", gripper=1.0))).ravel()
        self.assertEqual(len(out), 8)
        self.assertAlmostEqual(float(out[0]), 0.31, places=5)
        self.assertAlmostEqual(float(out[-1]), 1.0, places=5)


class TestRegistrationSelection(unittest.TestCase):
    """The action mode selects the registration; a mismatch fails silently."""

    def test_each_mode_maps_to_its_own_suffix(self):
        import harness.envs.robolab as r
        self.assertEqual(r.RoboLabEnv._REGISTRARS["ee_pose"][2], "AbsIK")
        self.assertEqual(r.RoboLabEnv._REGISTRARS["ee_delta"][2], "RelIK")
        self.assertEqual(r.RoboLabEnv._REGISTRARS["joint_position"][2], "")

    def test_an_unknown_mode_is_rejected(self):
        import harness.envs.robolab as r
        with self.assertRaises(ValueError):
            r.RoboLabEnv._registrar_for("cartesian_twist")

    def test_the_flavour_suffix_wins_over_a_bare_name_match(self):
        """A bare name matches several flavours once more than one is registered."""
        import harness.envs.robolab as r
        calls = []

        def get_envs(task=None):
            calls.append(task)
            return ["RubiksCubeTask", "RubiksCubeTaskAbsIK", "RubiksCubeTaskRelIK"]

        self.assertEqual(
            r.RoboLabEnv._resolve_task(get_envs, "RubiksCubeTask", "AbsIK"),
            "RubiksCubeTaskAbsIK")

    def test_a_missing_task_still_raises(self):
        import harness.envs.robolab as r
        with self.assertRaises(ValueError):
            r.RoboLabEnv._resolve_task(lambda task=None: [], "NopeTask", "AbsIK")


class TestClippingFeedback(unittest.TestCase):
    """Saturation the agent cannot observe is a dead end, not a control loop."""

    def test_a_capped_move_is_reported_in_the_text_state(self):
        from harness.types import Action
        env = _env_for_action_tests(ee=(0.30, 0.0, 0.20))
        env._to_env_action(Action(kind="ee_pose", value=[0.90, 0.0, 0.20]))
        text = env.get_text_state()
        self.assertIn("per-step limit", text)
        self.assertIn("remains", text)

    def test_a_move_within_the_limit_reports_nothing(self):
        from harness.types import Action
        env = _env_for_action_tests(ee=(0.30, 0.0, 0.20))
        env._to_env_action(Action(kind="ee_pose", value=[0.31, 0.0, 0.20]))
        self.assertNotIn("per-step limit", env.get_text_state())
