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
    # zero by default so each test below isolates one behaviour; the
    # tool-centre-point conversion has its own class.
    env._tcp_offset = 0.0
    env._quat_setpoint = None
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


class TestToolCentrePoint(unittest.TestCase):
    """The IK drives the gripper flange; the fingertips are 162.8mm below it.

    Commanding a grasp at an object's own z therefore drives the fingers through
    the table, the IK saturates against it, and nothing is ever picked up -- with
    no error anywhere. Tool coordinates are fingertip space; the adapter converts.
    """

    def _env(self, **kw):
        env = _env_for_action_tests(mode="ee_pose", space=_FakeAbsSpace(), **kw)
        env._tcp_offset = 0.1628
        return env

    def _env_down(self, **kw):
        """Wrist hanging straight down, expressed in eef-frame terms.

        _current_quat() reports the eef frame, so the fixture's base_link
        quaternion is chosen such that the eef frame comes out as (0,1,0,0).
        """
        import numpy as np

        import harness.envs.robolab as r
        base = r._quat_mul(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                           r._quat_inv(r._EEF_OFFSET_ROT))
        return self._env(quat=tuple(float(x) for x in base), **kw)

    def test_a_grasp_target_is_raised_to_the_flange(self):
        import numpy as np

        from harness.types import Action
        env = self._env_down(ee=(0.30, 0.0, 0.40))
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.43, -0.10, 0.034]))).ravel()
        self.assertAlmostEqual(float(out[2]), 0.034 + 0.1628, places=4)
        self.assertAlmostEqual(float(out[0]), 0.43, places=5)  # x, y unaffected

    def test_the_offset_follows_the_wrist_orientation(self):
        """The offset is fixed in the gripper's frame, not the world's.

        With the wrist rotated 90 degrees about y, the fingertips are displaced
        along +x, so a fixed world -z correction would aim the grasp elsewhere.
        """
        import numpy as np

        from harness.types import Action
        # eef frame rotated 90 degrees about y: the fingertips lie along +x
        import harness.envs.robolab as r
        base = r._quat_mul(np.array([0.7071, 0.0, 0.7071, 0.0], dtype=np.float32),
                           r._quat_inv(r._EEF_OFFSET_ROT))
        env = self._env(ee=(0.30, 0.0, 0.40), quat=tuple(float(x) for x in base))
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.43, 0.0, 0.20]))).ravel()
        self.assertAlmostEqual(float(out[0]), 0.43 - 0.1628, places=3)
        self.assertAlmostEqual(float(out[2]), 0.20, places=3)

    def test_the_reported_pose_is_the_fingertip(self):
        import numpy as np
        env = self._env_down(ee=(0.30, 0.0, 0.40))
        tip = np.asarray(env.get_ee_pos()).ravel()
        self.assertAlmostEqual(float(tip[2]), 0.40 - 0.1628, places=4)
        flange = np.asarray(env.get_flange_pos()).ravel()
        self.assertAlmostEqual(float(flange[2]), 0.40, places=5)

    def test_commands_and_feedback_share_one_frame(self):
        """Aim at where you already are; the command must be a no-op in z."""
        import numpy as np

        from harness.types import Action
        env = self._env_down(ee=(0.30, 0.0, 0.40))
        here = np.asarray(env.get_ee_pos()).ravel()
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=here))).ravel()
        self.assertAlmostEqual(float(out[2]), 0.40, places=4)

    def test_a_zero_offset_disables_the_conversion(self):
        import numpy as np

        from harness.types import Action
        env = self._env_down(ee=(0.30, 0.0, 0.40))
        env._tcp_offset = 0.0
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.43, 0.0, 0.034]))).ravel()
        self.assertAlmostEqual(float(out[2]), 0.034, places=5)


class TestOrientationControl(unittest.TestCase):
    """Two orientation problems, both fatal to grasping.

    The tools could not command an orientation at all, so the wrist kept whatever
    pose it started in -- pointing along +x on these tasks, which cannot close on
    an object lying on a table however accurately it reaches. And the commanded
    orientation was read back from the measurement each step, which compounds the
    differential IK's own orientation drift (RoboLab's DroidIKActionCfg docstring
    warns about it): measured drift went from (0.707,0,0.707,0) at reset to
    (-0.81,-0.08,-0.57,0.10) within a few dozen steps.
    """

    def _env(self, quat=(0.7071, 0.0, 0.7071, 0.0)):
        env = _env_for_action_tests(mode="ee_pose", space=_FakeAbsSpace(), quat=quat)
        return env

    def test_a_seven_vector_commands_the_orientation(self):
        import numpy as np

        import harness.envs.robolab as r
        from harness.types import Action
        env = self._env()
        want = [0.0, 1.0, 0.0, 0.0]
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.4, 0.0, 0.2] + want))).ravel()
        # sent in base_link terms, i.e. un-offset from the eef frame
        expected = r._quat_mul(np.asarray(want, dtype=np.float32),
                               r._quat_inv(r._EEF_OFFSET_ROT))
        for got, exp in zip(out[3:7], expected):
            self.assertAlmostEqual(float(got), float(exp), places=5)

    def test_the_setpoint_is_held_not_read_back_from_the_measurement(self):
        import numpy as np

        from harness.types import Action
        env = self._env()
        env._to_env_action(Action(kind="ee_pose",
                                  value=[0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0]))
        # the arm drifts...
        env._last_proprio["ee_quat"] = np.asarray([-0.81, -0.08, -0.57, 0.10],
                                                 dtype=np.float32)
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.4, 0.0, 0.2]))).ravel()
        # ...and the command still asks for the orientation we set
        import harness.envs.robolab as r
        expected = r._quat_mul(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                               r._quat_inv(r._EEF_OFFSET_ROT))
        for got, exp in zip(out[3:7], expected):
            self.assertAlmostEqual(float(got), float(exp), places=5)

    def test_the_setpoint_seeds_from_the_pose_when_never_commanded(self):
        import numpy as np

        from harness.types import Action
        env = self._env(quat=(0.7071, 0.0, 0.7071, 0.0))
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.4, 0.0, 0.2]))).ravel()
        # seeded from the measured pose, so it must send that pose back unchanged
        for got, exp in zip(out[3:7], (0.7071, 0.0, 0.7071, 0.0)):
            self.assertAlmostEqual(float(got), float(exp), places=3)

    def test_a_non_unit_quaternion_is_ignored_rather_than_sent(self):
        import numpy as np

        from harness.types import Action
        env = self._env()
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0]))).ravel()
        self.assertAlmostEqual(float(np.linalg.norm(out[3:7])), 1.0, places=4)

    def test_the_grasp_orientation_points_the_approach_axis_down(self):
        import numpy as np

        import harness.envs.robolab as r
        env = self._env()
        down = env.grasp_orientation()
        approach = r._rotate_by_quat([0.0, 0.0, 1.0], down)
        self.assertAlmostEqual(float(approach[2]), -1.0, places=4)


class TestCameraChoice(unittest.TestCase):
    """A wrist recording shows the scene sliding past with the robot never in it.

    RoboLab documents the mirrored/egocentric camera as the one intended for video
    recording, so the extractor ranks views rather than taking the first it finds:
    footage that never images the arm cannot support any claim about what the arm
    did.
    """

    def _obs(self, *names):
        import numpy as np
        # distinct fill per camera so we can tell which one was picked
        return {n: np.full((40, 40, 3), i + 1, dtype=np.uint8)
                for i, n in enumerate(names)}

    def _pick(self, obs):
        env = _env_for_action_tests()
        env._image_source = None
        return env._extract_image(obs)

    def test_the_recording_view_beats_an_over_shoulder_view(self):
        obs = self._obs("over_shoulder_left_camera", "egocentric_mirrored_camera")
        self.assertEqual(int(self._pick(obs)[0, 0, 0]), 2)

    def test_any_scene_view_beats_the_wrist_camera(self):
        obs = self._obs("wrist_camera", "over_shoulder_left_camera")
        self.assertEqual(int(self._pick(obs)[0, 0, 0]), 2)

    def test_the_wrist_camera_is_used_when_it_is_all_there_is(self):
        obs = self._obs("wrist_camera")
        self.assertIsNotNone(self._pick(obs))

    def test_a_viewport_wins_outright(self):
        obs = self._obs("egocentric_mirrored_camera", "viewport_cam")
        self.assertEqual(int(self._pick(obs)[0, 0, 0]), 2)

    def test_non_image_tensors_are_ignored(self):
        import numpy as np
        obs = {"proprio": np.zeros((7,), dtype=np.float32),
               "over_shoulder_left_camera": np.full((40, 40, 3), 9, dtype=np.uint8)}
        self.assertEqual(int(self._pick(obs)[0, 0, 0]), 9)


class TestEndEffectorFrameConversion(unittest.TestCase):
    """RoboLab's IK tracks base_link, but poses are expressed in the eef frame.

    The two frames share an origin and differ by a fixed rotation
    (EEF_OFFSET_ROT), so a target must be un-offset before it is sent:
    action_quat = target_eef_quat (x) R_offset^-1, exactly as RoboLab's own
    run_abs_ik_demo.py does. Skipping the conversion fails silently -- it points
    the gripper somewhere else, which is why a commanded top-down grasp closed
    beside the object at every approach depth tried.
    """

    def test_the_conversion_round_trips(self):
        import numpy as np

        import harness.envs.robolab as r
        base = np.array([0.7071, 0.0, 0.7071, 0.0], dtype=np.float32)
        eef = r._quat_mul(base, r._EEF_OFFSET_ROT)
        back = r._quat_mul(eef, r._quat_inv(r._EEF_OFFSET_ROT))
        self.assertTrue(np.allclose(base, back, atol=1e-5))

    def test_a_commanded_orientation_is_un_offset_before_sending(self):
        import numpy as np

        import harness.envs.robolab as r
        from harness.types import Action
        env = _env_for_action_tests(mode="ee_pose", space=_FakeAbsSpace())
        want_eef = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose",
                   value=np.concatenate([[0.4, 0.0, 0.2], want_eef])))).ravel()
        expected = r._quat_mul(want_eef, r._quat_inv(r._EEF_OFFSET_ROT))
        for got, exp in zip(out[3:7], expected):
            self.assertAlmostEqual(float(got), float(exp), places=5)

    def test_a_pose_read_back_can_be_commanded_unchanged(self):
        """Both directions must use the same frame or aiming is impossible."""
        import numpy as np

        import harness.envs.robolab as r
        from harness.types import Action
        env = _env_for_action_tests(mode="ee_pose", space=_FakeAbsSpace(),
                                    quat=(0.7071, 0.0, 0.7071, 0.0))
        here = env._current_quat()                     # eef-frame terms
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose",
                   value=np.concatenate([[0.4, 0.0, 0.2], here])))).ravel()
        # what we send must be the base_link orientation the arm already has
        for got, exp in zip(out[3:7], [0.7071, 0.0, 0.7071, 0.0]):
            self.assertAlmostEqual(float(got), float(exp), places=4)

    def test_the_sent_quaternion_stays_unit_length(self):
        import numpy as np

        from harness.types import Action
        env = _env_for_action_tests(mode="ee_pose", space=_FakeAbsSpace())
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.4, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0]))).ravel()
        self.assertAlmostEqual(float(np.linalg.norm(out[3:7])), 1.0, places=5)


class TestMeasuredGraspOffset(unittest.TestCase):
    """The grasp offset is measured, not taken from the gripper's spec sheet.

    RoboLab cites 162.8mm flange-to-fingertip for the Robotiq 2F-85. Sweeping the
    flange height directly found the grasp ~3.7cm lower: flange z=0.160 lifted a
    cube at z=0.034 by 78mm, while z=0.140 collided with its top. Using the spec
    figure closed the gripper just above the object every time.
    """

    def test_the_default_matches_the_measured_grasp_height(self):
        import inspect

        import harness.envs.robolab as r
        default = inspect.signature(r.RoboLabEnv.__init__).parameters["tcp_offset"].default
        self.assertAlmostEqual(default, 0.126, places=3)

    def test_a_target_at_an_object_puts_the_flange_at_the_grasp_height(self):
        import numpy as np

        import harness.envs.robolab as r
        from harness.types import Action
        # wrist down, expressed in eef-frame terms
        base = r._quat_mul(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                           r._quat_inv(r._EEF_OFFSET_ROT))
        env = _env_for_action_tests(mode="ee_pose", space=_FakeAbsSpace(),
                                    ee=(0.30, 0.0, 0.40),
                                    quat=tuple(float(x) for x in base))
        env._tcp_offset = 0.126
        out = np.asarray(env._to_env_action(
            Action(kind="ee_pose", value=[0.431, -0.097, 0.034]))).ravel()
        # the measured grasp height for a cube at 0.034
        self.assertAlmostEqual(float(out[2]), 0.160, places=3)


class TestSuccessSignalSource(unittest.TestCase):
    """RoboLab keeps success in the TERMINATION MANAGER, not the info dict.

    A benchmark task declares e.g.
        success = DoneTerm(func=object_in_container, params={...})
    and IsaacLab exposes the result via termination_manager.get_term(name).
    extras["log"]["Episode_Termination/success"] is an episode-averaged statistic,
    not this step's outcome, so reading info keys returned False for every episode
    regardless of what happened -- the scripted probe carried the cube 0.22m into
    the bowl, get_term("success") was True, and the adapter still reported failure.
    """

    class _Manager:
        def __init__(self, terms):
            self._terms = terms
            self.active_terms = list(terms)

        def get_term(self, name):
            import numpy as np
            return np.array([self._terms[name]])

    def _env_with(self, terms):
        import harness.envs.robolab as r
        env = r.RoboLabEnv.__new__(r.RoboLabEnv)
        inner = _FakeInner()
        inner.unwrapped = inner
        inner.termination_manager = self._Manager(terms)
        env._env = inner
        return env

    def test_the_success_term_is_read(self):
        env = self._env_with({"time_out": False, "success": True})
        self.assertTrue(env._extract_success({}))

    def test_a_timeout_is_not_success(self):
        env = self._env_with({"time_out": True, "success": False})
        self.assertFalse(env._extract_success({}))

    def test_a_goal_named_term_also_counts(self):
        env = self._env_with({"time_out": False, "goal_reached": True})
        self.assertTrue(env._extract_success({}))

    def test_the_info_dict_remains_a_fallback(self):
        import harness.envs.robolab as r
        env = r.RoboLabEnv.__new__(r.RoboLabEnv)
        env._env = _FakeInner()
        self.assertTrue(env._extract_success({"success": True}))
        self.assertFalse(env._extract_success({"success": False}))

    def test_is_success_uses_the_same_path(self):
        env = self._env_with({"time_out": False, "success": True})
        self.assertTrue(env._check_success())
