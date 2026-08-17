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
            r.RoboLabEnv(task="PickCubeTask")


if __name__ == "__main__":
    unittest.main()
