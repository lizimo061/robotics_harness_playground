import unittest


class TestOptionalEnvModules(unittest.TestCase):
    def test_gymnasium_lazy_import(self):
        import harness.envs.gymnasium as g

        self.assertEqual(g.GymnasiumEnv.name, "gymnasium")

    def test_robosuite_lazy_import(self):
        import harness.envs.robosuite as r

        self.assertEqual(r.RobosuiteEnv.name, "robosuite")


if __name__ == "__main__":
    unittest.main()
