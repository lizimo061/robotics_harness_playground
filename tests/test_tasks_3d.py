import unittest

from harness.tasks import available_tasks, generate_task


class TestTasks3D(unittest.TestCase):
    def test_all_kinds_registered(self):
        for k in ("pick_place", "pick_place_obstacle", "push", "stack", "sort", "reach", "reach_avoid"):
            self.assertIn(k, available_tasks(3))

    def test_positions_are_3d(self):
        s = generate_task("pick_place", seed=2, difficulty=0.5, dims=3)
        self.assertEqual(s.dims, 3)
        self.assertEqual(len(s.objects[0]["pos"]), 3)
        self.assertEqual(len(s.goals["goal"]), 3)

    def test_stack_has_base_and_top(self):
        s = generate_task("stack", seed=1, difficulty=0.5, dims=3)
        self.assertEqual({o["name"] for o in s.objects}, {"base", "top"})

    def test_obstacle_is_3d(self):
        s = generate_task("pick_place_obstacle", seed=1, difficulty=0.5, dims=3)
        self.assertEqual(len(s.obstacles), 1)
        self.assertEqual(len(s.obstacles[0]["pos"]), 3)

    def test_deterministic(self):
        a = generate_task("sort", seed=5, difficulty=0.7, dims=3)
        b = generate_task("sort", seed=5, difficulty=0.7, dims=3)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_2d_and_3d_are_separate(self):
        self.assertNotIn("reach", available_tasks(2))
        self.assertIn("reach", available_tasks(3))


class TestGenesisModule(unittest.TestCase):
    def test_lazy_import_without_genesis(self):
        # importing the module must NOT require genesis-world (heavy import is
        # deferred to __init__)
        import harness.envs.genesis as g

        self.assertEqual(g.GenesisFrankaEnv.name, "genesis")
        self.assertTrue(g.GenesisEnv is g.GenesisFrankaEnv)


if __name__ == "__main__":
    unittest.main()
