import unittest

from harness.tasks import available_tasks, generate_task
from harness.tasks.base import generate_curriculum


class TestTaskGenerators(unittest.TestCase):
    def test_all_kinds_registered(self):
        for kind in ("pick_place", "pick_place_obstacle", "push", "stack", "sort", "reach_avoid"):
            self.assertIn(kind, available_tasks())

    def test_deterministic(self):
        a = generate_task("stack", seed=7, difficulty=0.4)
        b = generate_task("stack", seed=7, difficulty=0.4)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_difficulty_increases_distance(self):
        easy = generate_task("pick_place", seed=1, difficulty=0.0)
        hard = generate_task("pick_place", seed=1, difficulty=1.0)
        import numpy as np
        de = np.linalg.norm(np.array(easy.objects[0]["pos"]) - np.array(easy.goals["goal"]))
        dh = np.linalg.norm(np.array(hard.objects[0]["pos"]) - np.array(hard.goals["goal"]))
        self.assertGreater(dh, de)

    def test_obstacle_task_has_obstacle(self):
        s = generate_task("pick_place_obstacle", seed=2, difficulty=0.5)
        self.assertEqual(len(s.obstacles), 1)

    def test_sort_has_multiple_objects(self):
        s = generate_task("sort", seed=3, difficulty=0.9)
        self.assertGreaterEqual(len(s.objects), 2)
        self.assertEqual(len(s.objects), len(s.goals))

    def test_curriculum(self):
        c = generate_curriculum("stack", [0, 1, 2], [0.0, 0.5, 1.0])
        self.assertEqual([x.difficulty for x in c], [0.0, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
