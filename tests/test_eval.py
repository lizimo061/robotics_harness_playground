import tempfile
import unittest
from pathlib import Path

from harness.eval import TrajectoryLogger, summarize
from harness.types import Episode


class TestEval(unittest.TestCase):
    def test_summarize(self):
        e1 = Episode()
        e1.success = True
        e1.total_reward = 1.0
        e2 = Episode()
        e2.success = False
        s = summarize([e1, e2])
        self.assertEqual(s["episodes"], 2)
        self.assertEqual(s["success_rate"], 0.5)

    def test_summarize_empty(self):
        self.assertEqual(summarize([]), {})

    def test_logger(self):
        with tempfile.TemporaryDirectory() as d:
            l = TrajectoryLogger(log_dir=d, run_name="r")
            ep = Episode()
            ep.success = True
            l.log(ep)
            self.assertTrue(Path(d, "r.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
