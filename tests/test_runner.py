import unittest

from harness.config import EnvConfig, EvalConfig, HarnessConfig, LLMConfig
from harness.runner import run_eval

SCRIPT = [
    {"action": "move", "delta": [0.25, 0.25]},
    {"action": "move", "delta": [0.15, 0.15]},
    {"action": "close"},
    {"action": "move", "delta": [0.25, 0.25]},
    {"action": "move", "delta": [0.10, 0.10]},
    {"action": "stop"},
]


class TestRunner(unittest.TestCase):
    def test_run_eval_toy_solves(self):
        cfg = HarnessConfig(
            llm=LLMConfig(provider="mock", extra={"script": SCRIPT}),
            env=EnvConfig(name="toy_tabletop", task="pick_and_place"),
            eval=EvalConfig(episodes=1, save_trajectories=False, verbose=False),
        )
        summary = run_eval(cfg)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["episodes"], 1)


if __name__ == "__main__":
    unittest.main()
