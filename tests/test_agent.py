import unittest

from harness.agent import LLMController
from harness.config import LLMConfig
from harness.envs.toy import ToyTabletopEnv
from harness.llm import get_llm

SCRIPT = [
    {"action": "move", "delta": [0.25, 0.25]},
    {"action": "move", "delta": [0.15, 0.15]},
    {"action": "close"},
    {"action": "move", "delta": [0.25, 0.25]},
    {"action": "move", "delta": [0.10, 0.10]},
    {"action": "stop"},
]


class TestAgent(unittest.TestCase):
    def test_json_mode_solves(self):
        llm = get_llm(LLMConfig(provider="mock", extra={"script": SCRIPT}))
        env = ToyTabletopEnv(seed=0)
        ctrl = LLMController(llm, mode="json", max_steps=20)
        ep = ctrl.run(env)
        self.assertTrue(ep.success)
        self.assertGreaterEqual(ep.steps, 1)
        self.assertEqual(ep.metadata["mode"], "json")

    def test_code_mode_solves(self):
        code = """move_delta(0.25, 0.25)
move_delta(0.15, 0.15)
grasp()
move_delta(0.25, 0.25)
move_delta(0.10, 0.10)
done()
"""
        llm = get_llm(LLMConfig(provider="mock", extra={"responses": [code]}))
        env = ToyTabletopEnv(seed=0)
        ctrl = LLMController(llm, mode="code", max_steps=10)
        ep = ctrl.run(env)
        self.assertTrue(ep.success)

    def test_plan_mode_runs(self):
        llm = get_llm(
            LLMConfig(
                provider="mock",
                extra={"responses": ["1. move to object 2. grasp 3. carry to goal"], "script": SCRIPT},
            )
        )
        env = ToyTabletopEnv(seed=0)
        ctrl = LLMController(llm, mode="plan", max_steps=20)
        ep = ctrl.run(env)
        self.assertTrue(ep.success)


if __name__ == "__main__":
    unittest.main()
