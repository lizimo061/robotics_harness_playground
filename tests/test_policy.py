import unittest

import numpy as np

from harness.agent.policy import PolicyAgent
from harness.config import LLMConfig
from harness.llm import get_llm
from harness.types import ActionSpace


class TestPolicyAgent(unittest.TestCase):
    def _agent(self, script, action_space=None):
        llm = get_llm(LLMConfig(provider="mock", extra={"script": script}))
        return PolicyAgent(llm, action_space=action_space, action_dim=8)

    def test_begin_and_act(self):
        agent = self._agent([{"action": "joints", "joint_positions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}])
        agent.begin("Pick up the cube")
        vec = agent.act("observation here")
        self.assertEqual(vec.shape, (8,))
        np.testing.assert_allclose(vec[:7], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    def test_gripper_last(self):
        agent = self._agent([{"action": "gripper", "value": 1.0}])
        agent.begin("close gripper")
        vec = agent.act("obs")
        self.assertEqual(vec[-1], 1.0)

    def test_stop_maps_to_zero(self):
        agent = self._agent([{"action": "stop"}])
        agent.begin("done")
        vec = agent.act("obs")
        self.assertTrue(np.allclose(vec, 0.0))

    def test_value_truncated_to_dim(self):
        agent = self._agent([{"action": "joints", "joint_positions": list(range(20))}])
        agent.begin("t")
        vec = agent.act("o")
        self.assertEqual(vec.shape, (8,))
        np.testing.assert_allclose(vec, list(range(8)))

    def test_action_space_from_begin(self):
        space = ActionSpace(kind="ee_delta", dim=4, description="4-dim ee delta")
        agent = self._agent([{"action": "move", "delta": [1, 2, 3, 4]}], action_space=space)
        agent.begin("move", action_space=space)
        self.assertEqual(agent._action_dim, 4)
        vec = agent.act("o")
        np.testing.assert_allclose(vec, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
