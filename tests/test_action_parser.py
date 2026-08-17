import unittest

import numpy as np

from harness.agent.action_parser import extract_json, parse_action


class TestExtractJson(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(
            extract_json('{"action": "move", "delta": [0.1, 0.2]}'),
            {"action": "move", "delta": [0.1, 0.2]},
        )

    def test_fenced(self):
        bt = chr(96) * 3
        nl = chr(10)
        text = "Sure!" + nl + bt + "json" + nl + '{"action": "move"}' + nl + bt
        self.assertEqual(extract_json(text), {"action": "move"})

    def test_prose_prefix(self):
        text = 'Here is my action: {"action": "stop"} and done'
        self.assertEqual(extract_json(text), {"action": "stop"})

    def test_none_for_garbage(self):
        self.assertIsNone(extract_json("no json here"))


class TestParseAction(unittest.TestCase):
    def test_move(self):
        a = parse_action('{"action": "move", "delta": [0.1, 0.2], "gripper": 1}')
        self.assertEqual(a.kind, "ee_delta")
        np.testing.assert_allclose(a.value, [0.1, 0.2], rtol=1e-5)
        self.assertEqual(a.gripper, 1.0)

    def test_stop(self):
        self.assertEqual(parse_action('{"action": "stop"}').kind, "stop")

    def test_close(self):
        self.assertEqual(parse_action('{"action": "close"}').gripper, 1.0)

    def test_move_to(self):
        a = parse_action('{"action": "move_to", "pose": [0.1, 0.2, 0.3]}')
        self.assertEqual(a.kind, "ee_pose")

    def test_canonical_kind(self):
        a = parse_action('{"kind": "joint_position", "value": [1, 2, 3]}')
        self.assertEqual(a.kind, "joint_position")
        self.assertEqual(len(a.value), 3)

    def test_garbage_is_noop(self):
        a = parse_action("blah blah no json")
        self.assertEqual(a.kind, "noop")


if __name__ == "__main__":
    unittest.main()
