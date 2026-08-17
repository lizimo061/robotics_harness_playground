import unittest

from harness.agent import LLMController
from harness.config import LLMConfig
from harness.envs.tabletop import TabletopEnv
from harness.llm import get_llm
from harness.tasks.base import TaskSpec
from harness.tools import ToolRegistry, get_default_tools, parse_tool_call


class TestToolRegistry(unittest.TestCase):
    def test_default_tools(self):
        tools = get_default_tools()
        names = {t.name for t in tools}
        self.assertIn("grasp", names)
        self.assertIn("move_to", names)
        self.assertIn("list_objects", names)
        self.assertIn("done", names)

    def test_registry_lookup(self):
        reg = ToolRegistry(get_default_tools())
        self.assertIsNotNone(reg.get("release"))
        with self.assertRaises(KeyError):
            reg.get("nope")

    def test_openai_schema(self):
        t = ToolRegistry(get_default_tools()).get("move_to")
        fn = t.to_openai_function()["function"]
        self.assertEqual(fn["name"], "move_to")
        self.assertIn("x", fn["parameters"]["properties"])


class TestToolFiltering(unittest.TestCase):
    def _fake_env(self, kind):
        from harness.types import ActionSpace

        class FakeEnv:
            action_space = ActionSpace(kind=kind)

        return FakeEnv()

    def _controller(self):
        return LLMController(get_llm(LLMConfig(provider="mock")), mode="tools")

    def test_ee_mode_drops_set_joints(self):
        ctrl = self._controller()
        names = {t.name for t in ctrl._filter_tools_for_env(self._fake_env("ee_delta"))}
        self.assertIn("move_to", names)
        self.assertNotIn("set_joints", names)

    def test_joint_mode_drops_move(self):
        ctrl = self._controller()
        names = {t.name for t in ctrl._filter_tools_for_env(self._fake_env("joint_position"))}
        self.assertIn("set_joints", names)
        self.assertNotIn("move_to", names)
        self.assertNotIn("move_delta", names)


class TestParseToolCall(unittest.TestCase):
    def test_tool_args(self):
        name, args = parse_tool_call('{"tool": "move_to", "args": {"x": 0.4, "y": 0.6}}')
        self.assertEqual(name, "move_to")
        self.assertEqual(args, {"x": 0.4, "y": 0.6})

    def test_name_arguments_form(self):
        name, args = parse_tool_call('{"name": "grasp", "arguments": {}}')
        self.assertEqual(name, "grasp")
        self.assertEqual(args, {})

    def test_fenced(self):
        bt = chr(96) * 3
        text = bt + "json" + chr(10) + '{"tool": "done", "args": {}}' + chr(10) + bt
        name, _ = parse_tool_call(text)
        self.assertEqual(name, "done")

    def test_garbage(self):
        self.assertIsNone(parse_tool_call("no json here")[0])


class TestToolsMode(unittest.TestCase):
    def _stack_env(self):
        spec = TaskSpec(
            kind="stack",
            description="stack",
            objects=[
                {"name": "base", "pos": [0.4, 0.4], "role": "base"},
                {"name": "top", "pos": [0.4, 0.7], "role": "top"},
            ],
            goals={},
            ee_start=[0.1, 0.1],
            params={"stack_radius": 0.1, "grasp_radius": 0.15},
        )
        return TabletopEnv(task_spec=spec)

    def test_tools_mode_solves_stack(self):
        script = [
            {"tool": "move_to", "args": {"x": 0.4, "y": 0.7}},
            {"tool": "grasp", "args": {}},
            {"tool": "move_to", "args": {"x": 0.4, "y": 0.4}},
            {"tool": "release", "args": {}},
            {"tool": "done", "args": {}},
        ]
        llm = get_llm(LLMConfig(provider="mock", extra={"script": script}))
        ctrl = LLMController(llm, mode="tools", max_steps=20)
        ep = ctrl.run(self._stack_env())
        self.assertTrue(ep.success)
        self.assertEqual(ep.metadata["mode"], "tools")


if __name__ == "__main__":
    unittest.main()
