import json
import tempfile
import unittest
from pathlib import Path

from harness.config import HarnessConfig, LLMConfig, VizConfig, load_config


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = HarnessConfig()
        self.assertEqual(c.llm.provider, "deepseek")
        self.assertEqual(c.env.name, "toy_tabletop")
        self.assertEqual(c.agent.mode, "json")
        self.assertEqual(c.viz.backend, "html")
        self.assertTrue(c.viz.enabled)

    def test_from_dict_nested(self):
        c = HarnessConfig.from_dict({
            "llm": {"provider": "mock"},
            "env": {"name": "tabletop", "task": "stack"},
            "viz": {"backend": "console"},
        })
        self.assertIsInstance(c.llm, LLMConfig)
        self.assertEqual(c.llm.provider, "mock")
        self.assertEqual(c.env.task, "stack")
        self.assertEqual(c.viz.backend, "console")

    def test_roundtrip(self):
        c = HarnessConfig.from_dict({"llm": {"provider": "mock"}})
        self.assertEqual(c.to_dict()["llm"]["provider"], "mock")

    def test_load_yaml_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            yp = Path(d) / "c.yaml"
            yp.write_text("llm:\n  provider: mock\n")
            self.assertEqual(load_config(yp).llm.provider, "mock")
            jp = Path(d) / "c.json"
            jp.write_text(json.dumps({"llm": {"provider": "mock"}}))
            self.assertEqual(load_config(jp).llm.provider, "mock")

    def test_viz_defaults(self):
        v = VizConfig()
        self.assertEqual(v.backend, "html")
        self.assertEqual(v.fps, 8)


if __name__ == "__main__":
    unittest.main()
