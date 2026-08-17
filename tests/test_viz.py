import tempfile
import unittest
from pathlib import Path

import numpy as np

from harness.viz.html import render_html, save_html
from harness.viz.recorder import TraceRecorder


class TestTraceRecorder(unittest.TestCase):
    def test_record_and_finish(self):
        r = TraceRecorder(capture_frames=True, metadata={"env": "toy"})
        r.record(
            step=0,
            observation_text="obs",
            llm_response="resp",
            action={"kind": "move"},
            reward=-0.5,
            success=False,
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
        )
        r.finish(success=True, total_reward=1.0)
        self.assertEqual(len(r.steps), 1)
        self.assertTrue(r.final_success)
        self.assertEqual(r.to_dict()["metadata"]["env"], "toy")

    def test_capture_frames_off(self):
        r = TraceRecorder(capture_frames=False)
        ts = r.record(frame=np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertIsNone(ts.frame)


class TestRenderHtml(unittest.TestCase):
    def _trace(self):
        r = TraceRecorder(capture_frames=True)
        r.record(
            step=0,
            observation_text="a",
            llm_response="b",
            action={"kind": "stop"},
            reward=1.0,
            success=True,
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
        )
        return r

    def test_renders_frames_and_trace(self):
        h = render_html(self._trace(), title="t", fps=8)
        self.assertIn("data:image/png", h)
        self.assertIn("LLM response", h)
        self.assertIn("stop", h)

    def test_save_writes_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = save_html(self._trace(), Path(d) / "v.html")
            self.assertTrue(p.exists())
            self.assertGreater(p.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
