import tempfile
import unittest
from pathlib import Path

import numpy as np


def _have_writer() -> bool:
    try:
        import imageio  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_have_writer(), "no video writer (imageio/cv2) available")
class TestVideo(unittest.TestCase):
    def test_write_mp4(self):
        from harness.viz.video import write_video

        frames = [np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(4)]
        with tempfile.TemporaryDirectory() as d:
            p = write_video(frames, Path(d) / "v.mp4", fps=4)
            self.assertTrue(p.exists())
            self.assertGreater(p.stat().st_size, 0)

    def test_write_gif(self):
        from harness.viz.video import write_video

        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
        with tempfile.TemporaryDirectory() as d:
            p = write_video(frames, Path(d) / "v.gif", fps=2)
            self.assertTrue(p.exists())

    def test_empty_frames_raises(self):
        from harness.viz.video import write_video

        with self.assertRaises(ValueError):
            write_video([], "x.mp4")


class TestFramesFromRecorder(unittest.TestCase):
    def test_skips_none(self):
        from harness.viz.recorder import TraceRecorder
        from harness.viz.video import frames_from_recorder

        r = TraceRecorder(capture_frames=True)
        r.record(frame=np.zeros((4, 4, 3), dtype=np.uint8))
        r.record(frame=None)
        self.assertEqual(len(frames_from_recorder(r)), 1)


if __name__ == "__main__":
    unittest.main()
