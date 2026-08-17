"""Write rendered RGB frames to a video file (mp4 / gif)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _normalize(frames):
    out = []
    for f in frames:
        if f is None:
            continue
        a = np.asarray(f)
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        out.append(a)
    return out


def write_video(frames, path, fps: float = 30.0) -> Path:
    """Write a list of RGB uint8 frames (H, W, 3) to a video file.

    The format is inferred from the extension (e.g. .mp4, .gif). Uses imageio
    (with imageio-ffmpeg) when available, falling back to OpenCV.
    """
    out = _normalize(frames)
    if not out:
        raise ValueError("no frames to write")
    path = str(path)
    try:
        import imageio

        writer = imageio.get_writer(path, fps=fps)
        for a in out:
            writer.append_data(a)
        writer.close()
    except ImportError:
        import cv2

        h, w = out[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for a in out:
            writer.write(cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
        writer.release()
    return Path(path)


def frames_from_recorder(recorder) -> list:
    """Extract captured frames from a TraceRecorder (one per recorded step)."""
    return [s.frame for s in recorder.steps if s.frame is not None]
