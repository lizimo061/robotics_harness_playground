"""Live visualization backends: console trace and optional matplotlib viewer."""
from __future__ import annotations

from typing import Optional

from harness.utils.logging import get_logger
from harness.viz.recorder import TraceStep

log = get_logger("harness.viz.live")

_NL = chr(10)


class ConsoleTracer:
    """Prints a compact, live step-by-step trace to stdout as the episode runs."""

    def on_step(self, ts: TraceStep) -> None:
        reward = f"{ts.reward:.3f}" if ts.reward is not None else "-"
        mark = "OK " if ts.success else "   "
        print(f"[{mark}] step {ts.step + 1}  reward={reward}")
        print(f"    state: {ts.observation_text.replace(_NL, ' | ')[:160]}")
        print(f"    llm  : {ts.llm_response.replace(_NL, ' ')[:200]}")
        print(f"    act  : {ts.action}")


class MatplotlibViewer:
    """Opens a matplotlib window and shows the latest frame live.

    Requires matplotlib and a display. If it cannot open a window, the runner
    falls back to the html backend.
    """

    def __init__(self, *, fps: float = 8, size: float = 4.0, title: str = "robot view") -> None:
        import matplotlib.pyplot as plt  # type: ignore

        self._plt = plt
        self._fps = fps
        self._fig, self._ax = plt.subplots(figsize=(size, size))
        self._ax.set_title(title)
        self._ax.axis("off")
        self._im: Optional[object] = None

    def on_step(self, ts: TraceStep) -> None:
        if ts.frame is None:
            return
        if self._im is None:
            self._im = self._ax.imshow(ts.frame)
            self._plt.ion()
            self._plt.show(block=False)
        else:
            self._im.set_data(ts.frame)
        self._plt.pause(1.0 / self._fps)

    def close(self) -> None:
        try:
            self._plt.close(self._fig)
        except Exception:
            pass
