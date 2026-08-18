"""Frame capture at the environment boundary.

The agent only renders once per LLM turn, but a single tool call can drive many
environment steps (``move_to`` interpolates, ``run_policy`` runs a whole closed
loop). A video built from per-turn frames is therefore jumpy and, worse, hides
exactly the motion a reviewer wants to see.

Capturing in a wrapper around ``step()`` sidesteps that: it is independent of
which agent mode ran and of whether the tool exposes an ``on_step`` hook, so
every environment step lands in the video regardless of what produced it.

    env = FrameCapture(env)
    ...run an episode...
    write_video(env.frames, "episode.mp4")
    env.clear()
"""
from __future__ import annotations

from typing import Any, Optional

from harness.utils.logging import get_logger

log = get_logger("harness.viz.capture")


class FrameCapture:
    """Wrap an Env and render a frame after every ``step()``.

    Delegates everything else, so the query tools an agent uses
    (``list_objects``, ``get_object_position``, ...) keep working untouched.
    """

    def __init__(self, env: Any, *, every: int = 1, max_frames: int = 4000,
                 skip_blank: bool = False) -> None:
        self.env = env
        self._every = max(1, int(every))
        #: Isaac's first camera frame after a reset is often solid black -- the
        #: renderer has not settled. That is a warm-up artifact rather than
        #: data, so leading blank frames may be dropped. Off by default, and it
        #: only ever drops *leading* frames, so a genuinely dark scene mid
        #: episode is preserved.
        self._skip_blank = skip_blank
        #: a hard cap, because an agent that loops forever would otherwise
        #: accumulate frames until the process dies
        self._max_frames = max_frames
        self.frames: list = []
        self._steps = 0
        self._warned = False

    # -- capture ---------------------------------------------------------- #
    def _capture(self) -> None:
        if len(self.frames) >= self._max_frames:
            if not self._warned:
                log.warning("frame cap (%d) reached; later steps are not recorded",
                            self._max_frames)
                self._warned = True
            return
        try:
            frame = self.env.render()
        except Exception as e:  # noqa: BLE001 - a render fault must not kill the run
            if not self._warned:
                log.warning("render() raised %s: %s; no frames captured",
                            type(e).__name__, e)
                self._warned = True
            return
        if frame is None:
            return
        if self._skip_blank and not self.frames and self._is_blank(frame):
            log.debug("dropping a blank leading frame (renderer warm-up)")
            return
        self.frames.append(frame)

    @staticmethod
    def _is_blank(frame: Any) -> bool:
        try:
            return float(frame.max()) <= 8.0  # near-black in 0..255
        except Exception:  # noqa: BLE001 - not an array we can judge
            return False

    def clear(self) -> None:
        """Drop captured frames (call between episodes)."""
        self.frames = []
        self._steps = 0
        self._warned = False

    # -- Env surface ------------------------------------------------------ #
    def reset(self, **kw: Any):
        obs = self.env.reset(**kw)
        self._steps = 0
        self._capture()  # the initial state is part of the story
        return obs

    def step(self, action: Any):
        result = self.env.step(action)
        self._steps += 1
        if self._steps % self._every == 0:
            self._capture()
        return result

    def render(self) -> Optional[Any]:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    # Anything else -- name, action_space, get_text_state, is_success, the query
    # API the tools call -- belongs to the wrapped env.
    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
