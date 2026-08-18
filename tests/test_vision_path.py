"""The vision path, which was a silent no-op in tools mode.

`use_vision: true` with `mode: tools` was accepted, forwarded into the policy
options, and then ignored: _tools_step appended text unconditionally, so the model
never received a pixel while every run looked like a vision run. Silent degradation
is the failure mode these tests exist to prevent.
"""
import unittest
from unittest import mock

import numpy as np

from harness.agent.llm_controller import LLMController
from harness.envs.tabletop import TabletopEnv
from harness.llm.base import LLMResponse


class _StubLLM:
    name = "stub"

    def __init__(self, reply='{"tool": "done", "args": {}}'):
        self.reply = reply
        self.seen: list = []

    def complete(self, messages, **kw):
        self.seen.append(list(messages))
        return LLMResponse(content=self.reply, model="stub", usage={}, raw={},
                           finish_reason="stop")


def _last_user(messages):
    return [m for m in messages if m.role == "user"][-1]


class TestVisionInToolsMode(unittest.TestCase):
    def _run(self, use_vision):
        llm = _StubLLM()
        env = TabletopEnv(task="pick_place")
        ctrl = LLMController(llm, mode="tools", max_steps=1, use_vision=use_vision,
                             task_description="t")
        ctrl.run(env, seed=0)
        env.close()
        return llm

    def test_an_image_is_attached_when_vision_is_on(self):
        llm = self._run(True)
        content = _last_user(llm.seen[0]).content
        self.assertIsInstance(content, list, "vision request sent text only")
        kinds = [b.get("type") for b in content]
        self.assertIn("image_url", kinds)
        self.assertIn("text", kinds)

    def test_text_only_when_vision_is_off(self):
        llm = self._run(False)
        self.assertIsInstance(_last_user(llm.seen[0]).content, str)

    def test_the_frame_is_a_data_uri(self):
        llm = self._run(True)
        blocks = _last_user(llm.seen[0]).content
        url = next(b["image_url"]["url"] for b in blocks if b.get("type") == "image_url")
        self.assertTrue(url.startswith("data:image/"), url[:40])

    def test_requesting_vision_turns_on_frame_capture(self):
        """Without this the env is never rendered, so there is nothing to attach."""
        ctrl = LLMController(_StubLLM(), mode="tools", use_vision=True)
        self.assertTrue(ctrl._capture_frames)

    def test_a_missing_frame_warns_instead_of_degrading_quietly(self):
        class _Blind(TabletopEnv):
            def render(self):
                return None

        llm = _StubLLM()
        env = _Blind(task="pick_place")
        ctrl = LLMController(llm, mode="tools", max_steps=1, use_vision=True,
                             task_description="t")
        with self.assertLogs("harness.agent.llm_controller", level="WARNING") as cm:
            # tabletop fills obs.image itself, so blank that too
            with mock.patch.object(_Blind, "_render_frame", create=True, return_value=None):
                ctrl._warned_no_image = False
                msg = ctrl._observation_message("state", None, None)
        self.assertIsInstance(msg.content, str)
        self.assertTrue(any("NOT a vision evaluation" in m for m in cm.output))
        env.close()


class TestPolicyFrameFreshness(unittest.TestCase):
    """A visuomotor policy must see the current frame, not the first one."""

    def test_the_frame_is_re_rendered_every_step(self):
        from harness.tools.policy_tool import RunPolicyTool

        seen: list = []

        class _Policy:
            def begin(self, instruction, **kw): pass
            def act(self, obs_text, image=None):
                seen.append(None if image is None else int(image[0, 0, 0]))
                return np.zeros(2, dtype=np.float32)
            def reset(self): pass
            def close(self): pass

        class _Env(TabletopEnv):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.n = 0

            def render(self):
                self.n += 1
                return np.full((4, 4, 3), self.n, dtype=np.uint8)

        env = _Env(task="pick_place")
        env.reset(seed=0)
        tool = RunPolicyTool(_Policy(), default_steps=4, use_vision=True)
        tool.run(env, instruction="go", steps=4)
        env.close()
        self.assertGreater(len(set(seen)), 1,
                           f"the same frame was reused for the whole rollout: {seen}")


if __name__ == "__main__":
    unittest.main()
