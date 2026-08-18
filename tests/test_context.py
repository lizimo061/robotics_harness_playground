"""Tests for the monitor/deliberation split.

The property that matters is *boundedness*: the monitor prompt must not grow with
episode length, or it cannot be read at the motion timescale. The second property is
that the two views cannot diverge, because they read one transcript.
"""
import unittest

import numpy as np

from harness.agent.context import (
    DeliberationContext,
    MonitorContext,
    Transcript,
    estimate_prompt_chars,
)


def _transcript(n_turns=0, system="SYS"):
    t = Transcript(system=system)
    for i in range(n_turns):
        t.begin_turn(f"observation {i} " + "x" * 200)
        t.record_reply(f'{{"tool": "move_to", "args": {{"x": {i}}}}}', decision=f"move_to({i})")
        t.record_feedback(f"moved to {i}")
    return t


class TestMonitorIsBounded(unittest.TestCase):
    def test_the_monitor_prompt_is_constant_in_episode_length(self):
        """O(1) in turns, once the decision window is full.

        Comparing 2 turns against 80 would measure the window *filling* (2 -> 4
        decisions), which is the bounded behaviour rather than a violation. The
        invariant is that past the cap, more history costs nothing.
        """
        mon = MonitorContext(decisions=4)
        at80 = estimate_prompt_chars(mon.messages(_transcript(80), "now"))
        at400 = estimate_prompt_chars(mon.messages(_transcript(400), "now"))
        self.assertLess(abs(at400 - at80), 20,
                        f"monitor grew with history: {at80} -> {at400} chars")

    def test_the_monitor_is_a_small_fraction_of_deliberation(self):
        t = _transcript(60)
        mon = estimate_prompt_chars(MonitorContext().messages(t, "now"))
        dlb = estimate_prompt_chars(DeliberationContext().messages(t, "now"))
        self.assertLess(mon * 20, dlb, f"monitor {mon} vs deliberation {dlb}")

    def test_deliberation_does_grow(self):
        """The slow clock is supposed to be expensive; that is the trade."""
        dlb = DeliberationContext()
        short = estimate_prompt_chars(dlb.messages(_transcript(2), "now"))
        long = estimate_prompt_chars(dlb.messages(_transcript(80), "now"))
        self.assertGreater(long, short * 5)

    def test_only_the_last_k_decisions_are_carried(self):
        t = _transcript(20)
        msgs = MonitorContext(decisions=3).messages(t, "now")
        body = msgs[-1].content
        self.assertIn("move_to(19)", body)
        self.assertIn("move_to(17)", body)
        self.assertNotIn("move_to(10)", body)

    def test_the_newest_observation_is_present_and_last(self):
        msgs = MonitorContext().messages(_transcript(5), "THE CURRENT STATE")
        self.assertTrue(msgs[-1].content.rstrip().endswith("THE CURRENT STATE"))

    def test_the_system_line_survives(self):
        msgs = MonitorContext().messages(_transcript(3, system="BE A ROBOT"), "now")
        self.assertEqual(msgs[0].role, "system")
        self.assertIn("BE A ROBOT", msgs[0].content)

    def test_zero_decisions_is_honoured(self):
        msgs = MonitorContext(decisions=0).messages(_transcript(5), "now")
        self.assertNotIn("Recent decisions", msgs[-1].content)


class TestSubgoals(unittest.TestCase):
    def test_the_active_subgoal_is_shown_with_progress(self):
        t = _transcript(1)
        t.set_subgoals(["pick the cube", "put it in the bowl"])
        body = MonitorContext().messages(t, "now")[-1].content
        self.assertIn("(1/2)", body)
        self.assertIn("pick the cube", body)

    def test_advancing_moves_to_the_next_subgoal(self):
        t = Transcript()
        t.set_subgoals(["a", "b"])
        self.assertEqual(t.active_subgoal, "a")
        self.assertEqual(t.advance_subgoal(), "b")
        self.assertIsNone(t.advance_subgoal())

    def test_advancing_past_the_end_is_not_an_error(self):
        t = Transcript()
        t.set_subgoals(["only"])
        t.advance_subgoal(); t.advance_subgoal()
        self.assertIsNone(t.active_subgoal)

    def test_deliberation_shows_the_plan_with_progress_markers(self):
        t = _transcript(2)
        t.set_subgoals(["one", "two", "three"])
        t.advance_subgoal()
        body = " ".join(m.content for m in DeliberationContext().messages(t, "now")
                        if isinstance(m.content, str))
        self.assertIn("x one", body)   # done
        self.assertIn("> two", body)   # active
        self.assertIn("  three", body)  # pending


class TestOneSourceOfTruth(unittest.TestCase):
    """Both views read the same transcript, so a fast-clock decision is visible slow."""

    def test_a_monitor_decision_appears_in_deliberation(self):
        t = Transcript(system="S")
        t.begin_turn("obs")
        t.record_reply("reply", decision="grasp(cube)")
        self.assertIn("grasp(cube)", MonitorContext().messages(t, "now")[-1].content)
        replayed = [m.content for m in DeliberationContext().messages(t, "now")]
        self.assertIn("reply", replayed)

    def test_feedback_is_replayed_as_a_tool_result(self):
        t = Transcript()
        t.begin_turn("obs"); t.record_reply("r"); t.record_feedback("moved 5cm")
        bodies = [m.content for m in DeliberationContext().messages(t, "x")]
        self.assertTrue(any("Tool result: moved 5cm" == b for b in bodies))

    def test_recording_without_a_turn_is_a_no_op_not_a_crash(self):
        t = Transcript()
        t.record_reply("r"); t.record_feedback("f")   # nothing begun yet
        self.assertEqual(t.turns, [])


class TestVision(unittest.TestCase):
    def test_the_monitor_attaches_a_frame_when_given_one(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        msgs = MonitorContext().messages(_transcript(2), "now", image=img)
        self.assertIsInstance(msgs[-1].content, list)
        self.assertIn("image_url", [b.get("type") for b in msgs[-1].content])

    def test_no_frame_means_plain_text(self):
        msgs = MonitorContext().messages(_transcript(2), "now", image=None)
        self.assertIsInstance(msgs[-1].content, str)

    def test_include_image_false_ignores_the_frame(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        msgs = MonitorContext(include_image=False).messages(_transcript(1), "now", image=img)
        self.assertIsInstance(msgs[-1].content, str)


class TestDeliberationCap(unittest.TestCase):
    def test_a_cap_trims_the_oldest_turns(self):
        t = _transcript(30)
        msgs = DeliberationContext(max_turns=5).messages(t, "now")
        joined = " ".join(m.content for m in msgs if isinstance(m.content, str))
        self.assertIn("observation 29", joined)
        self.assertNotIn("observation 3 ", joined)

    def test_uncapped_by_default(self):
        t = _transcript(12)
        joined = " ".join(m.content for m in DeliberationContext().messages(t, "now")
                          if isinstance(m.content, str))
        self.assertIn("observation 0", joined)


if __name__ == "__main__":
    unittest.main()


class TestControllerIntegration(unittest.TestCase):
    """The split must be opt-in and must not change default behaviour."""

    class _LLM:
        name = "stub"

        def __init__(self):
            self.prompts = []

        def complete(self, messages, **kw):
            from harness.llm.base import LLMResponse
            self.prompts.append(list(messages))
            return LLMResponse(content='{"tool": "get_end_effector_position", "args": {}}',
                               model="stub", usage={}, raw={}, finish_reason="stop")

    def _run(self, two_clock, steps=6):
        from harness.agent.llm_controller import LLMController
        from harness.envs.tabletop import TabletopEnv
        llm = self._LLM()
        env = TabletopEnv(task="pick_place")
        LLMController(llm, mode="tools", max_steps=steps, two_clock=two_clock,
                      task_description="t").run(env, seed=0)
        env.close()
        return llm

    def test_off_by_default_the_transcript_grows(self):
        llm = self._run(False)
        first = estimate_prompt_chars(llm.prompts[0])
        last = estimate_prompt_chars(llm.prompts[-1])
        self.assertGreater(last, first * 1.5, "default mode should replay history")

    def test_on_the_monitor_prompt_saturates_instead_of_accumulating(self):
        """It rises while the decision window fills, then stays flat.

        The invariant is saturation, not constancy: a 14-turn episode must end at
        the same prompt size as a 6-turn one, because history past the window
        costs nothing.
        """
        short = [estimate_prompt_chars(p) for p in self._run(True, steps=6).prompts]
        long = [estimate_prompt_chars(p) for p in self._run(True, steps=14).prompts]
        self.assertEqual(short[-1], long[-1],
                         f"prompt kept growing: {short[-1]} vs {long[-1]}")
        # and the flat tail is genuinely flat
        self.assertEqual(len(set(long[-4:])), 1, f"tail not flat: {long[-6:]}")

    def test_the_split_beats_the_default_on_a_long_episode(self):
        default = [estimate_prompt_chars(p) for p in self._run(False, steps=14).prompts]
        split = [estimate_prompt_chars(p) for p in self._run(True, steps=14).prompts]
        self.assertLess(split[-1] * 2, default[-1],
                        f"two-clock {split[-1]} vs default {default[-1]}")

    def test_the_first_turn_is_a_planning_point(self):
        from harness.agent.llm_controller import LLMController
        from harness.envs.tabletop import TabletopEnv
        llm = self._LLM()
        env = TabletopEnv(task="pick_place")
        ctrl = LLMController(llm, mode="tools", max_steps=3, two_clock=True,
                            task_description="t")
        ctrl.run(env, seed=0)
        env.close()
        self.assertTrue(ctrl._transcript.turns[0].deliberated)
        self.assertFalse(ctrl._transcript.turns[1].deliberated)

    def test_decisions_are_recorded_for_the_monitor_to_carry(self):
        from harness.agent.llm_controller import LLMController
        from harness.envs.tabletop import TabletopEnv
        env = TabletopEnv(task="pick_place")
        ctrl = LLMController(self._LLM(), mode="tools", max_steps=4, two_clock=True,
                            task_description="t")
        ctrl.run(env, seed=0)
        env.close()
        self.assertTrue(all("get_end_effector_position" in t.decision
                            for t in ctrl._transcript.turns))

    def test_tool_feedback_reaches_the_transcript(self):
        from harness.agent.llm_controller import LLMController
        from harness.envs.tabletop import TabletopEnv
        env = TabletopEnv(task="pick_place")
        ctrl = LLMController(self._LLM(), mode="tools", max_steps=3, two_clock=True,
                            task_description="t")
        ctrl.run(env, seed=0)
        env.close()
        self.assertTrue(any(t.feedback for t in ctrl._transcript.turns))
