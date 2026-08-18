"""P2: the policy as an interruptible capability.

The behaviour under test is VoLoAgent's monitor loop: a rollout that pauses at a
cadence so the agent can choose continue / advance / recover, instead of a blocking
call that only reports once the budget is gone.

Two properties are load-bearing and easy to get wrong:

- a paused rollout reports ``success=None``, not ``False``. False would tell the
  agent the sub-instruction failed when it has not finished being attempted.
- ``abort`` holds position. Preemption that leaves the arm running is not
  preemption; VoLoAgent idles the robot while reasoning continues.
"""
import unittest

import numpy as np

from harness.envs.tabletop import TabletopEnv
from harness.policies.scripted import ScriptedPolicy
from harness.tools.policy_tool import AbortPolicyTool, ContinuePolicyTool, RunPolicyTool


class _CountingPolicy:
    """Records how many act() calls it saw, so chunking is observable."""

    def __init__(self, dim=2):
        self.dim = dim
        self.calls = 0

    def begin(self, instruction, action_space=None):
        self.instruction = instruction

    def act(self, observation_text, image=None):
        self.calls += 1
        return np.zeros(self.dim, dtype=np.float32)

    def reset(self):
        self.calls = 0

    def close(self):
        pass


def _env():
    env = TabletopEnv(task="pick_place")
    env.reset(seed=0)
    return env


class TestMonitorCadence(unittest.TestCase):
    def test_a_rollout_pauses_at_the_cadence(self):
        pol = _CountingPolicy()
        tool = RunPolicyTool(pol, default_steps=20, monitor_every=5)
        env = _env()
        r = tool.run(env, instruction="go", steps=20)
        env.close()
        self.assertEqual(pol.calls, 5, "did not stop at the monitor breakpoint")
        self.assertEqual(r.steps, 5)
        self.assertTrue(tool.is_running())

    def test_a_paused_rollout_reports_success_none_not_false(self):
        tool = RunPolicyTool(_CountingPolicy(), default_steps=20, monitor_every=5)
        env = _env()
        r = tool.run(env, instruction="go", steps=20)
        env.close()
        self.assertIsNone(r.success,
                          "a paused rollout must not claim the sub-instruction failed")

    def test_the_status_tells_the_agent_its_three_choices(self):
        tool = RunPolicyTool(_CountingPolicy(), default_steps=20, monitor_every=5)
        env = _env()
        r = tool.run(env, instruction="pick the cube", steps=20)
        env.close()
        self.assertIn("STILL RUNNING", r.feedback)
        self.assertIn("continue_policy", r.feedback)
        self.assertIn("abort_policy", r.feedback)

    def test_continue_resumes_the_same_rollout(self):
        pol = _CountingPolicy()
        tool = RunPolicyTool(pol, default_steps=20, monitor_every=5)
        cont = ContinuePolicyTool(tool)
        env = _env()
        tool.run(env, instruction="go", steps=20)
        cont.run(env)
        env.close()
        self.assertEqual(pol.calls, 10)
        self.assertEqual(pol.instruction, "go", "resumed with a different instruction")

    def test_the_budget_is_spent_across_chunks_not_restarted(self):
        pol = _CountingPolicy()
        tool = RunPolicyTool(pol, default_steps=12, monitor_every=5)
        cont = ContinuePolicyTool(tool)
        env = _env()
        tool.run(env, instruction="go", steps=12)   # 5
        cont.run(env)                                # 10
        last = cont.run(env)                         # 12 -> closes
        env.close()
        self.assertEqual(pol.calls, 12, "budget was restarted per chunk")
        self.assertFalse(tool.is_running())
        self.assertIsNotNone(last.success, "final chunk must report a verdict")

    def test_monitoring_off_runs_to_completion(self):
        pol = _CountingPolicy()
        tool = RunPolicyTool(pol, default_steps=9, monitor_every=0)
        env = _env()
        r = tool.run(env, instruction="go", steps=9)
        env.close()
        self.assertEqual(pol.calls, 9)
        self.assertFalse(tool.is_running())
        self.assertIsNotNone(r.success)


class TestPreemption(unittest.TestCase):
    def test_abort_ends_the_rollout(self):
        tool = RunPolicyTool(_CountingPolicy(), default_steps=20, monitor_every=5)
        env = _env()
        tool.run(env, instruction="go", steps=20)
        r = tool.abort(env)
        env.close()
        self.assertFalse(tool.is_running())
        self.assertIn("Aborted", r.feedback)

    def test_abort_holds_position_rather_than_leaving_it_running(self):
        pol = _CountingPolicy()
        tool = RunPolicyTool(pol, default_steps=50, monitor_every=5)
        env = _env()
        tool.run(env, instruction="go", steps=50)
        before = pol.calls
        r = tool.abort(env)
        self.assertEqual(pol.calls, before, "the policy kept acting after abort")
        self.assertIn("holds its current pose", r.feedback)
        env.close()

    def test_abort_reports_the_reason_back(self):
        tool = RunPolicyTool(_CountingPolicy(), default_steps=20, monitor_every=5)
        stop = AbortPolicyTool(tool)
        env = _env()
        tool.run(env, instruction="go", steps=20)
        r = stop.run(env, reason="gripper closed on nothing")
        env.close()
        self.assertIn("gripper closed on nothing", r.feedback)

    def test_continue_without_a_rollout_is_a_message_not_a_crash(self):
        tool = RunPolicyTool(_CountingPolicy(), monitor_every=5)
        env = _env()
        r = ContinuePolicyTool(tool).run(env)
        env.close()
        self.assertIn("No policy is running", r.feedback)

    def test_abort_without_a_rollout_is_a_message_not_a_crash(self):
        tool = RunPolicyTool(_CountingPolicy(), monitor_every=5)
        env = _env()
        r = AbortPolicyTool(tool).run(env)
        env.close()
        self.assertIn("No policy rollout was running", r.feedback)


class TestAccountingStaysHonest(unittest.TestCase):
    """Chunking must not corrupt the episode's step and reward bookkeeping."""

    def test_inner_steps_are_reported_per_chunk(self):
        seen = []
        tool = RunPolicyTool(_CountingPolicy(), default_steps=12, monitor_every=4)
        cont = ContinuePolicyTool(tool)
        env = _env()
        tool.run(env, instruction="go", steps=12, on_step=lambda a, r: seen.append(a))
        self.assertEqual(len(seen), 4)
        cont.run(env, on_step=lambda a, r: seen.append(a))
        env.close()
        self.assertEqual(len(seen), 8, "on_step did not fire across the resume")

    def test_a_policy_error_closes_the_rollout(self):
        class _Broken(_CountingPolicy):
            def act(self, observation_text, image=None):
                raise RuntimeError("policy died")

        tool = RunPolicyTool(_Broken(), default_steps=20, monitor_every=5)
        env = _env()
        r = tool.run(env, instruction="go", steps=20)
        env.close()
        self.assertFalse(tool.is_running(), "a dead policy left the rollout open")
        self.assertFalse(r.success)
        self.assertIn("policy error", r.feedback)

    def test_success_mid_chunk_closes_and_reports_it(self):
        env = _env()
        # the scripted policy drives toward the goal; give it room to finish
        tool = RunPolicyTool(ScriptedPolicy(action_dim=2), default_steps=200,
                             monitor_every=50)
        r = tool.run(env, instruction="reach", steps=200)
        env.close()
        self.assertIn(r.success, (True, None))  # either finished or paused cleanly


class TestToolsAreOfferedOnlyWhenUsable(unittest.TestCase):
    def test_no_cadence_means_no_continue_or_abort(self):
        from harness.tools.registry import get_policy_tools
        names = [t.name for t in get_policy_tools(_CountingPolicy())]
        self.assertNotIn("continue_policy", names)
        self.assertNotIn("abort_policy", names)

    def test_a_cadence_offers_them(self):
        from harness.tools.registry import get_policy_tools
        names = [t.name for t in get_policy_tools(_CountingPolicy(), monitor_every=5)]
        self.assertIn("continue_policy", names)
        self.assertIn("abort_policy", names)

    def test_they_share_one_rollout_state(self):
        """Separate copies would let the agent continue a rollout that does not exist."""
        from harness.tools.registry import get_policy_tools
        tools = {t.name: t for t in get_policy_tools(_CountingPolicy(), monitor_every=5)}
        env = _env()
        tools["run_policy"].run(env, instruction="go", steps=20)
        self.assertTrue(tools["continue_policy"]._runner.is_running())
        tools["abort_policy"].run(env)
        self.assertFalse(tools["run_policy"].is_running())
        env.close()


if __name__ == "__main__":
    unittest.main()
