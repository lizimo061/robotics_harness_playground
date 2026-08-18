import json
import unittest

import numpy as np

from harness.agent import LLMController
from harness.config import LLMConfig
from harness.envs.tabletop import TabletopEnv
from harness.llm import get_llm
from harness.policies import ScriptedPolicy, get_policy
from harness.policies.remote import RemotePolicy
from harness.tasks import generate_task
from harness.tasks.base import TaskSpec
from harness.tools import RunPolicyTool, get_default_tools, get_policy_tools


def _env(task="pick_place", seed=1):
    return TabletopEnv(task_spec=generate_task(task, seed=seed))


def _reach_env():
    """Obstacle-free reach, so a straight-line policy provably succeeds."""
    spec = TaskSpec(
        kind="reach",
        description="Move the end-effector to the target.",
        ee_start=[0.1, 0.1],
        ee_target=[0.7, 0.7],
        params={"target_radius": 0.07},
    )
    return TabletopEnv(task_spec=spec)


def _homing_policy(env):
    """A stand-in 'trained policy': drive straight at the reach target."""
    target = np.asarray(env.task_spec.ee_target, dtype=float)

    def fn(obs_text, image, step):
        return np.clip(target - env.get_ee_pos(), -0.05, 0.05)

    return ScriptedPolicy(fn, action_dim=2)


class TestRunPolicyTool(unittest.TestCase):
    def test_runs_policy_for_requested_steps(self):
        env = _env()
        env.reset()
        policy = ScriptedPolicy([[0.05, 0.0]], action_dim=2)
        tool = RunPolicyTool(policy)

        steps = []
        res = tool.run(env, instruction="move right", steps=7, on_step=lambda a, r: steps.append((a, r)))

        self.assertEqual(res.steps, 7)
        self.assertEqual(len(steps), 7)
        self.assertEqual(policy.instruction, "move right")

    def test_forwards_instruction_and_action_space(self):
        env = _env()
        env.reset()
        policy = ScriptedPolicy([[0.0, 0.0]], action_dim=2)
        RunPolicyTool(policy).run(env, instruction="do the thing", steps=1)
        self.assertEqual(policy.instructions, ["do the thing"])

    def test_budget_is_clamped(self):
        env = _env()
        env.reset()
        tool = RunPolicyTool(ScriptedPolicy([[0.0, 0.0]], action_dim=2), max_steps=5)
        res = tool.run(env, instruction="x", steps=9999)
        self.assertEqual(res.steps, 5)

    def test_default_steps_on_missing_or_garbage_budget(self):
        env = _env()
        env.reset()
        tool = RunPolicyTool(ScriptedPolicy([[0.0, 0.0]], action_dim=2), default_steps=3)
        self.assertEqual(tool.run(env, instruction="x").steps, 3)
        env.reset()
        self.assertEqual(tool.run(env, instruction="x", steps="not-a-number").steps, 3)

    def test_empty_instruction_is_rejected_without_stepping(self):
        env = _env()
        env.reset()
        tool = RunPolicyTool(ScriptedPolicy([[0.1, 0.1]], action_dim=2))
        res = tool.run(env, instruction="   ")
        self.assertEqual(res.steps, 0)
        self.assertIn("instruction", res.feedback)

    def test_policy_error_is_reported_not_raised(self):
        class Boom(ScriptedPolicy):
            def act(self, observation_text, image=None):
                raise RuntimeError("policy server down")

        env = _env()
        env.reset()
        res = RunPolicyTool(Boom()).run(env, instruction="x", steps=3)
        self.assertFalse(res.success)
        self.assertIn("policy server down", res.feedback)

    def test_stops_early_on_success(self):
        env = _reach_env()
        env.reset()
        tool = RunPolicyTool(_homing_policy(env))
        res = tool.run(env, instruction="reach the target", steps=200)
        self.assertTrue(res.success)
        self.assertIn("TASK SUCCESS", res.feedback)
        self.assertLess(res.steps, 200)  # stopped early rather than burning the budget

    def test_feedback_includes_state(self):
        env = _env()
        env.reset()
        res = RunPolicyTool(ScriptedPolicy([[0.0, 0.0]], action_dim=2)).run(env, instruction="x", steps=1)
        self.assertIn("Resulting state:", res.feedback)


class TestToolsets(unittest.TestCase):
    def test_default_tools_have_no_policy_tool_without_a_policy(self):
        self.assertNotIn("run_policy", [t.name for t in get_default_tools()])

    def test_default_tools_gain_policy_tool(self):
        names = [t.name for t in get_default_tools(policy=ScriptedPolicy())]
        self.assertIn("run_policy", names)
        self.assertIn("move_to", names)

    def test_policy_toolset_drops_manual_motion_tools(self):
        names = [t.name for t in get_policy_tools(ScriptedPolicy())]
        self.assertIn("run_policy", names)
        self.assertIn("done", names)
        self.assertIn("list_objects", names)
        for manual in ("move_to", "move_delta", "set_joints", "grasp", "release"):
            self.assertNotIn(manual, names)

    def test_policy_tool_is_closed_loop(self):
        self.assertTrue(RunPolicyTool(ScriptedPolicy()).closed_loop)
        self.assertFalse(any(t.closed_loop for t in get_default_tools()))


class TestPolicyRegistry(unittest.TestCase):
    def test_passthrough_policy_object(self):
        p = ScriptedPolicy()
        self.assertIs(get_policy(p), p)

    def test_build_scripted(self):
        p = get_policy({"type": "scripted", "action_dim": 4})
        self.assertIsInstance(p, ScriptedPolicy)
        self.assertEqual(len(p.act("")), 4)

    def test_build_remote(self):
        p = get_policy({"type": "remote", "base_url": "http://localhost:9999", "action_dim": 7})
        self.assertIsInstance(p, RemotePolicy)

    def test_build_llm_policy(self):
        llm = get_llm(LLMConfig(provider="mock"))
        p = get_policy({"type": "llm", "action_dim": 6}, llm=llm)
        self.assertEqual(p.__class__.__name__, "PolicyAgent")

    def test_llm_policy_without_llm_raises(self):
        with self.assertRaises(ValueError):
            get_policy({"type": "llm"})

    def test_unknown_type_raises(self):
        with self.assertRaises(KeyError):
            get_policy({"type": "nope"})


class TestRemotePolicyRoundTrip(unittest.TestCase):
    """RemotePolicy against the harness's own policy server."""

    def test_begin_and_act_over_http(self):
        import threading
        from http.server import ThreadingHTTPServer
        from unittest import mock

        from harness.serving import PolicySessionManager, make_handler

        reply = json.dumps({"action": "joints", "joint_positions": [0.1, 0.2, 0.3]})
        llm = get_llm(LLMConfig(provider="mock", extra={"responses": [reply] * 4}))
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(PolicySessionManager(llm, action_dim=4)))
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

        # a local server must not be routed through a developer's proxy env vars
        no_proxy = {"NO_PROXY": "*", "no_proxy": "*", "http_proxy": "", "https_proxy": "", "all_proxy": ""}
        try:
            with mock.patch.dict("os.environ", no_proxy, clear=False):
                policy = RemotePolicy(f"http://127.0.0.1:{port}", action_dim=4)
                policy.begin("pick up the banana")
                vec = policy.act("ee=(0.1, 0.2) gripper=open")
        finally:
            httpd.shutdown()

        self.assertEqual(vec.shape, (4,))
        np.testing.assert_allclose(vec[:3], [0.1, 0.2, 0.3], rtol=1e-5)

    def test_act_reports_server_failure(self):
        from unittest import mock

        from harness.policies.remote import PolicyServerError

        no_proxy = {"NO_PROXY": "*", "no_proxy": "*", "http_proxy": "", "https_proxy": "", "all_proxy": ""}
        with mock.patch.dict("os.environ", no_proxy, clear=False):
            # nothing is listening on this port -> transient, retried, then raises
            policy = RemotePolicy("http://127.0.0.1:1", action_dim=4, retries=0, timeout=2.0)
            with self.assertRaises(PolicyServerError):
                policy.act("state")


class TestPolicyAsToolInController(unittest.TestCase):
    def _controller(self, responses, policy, **kw):
        llm = get_llm(LLMConfig(provider="mock", extra={"responses": responses}))
        return LLMController(llm, mode="tools", policy=policy, max_steps=6, **kw)

    def test_controller_builds_policy_toolset(self):
        agent = self._controller([], ScriptedPolicy())
        self.assertIn("run_policy", [t.name for t in agent._tools_full])

    def test_inner_policy_steps_land_in_the_episode(self):
        env = _env()
        responses = [
            json.dumps({"tool": "run_policy", "args": {"instruction": "go right", "steps": 4}}),
            json.dumps({"tool": "done", "args": {}}),
        ]
        agent = self._controller(responses, ScriptedPolicy([[0.05, 0.0]], action_dim=2))
        ep = agent.run(env, seed=0)
        # 4 policy steps recorded, plus the stop action from done()
        self.assertEqual(len([a for a in ep.actions if a.kind != "stop"]), 4)
        self.assertEqual(len(ep.rewards), 4)

    def test_trace_records_one_entry_per_tool_call(self):
        from harness.viz.recorder import TraceRecorder

        env = _env()
        rec = TraceRecorder(capture_frames=False)
        responses = [
            json.dumps({"tool": "run_policy", "args": {"instruction": "a", "steps": 3}}),
            json.dumps({"tool": "run_policy", "args": {"instruction": "b", "steps": 2}}),
            json.dumps({"tool": "done", "args": {}}),
        ]
        agent = self._controller(responses, ScriptedPolicy([[0.01, 0.0]], action_dim=2), recorder=rec)
        agent.run(env, seed=0)
        policy_steps = [s for s in rec.steps if s.action.get("tool") == "run_policy"]
        self.assertEqual(len(policy_steps), 2)
        self.assertEqual(policy_steps[0].action["steps"], 3)
        self.assertEqual(policy_steps[1].action["steps"], 2)

    def test_solves_reach_via_policy_delegation(self):
        env = _reach_env()
        responses = [json.dumps({"tool": "run_policy", "args": {"instruction": "reach it", "steps": 200}})]
        agent = self._controller(responses, _homing_policy(env))
        ep = agent.run(env, seed=0)
        self.assertTrue(ep.success)

    def test_bad_tool_args_do_not_crash_the_episode(self):
        env = _env()
        responses = [
            json.dumps({"tool": "run_policy", "args": {"bogus_kwarg": 1}}),
            json.dumps({"tool": "done", "args": {}}),
        ]
        agent = self._controller(responses, ScriptedPolicy([[0.0, 0.0]], action_dim=2))
        ep = agent.run(env, seed=0)  # must not raise
        self.assertIsNotNone(ep)

    def test_prompt_switches_to_planner_framing(self):
        from harness.agent.prompts import build_tools_system_prompt

        planner = build_tools_system_prompt(task="t", tools=get_policy_tools(ScriptedPolicy()))
        self.assertIn("high-level planner", planner)
        self.assertIn("run_policy", planner)

        plain = build_tools_system_prompt(task="t", tools=get_default_tools())
        self.assertNotIn("high-level planner", plain)


if __name__ == "__main__":
    unittest.main()
