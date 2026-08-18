"""Tests for the local-CLI LLM backend.

Everything here mocks subprocess.run, so the suite passes on machines with no
`claude` binary and no network. The one test that really spawns the CLI is
opt-in via RUN_CLAUDE_CLI_SMOKE=1 (it costs seconds and needs local auth).
"""
import json
import os
import shutil
import subprocess
import unittest
from unittest import mock

from harness.config import LLMConfig
from harness.llm.base import ChatMessage, LLMError, TransientLLMError
from harness.llm.claude_code import ClaudeCodeClient
from harness.llm.registry import get_llm

# A trimmed copy of a real `claude -p --output-format json` envelope.
GOOD_ENVELOPE = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "stop_reason": "end_turn",
    "result": '{"action": "stop"}',
    "session_id": "c8aacd9b-ae91-4982-a8e3-df8ae307d555",
    "total_cost_usd": 0.0347,
    "duration_ms": 3042,
    "usage": {
        "input_tokens": 2,
        "output_tokens": 9,
        "cache_read_input_tokens": 3289,
        "cache_creation_input_tokens": 5431,
        "service_tier": "standard",
        "iterations": [{"input_tokens": 2, "output_tokens": 9}],
    },
    "modelUsage": {
        # The CLI routes side work (titles, classifiers) to a small model, so
        # more than one model shows up for a single turn.
        "claude-haiku-4-5-20251001": {"inputTokens": 904, "outputTokens": 12},
        "claude-sonnet-5": {"inputTokens": 2, "outputTokens": 9},
    },
}


def completed(stdout, *, returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr)


def client(**extra):
    extra.setdefault("retries", 0)  # keep failure tests from sleeping through backoff
    return ClaudeCodeClient(model="sonnet", timeout=30.0, extra=extra)


class TestClaudeCodeParsing(unittest.TestCase):
    def test_a_well_formed_json_envelope_parses_into_content_model_and_usage(self):
        with mock.patch(
            "harness.llm.claude_code.subprocess.run",
            return_value=completed(json.dumps(GOOD_ENVELOPE)),
        ):
            resp = client().complete([ChatMessage.user("hi")])

        self.assertEqual(resp.content, '{"action": "stop"}')
        # Matched against the top-level usage totals, not just "first key".
        self.assertEqual(resp.model, "claude-sonnet-5")
        self.assertEqual(resp.finish_reason, "end_turn")
        self.assertEqual(resp.usage["input_tokens"], 2)
        self.assertEqual(resp.usage["output_tokens"], 9)
        self.assertEqual(resp.usage["cache_read_input_tokens"], 3289)
        self.assertEqual(resp.usage["cache_creation_input_tokens"], 5431)
        self.assertAlmostEqual(resp.usage["total_cost_usd"], 0.0347)
        self.assertEqual(resp.raw["session_id"], GOOD_ENVELOPE["session_id"])

    def test_usage_stays_empty_when_the_cli_reports_no_numbers(self):
        envelope = {k: v for k, v in GOOD_ENVELOPE.items() if k not in ("usage", "total_cost_usd")}
        with mock.patch(
            "harness.llm.claude_code.subprocess.run", return_value=completed(json.dumps(envelope))
        ):
            resp = client().complete([ChatMessage.user("hi")])
        self.assertEqual(resp.usage, {})

    def test_model_falls_back_to_the_configured_value_when_the_cli_reports_none(self):
        envelope = {k: v for k, v in GOOD_ENVELOPE.items() if k != "modelUsage"}
        with mock.patch(
            "harness.llm.claude_code.subprocess.run", return_value=completed(json.dumps(envelope))
        ):
            resp = client().complete([ChatMessage.user("hi")])
        self.assertEqual(resp.model, "sonnet")


class TestClaudeCodeInvocation(unittest.TestCase):
    def _capture(self, messages, **extra):
        seen = {}

        def fake_run(argv, **kwargs):
            seen["argv"] = argv
            seen["kwargs"] = kwargs
            return completed(json.dumps(GOOD_ENVELOPE))

        with mock.patch("harness.llm.claude_code.subprocess.run", side_effect=fake_run):
            client(**extra).complete(messages)
        return seen

    def test_role_flattening_keeps_system_user_and_assistant_in_order(self):
        seen = self._capture(
            [
                ChatMessage.system("you are a robot planner"),
                ChatMessage.user("what next"),
                ChatMessage.assistant('{"action": "move"}'),
            ]
        )
        prompt = seen["kwargs"]["input"]
        for needle in ("System:", "you are a robot planner", "User:", "what next", "Assistant:", '{"action": "move"}'):
            self.assertIn(needle, prompt)
        self.assertLess(prompt.index("System:"), prompt.index("User:"))
        self.assertLess(prompt.index("User:"), prompt.index("Assistant:"))

    def test_it_runs_headless_with_json_output_and_no_tools(self):
        seen = self._capture([ChatMessage.user("hi")])
        argv = seen["argv"]
        self.assertEqual(argv[0], "claude")
        self.assertIn("--print", argv)
        self.assertEqual(argv[argv.index("--output-format") + 1], "json")
        self.assertEqual(argv[argv.index("--tools") + 1], "")  # every built-in tool off
        self.assertIn("--disable-slash-commands", argv)
        self.assertIn("--strict-mcp-config", argv)
        self.assertEqual(argv[argv.index("--model") + 1], "sonnet")

    def test_the_timeout_is_handed_to_subprocess_run(self):
        seen = self._capture([ChatMessage.user("hi")])
        self.assertEqual(seen["kwargs"]["timeout"], 30.0)

    def test_a_custom_cli_path_and_extra_args_are_honoured(self):
        seen = self._capture([ChatMessage.user("hi")], cli_path="/opt/bin/claude", extra_args=["--effort", "low"])
        self.assertEqual(seen["argv"][0], "/opt/bin/claude")
        self.assertEqual(seen["argv"][-2:], ["--effort", "low"])

    def test_it_does_not_spawn_the_cli_in_the_callers_project_directory(self):
        # CLAUDE.md auto-discovery would otherwise inject repo instructions
        # into every completion.
        seen = self._capture([ChatMessage.user("hi")])
        self.assertNotEqual(os.path.realpath(seen["kwargs"]["cwd"]), os.path.realpath(os.getcwd()))


class TestClaudeCodeImages(unittest.TestCase):
    def test_an_image_bearing_message_raises_instead_of_dropping_the_image(self):
        msg = ChatMessage.user_vision("what do you see", "aGVsbG8=")
        with mock.patch("harness.llm.claude_code.subprocess.run") as run:
            with self.assertRaises(LLMError) as ctx:
                client().complete([msg])
        run.assert_not_called()  # fail before spending a CLI call
        self.assertIn("cannot send images", str(ctx.exception))

    def test_on_image_warn_degrades_loudly_rather_than_silently(self):
        msg = ChatMessage.user_vision("what do you see", "aGVsbG8=")
        seen = {}

        def fake_run(argv, **kwargs):
            seen["input"] = kwargs["input"]
            return completed(json.dumps(GOOD_ENVELOPE))

        with mock.patch("harness.llm.claude_code.subprocess.run", side_effect=fake_run):
            with self.assertLogs("harness.llm.claude_code", level="WARNING") as logs:
                resp = client(on_image="warn").complete([msg])

        self.assertEqual(resp.content, '{"action": "stop"}')
        self.assertIn("cannot send images", "\n".join(logs.output))
        self.assertIn("image(s) omitted", seen["input"])  # the model is told, too
        self.assertIn("what do you see", seen["input"])


class TestClaudeCodeFailures(unittest.TestCase):
    def test_a_non_zero_exit_raises_with_the_stderr_included(self):
        stderr = "[claude-code:unrecognized_model] {\"model\":\"nope\"}"
        envelope = dict(GOOD_ENVELOPE, is_error=True, api_error_status=404, result="no such model")
        with mock.patch(
            "harness.llm.claude_code.subprocess.run",
            return_value=completed(json.dumps(envelope), returncode=1, stderr=stderr),
        ):
            with self.assertRaises(LLMError) as ctx:
                client().complete([ChatMessage.user("hi")])
        msg = str(ctx.exception)
        self.assertIn("exited 1", msg)
        self.assertIn("unrecognized_model", msg)
        self.assertIn("404", msg)
        # A 404 is a configuration bug; retrying it just burns sweep time.
        self.assertNotIsInstance(ctx.exception, TransientLLMError)

    def test_long_stderr_is_truncated_rather_than_dumped_whole(self):
        stderr = "x" * 5000
        with mock.patch(
            "harness.llm.claude_code.subprocess.run",
            return_value=completed("", returncode=2, stderr=stderr),
        ):
            with self.assertRaises(LLMError) as ctx:
                client().complete([ChatMessage.user("hi")])
        self.assertLess(len(str(ctx.exception)), 2000)
        self.assertIn("chars]", str(ctx.exception))

    def test_a_server_side_overload_is_retryable(self):
        envelope = dict(GOOD_ENVELOPE, is_error=True, api_error_status=503)
        with mock.patch(
            "harness.llm.claude_code.subprocess.run",
            return_value=completed(json.dumps(envelope), returncode=1, stderr="upstream boom"),
        ):
            with self.assertRaises(TransientLLMError):
                client().complete([ChatMessage.user("hi")])

    def test_an_error_envelope_on_a_zero_exit_still_raises(self):
        envelope = dict(GOOD_ENVELOPE, is_error=True, api_error_status=401, result="auth expired")
        with mock.patch(
            "harness.llm.claude_code.subprocess.run", return_value=completed(json.dumps(envelope))
        ):
            with self.assertRaises(LLMError) as ctx:
                client().complete([ChatMessage.user("hi")])
        self.assertIn("auth expired", str(ctx.exception))

    def test_a_timeout_raises_the_retryable_error_type(self):
        with mock.patch(
            "harness.llm.claude_code.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=30.0, stderr=b"partial"),
        ):
            with self.assertRaises(TransientLLMError) as ctx:
                client().complete([ChatMessage.user("hi")])
        self.assertIn("timed out", str(ctx.exception))

    def test_unparseable_stdout_raises_rather_than_returning_empty_content(self):
        with mock.patch(
            "harness.llm.claude_code.subprocess.run",
            return_value=completed("Welcome to Claude Code!\nnot json at all"),
        ):
            with self.assertRaises(TransientLLMError) as ctx:
                client().complete([ChatMessage.user("hi")])
        self.assertIn("unparseable", str(ctx.exception))

    def test_a_success_envelope_with_no_result_field_raises(self):
        envelope = {k: v for k, v in GOOD_ENVELOPE.items() if k != "result"}
        with mock.patch(
            "harness.llm.claude_code.subprocess.run", return_value=completed(json.dumps(envelope))
        ):
            with self.assertRaises(LLMError):
                client().complete([ChatMessage.user("hi")])

    def test_a_missing_cli_is_not_retried(self):
        with mock.patch(
            "harness.llm.claude_code.subprocess.run", side_effect=FileNotFoundError("no claude")
        ):
            with self.assertRaises(LLMError) as ctx:
                client(cli_path="/nope/claude").complete([ChatMessage.user("hi")])
        self.assertNotIsInstance(ctx.exception, TransientLLMError)
        self.assertIn("/nope/claude", str(ctx.exception))

    def test_an_empty_prompt_is_refused_before_spawning_anything(self):
        with mock.patch("harness.llm.claude_code.subprocess.run") as run:
            with self.assertRaises(LLMError):
                client().complete([ChatMessage.user("   ")])
        run.assert_not_called()

    def test_a_transient_failure_is_retried_when_retries_are_enabled(self):
        attempts = []

        def fake_run(argv, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                return completed("garbage")
            return completed(json.dumps(GOOD_ENVELOPE))

        c = ClaudeCodeClient(model="sonnet", extra={"retries": 1})
        with mock.patch("harness.llm.claude_code.subprocess.run", side_effect=fake_run):
            with mock.patch("harness.llm.retry.time.sleep"):
                resp = c.complete([ChatMessage.user("hi")])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(resp.content, '{"action": "stop"}')


class TestClaudeCodeRegistry(unittest.TestCase):
    def test_the_cli_provider_aliases_all_resolve_to_the_cli_client(self):
        for provider in ("claude_code", "claude-code", "cli", "CLAUDE_CODE"):
            with self.subTest(provider=provider):
                self.assertIsInstance(get_llm(LLMConfig(provider=provider)), ClaudeCodeClient)

    def test_the_plain_claude_provider_still_uses_the_http_api(self):
        from harness.llm.anthropic import AnthropicClient

        for provider in ("claude", "anthropic"):
            with self.subTest(provider=provider):
                llm = get_llm(LLMConfig(provider=provider, api_key="x"))
                self.assertIsInstance(llm, AnthropicClient)
                self.assertNotIsInstance(llm, ClaudeCodeClient)

    def test_config_fields_reach_the_client(self):
        cfg = LLMConfig(provider="cli", model="opus", timeout=12.0, extra={"on_image": "warn"})
        llm = get_llm(cfg)
        self.assertEqual(llm.name, "claude-code(opus)")
        self.assertEqual(llm._timeout, 12.0)


@unittest.skipUnless(
    os.environ.get("RUN_CLAUDE_CLI_SMOKE") == "1" and shutil.which("claude"),
    "real CLI smoke test: set RUN_CLAUDE_CLI_SMOKE=1 and have `claude` on PATH",
)
class TestClaudeCodeRealCLI(unittest.TestCase):
    """Opt-in: actually spawns the CLI. Slow (seconds) and needs local auth."""

    def test_the_real_cli_round_trips_a_fixed_token(self):
        c = ClaudeCodeClient(timeout=180.0, extra={"retries": 0})
        resp = c.complete(
            [
                ChatMessage.system("Answer with the requested token and nothing else."),
                ChatMessage.user("Reply with exactly this one word: PONGFIXED"),
            ]
        )
        self.assertIn("PONGFIXED", resp.content)
        self.assertTrue(resp.model)


if __name__ == "__main__":
    unittest.main()
