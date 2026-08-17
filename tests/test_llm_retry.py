import unittest
from unittest import mock

from harness.llm.base import ChatMessage, TransientLLMError
from harness.llm.retry import with_retries


class TestRetry(unittest.TestCase):
    def test_success_first_try(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        self.assertEqual(with_retries(fn, retries=2, base_delay=0), "ok")
        self.assertEqual(len(calls), 1)

    def test_retries_then_succeeds(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise TransientLLMError("retry")
            return "ok"

        self.assertEqual(with_retries(fn, retries=3, base_delay=0), "ok")
        self.assertEqual(len(calls), 3)

    def test_exhausts(self):
        def fn():
            raise TransientLLMError("always")

        with self.assertRaises(TransientLLMError):
            with_retries(fn, retries=2, base_delay=0)


class TestAnthropicClient(unittest.TestCase):
    def test_complete_parses(self):
        from harness.llm.anthropic import AnthropicClient

        def fake_post(url, headers, json, timeout):
            class R:
                status_code = 200
                text = "{}"

                def json(self):
                    return {
                        "content": [{"type": "text", "text": "hi"}],
                        "model": "claude-x",
                        "stop_reason": "end_turn",
                        "usage": {},
                    }

            return R()

        client = AnthropicClient(model="claude-x", api_key="k", base_url="http://x")
        with mock.patch("harness.llm.anthropic.httpx.post", side_effect=fake_post):
            resp = client.complete([ChatMessage.user("hello")])
        self.assertEqual(resp.content, "hi")
        self.assertEqual(resp.model, "claude-x")


if __name__ == "__main__":
    unittest.main()
