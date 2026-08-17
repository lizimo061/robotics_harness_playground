import json
import unittest
from unittest import mock

from harness.llm.base import ChatMessage
from harness.llm.mock import MockLLMClient
from harness.llm.openai_compat import OpenAICompatClient


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        if isinstance(payload, (dict, list)):
            self.text = json.dumps(payload)
        else:
            self.text = str(payload)

    def json(self):
        return self._payload


class TestOpenAICompatClient(unittest.TestCase):
    def test_complete_sends_and_parses(self):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse(200, {
                "choices": [{
                    "message": {"role": "assistant", "content": '{"action": "stop"}'},
                    "finish_reason": "stop",
                }],
                "model": "deepseek-chat",
                "usage": {"total_tokens": 12},
            })

        client = OpenAICompatClient(base_url="http://localhost:1/v1", model="deepseek-chat", api_key="secret")
        with mock.patch("harness.llm.openai_compat.httpx.post", side_effect=fake_post):
            resp = client.complete([ChatMessage.user("hi")])

        self.assertEqual(resp.content, '{"action": "stop"}')
        self.assertEqual(resp.model, "deepseek-chat")
        self.assertEqual(captured["url"], "http://localhost:1/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(captured["json"]["model"], "deepseek-chat")
        self.assertEqual(captured["json"]["messages"][0]["role"], "user")

    def test_http_error_raises(self):
        def fake_post(url, headers, json, timeout):
            return FakeResponse(401, {"error": {"message": "bad key"}})

        client = OpenAICompatClient(base_url="http://x/v1", model="m", api_key="k")
        with mock.patch("harness.llm.openai_compat.httpx.post", side_effect=fake_post):
            with self.assertRaises(Exception):
                client.complete([ChatMessage.user("hi")])


class TestMockLLMClient(unittest.TestCase):
    def test_scripted_responses(self):
        c = MockLLMClient(extra={"script": [{"a": 1}, {"b": 2}], "fallback": '{"a": 0}'})
        self.assertEqual(json.loads(c.complete([]).content), {"a": 1})
        self.assertEqual(json.loads(c.complete([]).content), {"b": 2})
        self.assertEqual(json.loads(c.complete([]).content), {"a": 0})  # fallback


if __name__ == "__main__":
    unittest.main()
