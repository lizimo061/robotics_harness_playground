import unittest

import httpx

from harness.agent import LLMController
from harness.config import LLMConfig
from harness.envs.tabletop import TabletopEnv
from harness.llm import get_llm
from harness.llm.anthropic import AnthropicClient
from harness.llm.base import ChatMessage
from harness.llm.capabilities import TokenUsage, estimate_cost, get_caps
from harness.tasks import generate_task


class TestModelCaps(unittest.TestCase):
    def test_current_claude_models_reject_sampling_params(self):
        for model in ("claude-opus-5", "claude-sonnet-5", "claude-opus-4-7", "claude-fable-5"):
            self.assertFalse(get_caps(model).sampling_params, model)

    def test_older_claude_models_still_accept_sampling_params(self):
        for model in ("claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"):
            self.assertTrue(get_caps(model).sampling_params, model)

    def test_unknown_model_gets_permissive_default(self):
        caps = get_caps("some-self-hosted-llama")
        self.assertTrue(caps.sampling_params)
        self.assertIsNone(caps.price_in)

    def test_prefix_match_picks_longest(self):
        # a dated/suffixed variant resolves to its base model's caps
        self.assertFalse(get_caps("claude-opus-5-some-suffix").sampling_params)

    def test_empty_model_does_not_raise(self):
        self.assertTrue(get_caps("").sampling_params)


class TestTokenUsage(unittest.TestCase):
    def test_parses_anthropic_shape(self):
        u = TokenUsage.from_raw({
            "input_tokens": 100, "output_tokens": 20,
            "cache_read_input_tokens": 900, "cache_creation_input_tokens": 50,
        })
        self.assertEqual((u.input_tokens, u.output_tokens), (100, 20))
        self.assertEqual((u.cache_read_tokens, u.cache_write_tokens), (900, 50))

    def test_parses_openai_shape(self):
        u = TokenUsage.from_raw({"prompt_tokens": 80, "completion_tokens": 12})
        self.assertEqual((u.input_tokens, u.output_tokens), (80, 12))

    def test_parses_openai_cached_tokens(self):
        u = TokenUsage.from_raw({
            "prompt_tokens": 80, "completion_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 64},
        })
        self.assertEqual(u.cache_read_tokens, 64)

    def test_empty_and_none_are_zero(self):
        self.assertEqual(TokenUsage.from_raw(None).total, 0)
        self.assertEqual(TokenUsage.from_raw({}).total, 0)

    def test_addition_accumulates(self):
        a = TokenUsage(input_tokens=10, output_tokens=1)
        b = TokenUsage(input_tokens=5, output_tokens=2)
        self.assertEqual((a + b).input_tokens, 15)
        self.assertEqual((a + b).output_tokens, 3)


class TestCost(unittest.TestCase):
    def test_priced_model(self):
        # opus-5 at $5/1M in, $25/1M out
        cost = estimate_cost("claude-opus-5", {"input_tokens": 1_000_000, "output_tokens": 1_000_000})
        self.assertAlmostEqual(cost, 30.0, places=6)

    def test_cache_reads_are_cheaper_than_fresh_input(self):
        fresh = estimate_cost("claude-opus-5", {"input_tokens": 1_000_000})
        cached = estimate_cost("claude-opus-5", {"cache_read_input_tokens": 1_000_000})
        self.assertAlmostEqual(cached, fresh * 0.1, places=6)

    def test_unknown_price_returns_none_not_zero(self):
        self.assertIsNone(estimate_cost("some-self-hosted-llama", {"input_tokens": 1000}))


class TestAnthropicPayload(unittest.TestCase):
    """The regression that motivated this module: temperature caused a 400."""

    def _payload_for(self, model):
        cap = {}
        real_post = httpx.post

        def fake_post(url, headers=None, json=None, timeout=None):
            cap["payload"] = json
            raise httpx.TransportError("blocked")

        httpx.post = fake_post
        try:
            c = AnthropicClient(model=model, api_key="k", temperature=0.2, max_tokens=64)
            try:
                c.complete([ChatMessage.system("s"), ChatMessage.user("u")])
            except Exception:
                pass
        finally:
            httpx.post = real_post
        return cap["payload"]

    def test_temperature_omitted_for_models_that_reject_it(self):
        for model in ("claude-opus-5", "claude-sonnet-5"):
            self.assertNotIn("temperature", self._payload_for(model), model)

    def test_temperature_sent_for_models_that_accept_it(self):
        for model in ("claude-opus-4-6", "claude-haiku-4-5"):
            self.assertIn("temperature", self._payload_for(model), model)

    def test_core_fields_survive(self):
        p = self._payload_for("claude-opus-5")
        for key in ("model", "messages", "max_tokens", "system"):
            self.assertIn(key, p)


class TestControllerAccounting(unittest.TestCase):
    def _run(self):
        llm = get_llm(LLMConfig(provider="mock", extra={"responses": ['{"action": "stop"}']}))
        env = TabletopEnv(task_spec=generate_task("pick_place", seed=1))
        return LLMController(llm, mode="json", max_steps=3).run(env, seed=0)

    def test_episode_carries_usage_and_call_count(self):
        ep = self._run()
        self.assertIn("usage", ep.metadata)
        self.assertIn("llm_calls", ep.metadata)
        self.assertGreaterEqual(ep.metadata["llm_calls"], 1)

    def test_unpriced_model_reports_cost_none_not_zero(self):
        # the mock provider has no published price; None must not become 0.0
        self.assertIsNone(self._run().metadata["cost_usd"])

    def test_counters_reset_between_episodes(self):
        llm = get_llm(LLMConfig(provider="mock", extra={"responses": ['{"action": "stop"}']}))
        agent = LLMController(llm, mode="json", max_steps=3)
        env = TabletopEnv(task_spec=generate_task("pick_place", seed=1))
        first = agent.run(env, seed=0).metadata["llm_calls"]
        second = agent.run(env, seed=1).metadata["llm_calls"]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
