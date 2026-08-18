"""Vision gating: a run that needs pixels may not be pointed at a blind model.

Two configurations in this harness require image input:

  * ``use_vision=True`` attaches the rendered camera frame to every observation;
  * ``tier="perception"`` withdraws the ground-truth object queries, so the only
    way left to locate an object is ``detect``/``point_at`` over that frame.

Paired with a text-only model -- DeepSeek's ``deepseek-chat`` and
``deepseek-reasoner`` are the ones this repo's configs actually name -- both used
to run to completion and produce a full results row: a success rate, a cost, a
failure mode. Nothing on the way out recorded that the agent had been asked to
look at something it never received, so the zero read as incapability rather than
as a broken experiment. These tests exist to keep that failure loud.
"""
import unittest

from harness.agent.llm_controller import LLMController, is_offline_llm, model_id_of
from harness.config import AgentConfig, EnvConfig, HarnessConfig, LLMConfig
from harness.llm import get_llm
from harness.llm.base import LLMResponse
from harness.llm.capabilities import (
    VisionUnsupportedError,
    check_vision_config,
    get_caps,
    supports_vision,
)


class _NamedLLM:
    """A client that names its model, the way every `get_llm` client does."""

    def __init__(self, model: str):
        self._model = model
        self.name = f"named({model})"

    def complete(self, messages, **kw):
        return LLMResponse(content='{"action": "stop"}', model=self._model,
                           usage={}, raw={}, finish_reason="stop")


def _blind() -> _NamedLLM:
    return _NamedLLM("deepseek-chat")


def _sighted() -> _NamedLLM:
    return _NamedLLM("claude-opus-5")


class TestTheCapabilityTable(unittest.TestCase):
    """Spot-checks for the models this repo actually configures."""

    def test_deepseek_chat_is_recorded_as_text_only(self):
        self.assertFalse(get_caps("deepseek-chat").vision)
        self.assertFalse(supports_vision("deepseek-chat"))

    def test_deepseek_reasoner_is_recorded_as_text_only(self):
        self.assertFalse(get_caps("deepseek-reasoner").vision)
        self.assertFalse(supports_vision("deepseek-reasoner"))

    def test_claude_models_are_recorded_as_vision_capable(self):
        for model in ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"):
            self.assertTrue(supports_vision(model), model)

    def test_a_dated_claude_variant_inherits_vision_through_prefix_match(self):
        # get_caps resolves suffixed ids by longest prefix; vision must ride along
        # or a pinned snapshot id would be refused as unverified.
        self.assertTrue(supports_vision("claude-opus-5-20260101"))

    def test_an_unknown_model_does_not_silently_claim_vision(self):
        """The default must be the answer that fails loudly, not the convenient one."""
        self.assertFalse(supports_vision("some-self-hosted-llama"))
        self.assertFalse(supports_vision("gpt-nonexistent-9"))
        self.assertFalse(supports_vision(""))

    def test_recording_vision_did_not_disturb_the_other_capabilities(self):
        caps = get_caps("claude-opus-5")
        self.assertFalse(caps.sampling_params)
        self.assertEqual(caps.price_in, 5.0)


class TestCheckVisionConfig(unittest.TestCase):
    def test_a_text_only_model_with_use_vision_is_refused(self):
        with self.assertRaises(VisionUnsupportedError):
            check_vision_config("deepseek-chat", use_vision=True)

    def test_a_text_only_model_with_the_perception_tier_is_refused(self):
        with self.assertRaises(VisionUnsupportedError):
            check_vision_config("deepseek-reasoner", tier="perception")

    def test_a_privileged_text_only_run_is_the_normal_case_and_is_allowed(self):
        check_vision_config("deepseek-chat", use_vision=False, tier="privileged")

    def test_a_vision_model_is_allowed_in_both_configurations(self):
        check_vision_config("claude-opus-5", use_vision=True)
        check_vision_config("claude-opus-5", tier="perception")

    def test_the_error_says_what_was_requested_and_how_to_fix_it(self):
        with self.assertRaises(VisionUnsupportedError) as cm:
            check_vision_config("deepseek-chat", use_vision=True, tier="perception")
        msg = str(cm.exception)
        self.assertIn("deepseek-chat", msg)
        self.assertIn("use_vision=True", msg)
        self.assertIn("perception", msg)
        self.assertIn("privileged", msg)

    def test_an_unverified_model_is_refused_but_not_called_blind(self):
        with self.assertRaises(VisionUnsupportedError) as cm:
            check_vision_config("some-self-hosted-llama", use_vision=True)
        self.assertIn("unverified", str(cm.exception))

    def test_a_recorded_text_only_model_is_named_as_such_not_as_unverified(self):
        """A reader must not be told "unverified" when the table says text-only."""
        with self.assertRaises(VisionUnsupportedError) as cm:
            check_vision_config("deepseek-chat-2026-preview", use_vision=True)
        self.assertIn("recorded as text-only", str(cm.exception))


class TestControllerRefusesBlindVisionRuns(unittest.TestCase):
    def test_a_text_only_model_with_use_vision_raises_and_names_the_model(self):
        with self.assertRaises(VisionUnsupportedError) as cm:
            LLMController(_blind(), mode="tools", use_vision=True)
        self.assertIn("deepseek-chat", str(cm.exception))

    def test_a_text_only_model_with_the_perception_tier_raises(self):
        with self.assertRaises(VisionUnsupportedError) as cm:
            LLMController(_blind(), mode="tools", tier="perception")
        self.assertIn("deepseek-chat", str(cm.exception))

    def test_a_vision_model_with_use_vision_is_allowed(self):
        ctrl = LLMController(_sighted(), mode="tools", use_vision=True)
        self.assertTrue(ctrl._capture_frames)

    def test_a_vision_model_with_the_perception_tier_is_allowed(self):
        LLMController(_sighted(), mode="tools", tier="perception")

    def test_the_privileged_text_only_default_is_untouched(self):
        # every result in this repo so far was measured this way; it must not break
        LLMController(_blind(), mode="tools")
        LLMController(_blind(), mode="json", tier="privileged", use_vision=False)

    def test_it_refuses_before_touching_an_environment(self):
        """The point of a construction-time check: no env, no episode, no results row."""
        with self.assertRaises(VisionUnsupportedError):
            LLMController(_blind(), mode="tools", use_vision=True, max_steps=1)

    def test_run_re_checks_a_tier_switched_after_construction(self):
        from harness.envs.tabletop import TabletopEnv

        ctrl = LLMController(_blind(), mode="tools", max_steps=1, task_description="t")
        ctrl._tier = "perception"  # what an examples script does by hand
        env = TabletopEnv(task="pick_place")
        try:
            with self.assertRaises(VisionUnsupportedError):
                ctrl.run(env, seed=0)
        finally:
            env.close()


class TestTheEscapeHatch(unittest.TestCase):
    def test_allow_blind_vision_is_explicit_and_permits_the_run(self):
        ctrl = LLMController(_blind(), mode="tools", use_vision=True,
                             allow_blind_vision=True)
        self.assertTrue(ctrl._allow_blind_vision)

    def test_allow_blind_vision_also_covers_the_perception_tier(self):
        LLMController(_blind(), mode="tools", tier="perception",
                      allow_blind_vision=True)

    def test_it_is_off_by_default_so_the_hatch_cannot_be_taken_by_accident(self):
        self.assertFalse(LLMController(_sighted(), mode="tools")._allow_blind_vision)

    def test_the_offline_mock_provider_is_exempt(self):
        for provider in ("mock", "offline"):
            llm = get_llm(LLMConfig(provider=provider))
            self.assertTrue(is_offline_llm(llm), provider)
            LLMController(llm, mode="tools", use_vision=True)

    def test_the_offline_mock_still_runs_a_vision_episode_end_to_end(self):
        from harness.envs.tabletop import TabletopEnv

        llm = get_llm(LLMConfig(provider="mock",
                                extra={"responses": ['{"tool": "done", "args": {}}']}))
        env = TabletopEnv(task="pick_place")
        ep = LLMController(llm, mode="tools", max_steps=1, use_vision=True,
                           task_description="t").run(env, seed=0)
        env.close()
        self.assertGreaterEqual(ep.metadata["llm_calls"], 1)

    def test_a_client_that_names_no_model_is_unchecked_but_says_so(self):
        """In-process test doubles cannot be looked up; that is logged, not hidden."""
        class _Anonymous:
            name = "stub"

            def complete(self, messages, **kw):
                return LLMResponse(content="{}", model="stub", usage={}, raw={})

        llm = _Anonymous()
        self.assertIsNone(model_id_of(llm))
        with self.assertLogs("harness.agent.llm_controller", level="WARNING") as cm:
            LLMController(llm, mode="tools", use_vision=True)
        self.assertTrue(any("cannot verify image support" in m for m in cm.output))


class TestModelResolution(unittest.TestCase):
    def test_it_reads_the_model_a_real_client_was_built_with(self):
        from harness.llm.anthropic import AnthropicClient

        self.assertEqual(model_id_of(AnthropicClient(model="claude-opus-5", api_key="k")),
                         "claude-opus-5")

    def test_a_deepseek_client_reports_the_alias_it_will_request(self):
        llm = get_llm(LLMConfig(provider="deepseek", api_key="k"))
        self.assertEqual(model_id_of(llm), "deepseek-chat")

    def test_only_the_mock_module_counts_as_offline(self):
        self.assertFalse(is_offline_llm(_blind()))


class TestRunnerLevelGating(unittest.TestCase):
    """The runner owns `agent.tier`, which it does not forward to the controller."""

    def _cfg(self, **agent_kw) -> HarnessConfig:
        return HarnessConfig(
            llm=LLMConfig(provider="deepseek", model="deepseek-chat", api_key="k"),
            env=EnvConfig(name="toy_tabletop", task="pick_place"),
            agent=AgentConfig(mode="tools", max_steps=1, **agent_kw),
        )

    def _check(self, cfg) -> None:
        from harness.runner import _check_vision_config

        _check_vision_config(cfg, get_llm(cfg.llm))

    def test_a_perception_tier_config_on_a_text_only_model_is_refused(self):
        with self.assertRaises(VisionUnsupportedError) as cm:
            self._check(self._cfg(tier="perception"))
        self.assertIn("deepseek-chat", str(cm.exception))

    def test_a_use_vision_config_on_a_text_only_model_is_refused(self):
        with self.assertRaises(VisionUnsupportedError):
            self._check(self._cfg(use_vision=True))

    def test_the_default_privileged_text_config_passes(self):
        self._check(self._cfg())

    def test_a_blank_model_falls_back_to_the_provider_default(self):
        cfg = self._cfg(use_vision=True)
        cfg.llm.model = ""  # provider default fills in deepseek-chat
        with self.assertRaises(VisionUnsupportedError):
            self._check(cfg)

    def test_the_extra_escape_hatch_is_honoured(self):
        self._check(self._cfg(tier="perception", extra={"allow_blind_vision": True}))

    def test_a_vision_model_passes_both_configurations(self):
        for agent_kw in ({"use_vision": True}, {"tier": "perception"}):
            cfg = HarnessConfig(
                llm=LLMConfig(provider="anthropic", model="claude-opus-5", api_key="k"),
                env=EnvConfig(name="toy_tabletop", task="pick_place"),
                agent=AgentConfig(mode="tools", max_steps=1, **agent_kw),
            )
            self._check(cfg)


if __name__ == "__main__":
    unittest.main()
