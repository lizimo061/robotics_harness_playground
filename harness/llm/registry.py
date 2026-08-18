"""LLM provider factory: turn an LLMConfig into a concrete client."""
from __future__ import annotations

from harness.config import LLMConfig
from harness.llm.base import LLMClient

# provider -> (default base_url, default model, default api_key_env)
_PROVIDER_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "deepseek": ("https://api.deepseek.com", "deepseek-chat", "DEEPSEEK_API_KEY"),
    "kimi": ("https://api.moonshot.cn/v1", "moonshot-v1-8k", "MOONSHOT_API_KEY"),
    "moonshot": ("https://api.moonshot.cn/v1", "moonshot-v1-8k", "MOONSHOT_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-5.5", "OPENAI_API_KEY"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-3-pro",
        "GEMINI_API_KEY",
    ),
    "ollama": ("http://localhost:11434/v1", "llama3", ""),
    "vllm": ("http://localhost:8000/v1", "meta-llama/Llama-3-8B-Instruct", ""),
}


def get_llm(cfg: LLMConfig) -> LLMClient:
    provider = cfg.provider.lower()

    if provider in ("mock", "echo", "dummy", "offline"):
        from harness.llm.mock import MockLLMClient

        return MockLLMClient(model=cfg.model or "mock", extra=cfg.extra)

    # Local `claude` CLI instead of the paid HTTP API. Kept strictly separate
    # from the "claude"/"anthropic" branch below: this one is a no-spend
    # testing backend, not a substitute for the Messages API.
    if provider in ("claude_code", "claude-code", "cli"):
        from harness.llm.claude_code import ClaudeCodeClient

        return ClaudeCodeClient(model=cfg.model, timeout=cfg.timeout, extra=cfg.extra)

    if provider in ("claude", "anthropic"):
        from harness.llm.anthropic import AnthropicClient

        return AnthropicClient(
            model=cfg.model or "claude-opus-5",
            api_key=cfg.api_key,
            api_key_env=cfg.api_key_env or "ANTHROPIC_API_KEY",
            base_url=cfg.base_url or "https://api.anthropic.com",
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            extra=cfg.extra,
        )

    # OpenAI-compatible family (deepseek, kimi, openai, ollama, vllm, custom, ...)
    from harness.llm.openai_compat import OpenAICompatClient

    base_url, model, key_env = _PROVIDER_DEFAULTS.get(
        provider, (cfg.base_url, cfg.model, cfg.api_key_env or "CUSTOM_API_KEY")
    )
    return OpenAICompatClient(
        model=cfg.model or model,
        api_key=cfg.api_key,
        api_key_env=cfg.api_key_env or key_env,
        base_url=cfg.base_url or base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        extra=cfg.extra,
    )
