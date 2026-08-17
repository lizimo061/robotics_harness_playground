# llm (harness/llm/)
> Purpose: talk to LLMs through one interface (DeepSeek, Kimi, OpenAI, Claude, mock).
> Read when: adding a provider, debugging requests, or changing prompt payloads.
> Key files: base.py, openai_compat.py, anthropic.py, mock.py, retry.py, registry.py

## Public API

- get_llm(LLMConfig) -> LLMClient  (the factory - the only thing most code calls)
- ChatMessage (system/user/assistant/user_vision) + to_openai/to_anthropic
- LLMResponse (content, model, usage, raw, finish_reason)
- LLMError / TransientLLMError, with_retries(fn, retries=3)
- Clients: OpenAICompatClient, AnthropicClient, MockLLMClient

## Providers

- deepseek -> https://api.deepseek.com, deepseek-chat (DEEPSEEK_API_KEY)
- kimi/moonshot -> https://api.moonshot.cn/v1 (MOONSHOT_API_KEY)
- openai -> https://api.openai.com/v1 (OPENAI_API_KEY)
- claude/anthropic -> Anthropic Messages API (ANTHROPIC_API_KEY)
- ollama/vllm/custom -> any OpenAI-compatible /chat/completions endpoint
- mock -> offline scripted replies (extra.responses / extra.script / extra.fallback)

## How it works

OpenAICompatClient POSTs {base_url}/chat/completions with an Authorization Bearer
header. Transient statuses (429, 5xx) and network errors raise TransientLLMError
and are retried with backoff. Reasoning models that return empty content fall
back to reasoning_content.

## Extension points

- OpenAI-compatible provider: add a tuple to _PROVIDER_DEFAULTS in registry.py.
- Non-OpenAI provider: subclass LLMClient and add a branch in get_llm.

## Related
- guides/add-llm-provider.md, modules/agent.md (how the client is called).
