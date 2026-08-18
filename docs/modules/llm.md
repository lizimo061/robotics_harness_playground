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
- claude_code/claude-code/cli -> local `claude` CLI subprocess, no API spend (see below)
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

## claude_code: the local CLI as a no-spend backend

`provider: claude_code` (aliases `claude-code`, `cli`) shells out to the locally
installed `claude` CLI instead of calling a paid HTTP endpoint, so a full sweep
can be exercised end-to-end without API spend.

Invocation (flags verified against CLI 2.1.234):

    claude --print --output-format json --tools "" \
           --disable-slash-commands --strict-mcp-config --no-session-persistence \
           [--model M] [--system-prompt P]   < flattened-prompt-on-stdin

`--tools ""` disables every built-in tool, so this is a plain text completion
backend with no file access. The message list is flattened into one prompt with
`System:` / `User:` / `Assistant:` labels. `content` comes from the envelope's
`result`, `usage` from its token/cost fields (left empty if absent, never
invented), `model` from `modelUsage` (matched against the top-level usage
totals, since the CLI also routes side work to small helper models).

Config knobs live in `extra`: `cli_path`, `on_image` (`error` | `warn`),
`retries` (default 1 - a CLI retry costs a whole process spawn), `cwd`
(defaults to the system temp dir so the project's CLAUDE.md does not leak into
prompts), `system_prompt`, `extra_args`. `temperature` / `max_tokens` have no
CLI equivalent and are ignored rather than faked.

Errors: timeouts and 429/5xx `api_error_status` raise `TransientLLMError`; a
missing binary, a 4xx, or any other non-zero exit raises `LLMError`. Truncated
stderr is always in the message.

Limitations, deliberately: it is **text-only** - an image-bearing message
raises rather than silently dropping the frame (set `extra.on_image: warn` to
opt into a loud text-only degradation). It is not the Messages API: latency is
dominated by process spawn plus the CLI's own agent harness, and the output
distribution is Claude Code's agent prompt, not the raw model's. Do not publish
results from this backend as model-comparison numbers - use `provider: claude`
for those.

## Related
- guides/add-llm-provider.md, modules/agent.md (how the client is called).
