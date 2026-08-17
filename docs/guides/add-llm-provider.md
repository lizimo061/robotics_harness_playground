# Add an LLM provider

## Case 1: OpenAI-compatible endpoint (most providers)

1. Open harness/llm/registry.py.
2. Add a tuple to _PROVIDER_DEFAULTS:

       "myprovider": ("https://api.myprovider.com/v1", "my-model", "MYPROVIDER_API_KEY"),

3. Set llm.provider: myprovider in config. Done - OpenAICompatClient handles the rest.

## Case 2: non-OpenAI protocol

1. Subclass LLMClient (harness/llm/base.py); implement complete().
2. Add a branch in get_llm() that constructs your client.

## Checklist

- Raise LLMError for non-retryable failures, TransientLLMError for retryable ones.
- Return LLMResponse(content=..., model=..., usage=..., raw=...).
- Add the api_key_env default so secrets stay out of config files.
