# config (harness/config.py)
> Purpose: typed configuration - YAML/JSON loaded into dataclasses.
> Read when: you need to add, change, or read a config field.
> Key files: harness/config.py

## Public API

- load_config(path) -> HarnessConfig
- HarnessConfig (fields: llm, env, agent, eval, viz, seed) with from_dict / from_file / to_dict
- LLMConfig (provider, model, api_key, api_key_env, base_url, temperature, max_tokens, timeout, extra)
- EnvConfig (name, task, seed, max_episode_steps, render, params)
- AgentConfig (name, mode, max_steps, use_vision, system_prompt, temperature, extra)
- EvalConfig (episodes, log_dir, save_trajectories, verbose)
- VizConfig (enabled, backend, output, fps, capture_frames, title)

## How it works

_instantiate() walks dataclass fields and recurses into nested dataclasses using
typing.get_type_hints (so `from __future__ import annotations` is safe). Missing
keys keep their declared defaults. YAML is preferred; JSON is a fallback.

## Extension points

- Add a field to the relevant dataclass; it flows through from_dict automatically.
- Add a whole new subsystem: add a dataclass + a field on HarnessConfig.
- env.params / agent.extra / llm.extra are free-form dicts passed to the
  respective factory - use them for per-env or per-provider knobs.

## Related
- modules/envs.md (how env.params is consumed), modules/llm.md, modules/viz.md.
