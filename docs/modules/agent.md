# agent (harness/agent/)
> Purpose: the control loop that connects an LLM to an environment.
> Read when: changing agent behavior, modes, prompts, or parsing.
> Key files: llm_controller.py, action_parser.py, prompts.py, skills.py, code_executor.py

## Public API

- LLMController(llm, *, mode, max_steps, use_vision, system_prompt, temperature,
  task_description, recorder, on_step, tools)
- LLMController.run(env, *, seed) -> Episode
- parse_action(text, action_space) -> Action; extract_json(text) -> dict
- execute_code(code, namespace, timeout) -> dict (restricted sandbox)
- build_system_prompt / build_tools_system_prompt / build_observation_message

## Modes

- json  (default): LLM emits one JSON action; closed-loop ReAct.
- tools:          LLM calls named tools; harness executes and feeds back results.
- code:           LLM writes Python using a skill library (SkillContext);
                  executed with a builtins allowlist + AST import blocklist + timeout.
- plan:           LLM writes a plan first, then json loop with the plan in context.

## Recording / visualization

Pass recorder (TraceRecorder) and/or on_step (callback) to capture per-step
observation, prompt, response, action, reward, and frame. The controller prefers
the live env.get_text_state() for observations.

## Extension points

- Add a mode: branch in LLMController.run; add a prompt builder in prompts.py.
- Add an action shorthand: extend parse_action in action_parser.py.

## Related
- modules/tools.md, modules/viz.md, modules/tasks.md, guides/add-tool.md.
