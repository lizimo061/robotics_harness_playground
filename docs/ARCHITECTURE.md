# Architecture

## Layers

```
                         config (YAML/JSON -> dataclasses)
                                   |
        runner.py:  build env + llm + agent, run episodes, eval, visualize
                                   |
   +----------+  +--------------+  +-----------------+  +----------------+
   |  llm/    |  |  agent/      |  |  envs/          |  |  tasks/        |
   |  clients |->|  controller  |->|  Env adapters   |<-|  TaskSpec data |
   +----------+  |  (ReAct)     |  +-----------------+  +----------------+
                 |    |         |        ^
                 |    +--> tools/         | object-aware query API
                 +--> viz/ (trace)  +--> eval/ (metrics/log)
```

## Data flow (one agent step)

1. agent reads the live state: env.get_text_state() (and env.render() for a frame).
2. agent builds a message and calls llm.complete(messages).
3. Depending on mode, the LLM output is an action (json), a tool call (tools),
   or code (code). The agent parses it.
4. The action/tool maps to an env.step(action) -> StepResult.
5. The StepResult (obs, reward, success) is recorded into the Episode and the
   TraceRecorder, then the loop repeats until success / truncation / done.

## Key interfaces (the contracts everything else plugs into)

- LLMClient.complete(messages, **kw) -> LLMResponse            (llm/base.py)
- Env.reset/step/render/get_text_state + object-aware queries (envs/base.py)
- Agent.run(env, *, seed) -> Episode                          (agent/base.py)
- Tool.run(env, **args) -> ToolResult                         (tools/base.py)
- TaskSpec (data) + generate_task(name, seed, difficulty)     (tasks/base.py)

## Extension seams

- New LLM: subclass LLMClient, register in llm/registry.py.
- New env: subclass Env, register in envs/registry.py.
- New task: add a generator in tasks/specs.py (auto-registered via @register_task).
- New tool: subclass Tool, add to tools/registry.get_default_tools().
- New agent mode: add a branch in agent/llm_controller.py.

## Why tasks are data

TaskSpec is a declarative scenario description (objects, goals, obstacles,
success criteria). Envs implement the physics once and consume any TaskSpec.
This separation is what lets tasks be generated procedurally and curriculated
by difficulty, and later reused across toy/tabletop/MuJoCo/Genesis backends.
