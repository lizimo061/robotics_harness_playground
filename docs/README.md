# Robotics Harness - Documentation

This is the entry point. To keep context small as the repo grows, read this
map first, then open ONLY the file(s) relevant to your task. Every doc is
self-contained and links onward instead of duplicating content, and every
module doc starts with a Purpose / Location / Read-when header so you can
decide whether to load it without reading the body.

## Repo at a glance

harness/ is a layered control stack for LLM-driven robot control in simulation:

    LLM (llm/)  ->  Agent loop (agent/)  ->  Environment (envs/)  ->  Task (tasks/)
                          |                    |
                          +--> Tools (tools/)   +--> Visualization (viz/) + Evaluation (eval/)

## Where to look

| I want to... | Read |
| --- | --- |
| Understand the big picture / data flow | ARCHITECTURE.md |
| Understand core types (Obs, Action, Episode, TaskSpec, Tool) | concepts.md |
| Quick API reference | cheatsheet.md |
| Change or add an LLM provider | modules/llm.md + guides/add-llm-provider.md |
| Add a simulator backend | modules/envs.md + guides/add-environment.md |
| Generate or add a harder task | modules/tasks.md + guides/add-task.md |
| Add a tool for the agent | modules/tools.md + guides/add-tool.md |
| Solve a long-horizon task (plan -> skills -> actions) | guides/long-horizon.md + modules/skills.md |
| Tweak the control loop / agent modes | modules/agent.md |
| Visualize a run (animation + LLM trace) | modules/viz.md |
| Robot metadata (Franka, UR5e) | modules/robot.md |
| Vision / image encoding | modules/perception.md |
| Evaluation metrics + logging | modules/eval.md |
| Config file format | modules/config.md |

## Module map

- agent/      -> modules/agent.md
- envs/       -> modules/envs.md
- genesis (envs/genesis.py) -> modules/genesis.md
- robolab (envs/robolab.py) -> modules/robolab.md
- llm/        -> modules/llm.md
- tasks/      -> modules/tasks.md
- tools/      -> modules/tools.md
- skills/ + planning -> modules/skills.md
- policy + serving (agent/policy.py, serving.py) -> modules/policy.md
- core primitives (types.py, registry.py, utils/, runner.py, cli.py) -> modules/core.md
- robot/      -> modules/robot.md
- perception/ -> modules/perception.md
- viz/        -> modules/viz.md
- eval/       -> modules/eval.md
- config.py, types.py, runner.py, cli.py, registry.py -> modules/config.md + concepts.md

## Conventions

- Public API is listed; internals are summarized, not reproduced.
- 'Extension points' sections say exactly what to subclass or register.
- Prefer reading one module doc + one guide over the whole tree.
