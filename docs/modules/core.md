# core (harness/types.py, registry.py, utils/, runner.py, cli.py)
> Purpose: the dependency-free primitives and the top-level entry points.
> Read when: you need the shared types, the generic registry, or the run CLI.

## types.py

Obs, Action, StepResult, Episode, ActionSpace, ObservationSpace - the common
currency passed between llm / agent / envs / eval. Obs carries state + image +
text; Action carries kind + value + gripper; StepResult carries reward +
terminated/truncated + info (with a success flag).

## registry.py

Registry: a tiny name -> factory dict used by llm/envs/tools/skills/tasks to
register and look up providers by string key.

## utils/

- get_logger(name) -> a shared logging.Logger
- set_seed(seed) -> seeds random + numpy

## runner.py

run_eval(cfg) -> summary dict. Builds env + llm + agent + eval + viz from a
HarnessConfig and runs the configured episodes. This is what the CLI and most
examples call.

## cli.py

    python -m harness.cli configs/toy_pick_place.yaml [--episodes N]

## Related
- modules/config.md, concepts.md, modules/agent.md.
