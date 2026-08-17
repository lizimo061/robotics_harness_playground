# Add an environment backend

1. Create harness/envs/<name>.py with a class subclassing harness.envs.base.Env.
2. Implement observation_space, action_space, reset(seed), step(action).
3. Optionally implement render(), get_text_state(), and the object-aware query
   API (list_objects, get_object_pos, get_goal_pos, is_grasped, ...) if the
   backend has objects (needed for tools and harder tasks).
4. Register it in harness/envs/registry.py get_env().

## Contract

- step() returns StepResult(obs, reward, terminated, truncated, info).
- info['success'] is the canonical success signal used by the agent and eval.
- Keep heavy imports (genesis, gymnasium, robosuite, torch) inside __init__ so
  the module imports lazily and does not break machines without them.

## Example

See harness/envs/tabletop.py (multi-object, task-driven) or toy.py (minimal).
