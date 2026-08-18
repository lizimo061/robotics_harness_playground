# envs (harness/envs/)
> Purpose: a single Env interface over every simulator backend.
> Read when: adding a backend, or understanding action/observation mapping.
> Key files: base.py, toy.py, tabletop.py, gymnasium.py, genesis.py, robosuite.py, registry.py

## Public API

- get_env(EnvConfig) -> Env  (the factory)
- Env (ABC): reset(seed) / step(Action) / render() / get_text_state() / is_success / close
- Properties: observation_space, action_space
- Object-aware queries: list_objects, get_object_pos, list_goals, get_goal_pos,
  list_obstacles, get_ee_pos, is_grasped, grasped_object (safe defaults on base)

## Backends

- ToyTabletopEnv (toy_tabletop): single-object 2D pick-and-place, zero deps.
- TabletopEnv (tabletop): multi-object 2D env consuming a TaskSpec; runs all
  harder tasks (pick_place, obstacle, push, stack, sort, reach_avoid).
- KitchenEnv (kitchen): long-horizon env with objects + containers (oven) +
  buttons; consumes a TaskSpec with ordered steps (specs_long.cook_bread).
- GymnasiumEnv (gymnasium:<id>): wraps any gymnasium.Env (MuJoCo via
  gymnasium-robotics). Maps Box/Discrete spaces to Obs/Action.
- GenesisFrankaEnv (genesis:<task>): Franka Panda in genesis-world; consumes 3D
  TaskSpecs (pick_place, stack, sort, push, reach, obstacle variants) via IK +
  kinematic grasp. See modules/genesis.md.
- RobosuiteEnv (robosuite:<task>): standardized MuJoCo manipulation.

## Action kinds

ee_delta (incremental), ee_pose (absolute), joint_position, discrete, noop, stop.
gripper is a separate channel (0 open .. 1 closed) carried on Action.gripper.

## Extension points

- Subclass Env, implement spaces + reset + step, register in registry.get_env.

## Related
- guides/add-environment.md, modules/tasks.md, modules/tools.md (query API users).

## The seed contract

`reset(seed=n)` must produce **instance n**, not merely reseed the RNG. An env
that restores a layout fixed at construction time makes every seed replay the
identical episode, which turns a grid of N seeds into one trial repeated N times
-- pseudo-replication that shrinks every confidence interval while looking like
real sampling. `TabletopEnv` regenerates its layout when the seed changes, and
leaves an explicitly supplied `task_spec` alone (that layout is the caller's).
