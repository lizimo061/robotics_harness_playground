# tasks (harness/tasks/)
> Purpose: declarative task specifications + procedural generators for harder tasks.
> Read when: generating harder tasks, adding a task, or tuning difficulty.
> Key files: base.py (TaskSpec, registry), specs.py (generators)

## Public API

- generate_task(name, seed, difficulty, dims=2, **kw) -> TaskSpec
- available_tasks(dims=2) -> list[str]
- generate_curriculum(kind, seeds, difficulties, dims=2) -> list[TaskSpec]
- register_task(kind) / register_task_3d(kind) decorators; TaskSpec has a dims field

## TaskSpec fields

kind, description, difficulty, objects (name/pos/target/role), goals (name->pos),
obstacles (name/pos/radius), ee_start, ee_target, params (radii etc.), seed.

## Built-in kinds

- pick_place: one object -> one goal.
- pick_place_obstacle: same, with an obstacle to route around.
- push: object -> goal (grasping optional).
- stack: place the top block onto the base block.
- sort: 2-3 objects -> their matching bins (scales with difficulty).
- reach_avoid: move the end-effector to a target without hitting an obstacle.

## 3D tasks (Franka / Genesis)

harness/tasks/specs3d.py registers the same kinds with 3D positions via
register_task_3d; generate_task(name, dims=3) returns them. It adds reach /
reach_avoid (an end-effector target) and a 3D stack (height check). The
GenesisFrankaEnv consumes these; a 3D Franka backend uses the same TaskSpec
schema with no change.

## Long-horizon tasks

harness/tasks/specs_long.py adds multi-step tasks with containers, buttons, and
ordered subgoals (TaskSpec.steps). cook_bread = "put the bread into the oven,
then press the button". The skills agent mode (modules/skills.md) plans over
these and the env verifies each subgoal via check_subgoal.

## Difficulty

0.0 (easy) .. 1.0 (hard) scales distances, obstacle radius, and object count.
Generators are seeded, so the same (name, seed, difficulty) is reproducible.

## Extension points

- Add a generator in specs.py decorated with @register_task. For 3D Franka use
  3D positions; the TaskSpec schema is unchanged.

## Related
- guides/add-task.md, modules/envs.md (TabletopEnv consumes TaskSpec).
