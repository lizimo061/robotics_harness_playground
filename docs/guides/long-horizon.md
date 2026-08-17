# Long-horizon tasks: understanding -> action

This guide explains how the harness turns a natural-language, multi-step task
(such as 'put the bread into the oven, then press the button to heat it') into
executed actions.

## The core idea: two levels, with verification

Long-horizon tasks are hard because they need (a) decomposition, (b) sequencing
with dependencies, and (c) recovery. The harness splits control into two levels:

1. Understand (plan)  - the LLM reads the task + scene and returns an ordered
   plan of skills. Pure reasoning, no physics.
2. Act (execute)      - each skill is a closed-loop controller that emits
   primitive actions until its subgoal is verified.

Every skill carries a subgoal CHECK (satisfied(env)); the executor runs the
skill until that check passes, retries on failure, and the env verifies each
subgoal independently (check_subgoal). The system does not trust the plan
blindly - which is what makes it robust over long horizons.

## From words to a plan

The planner gets the task description, the scene as text, and the skill catalog:

    - open(container)
    - put_in(object, container)
    - press(button)
    - pick(object)
    - place(object, target)

and returns JSON:

    {"plan": [
      {"skill": "open",   "args": {"container": "oven"}},
      {"skill": "put_in", "args": {"object": "bread", "container": "oven"}},
      {"skill": "press",  "args": {"button": "button"}}]}

## From a plan to actions

The executor runs each skill with run_skill(env, skill): loop plan_action(env)
-> env.step(action) until satisfied(env). Each skill is a tiny state machine:

- open(oven):    move to door -> close gripper -> done when door open.
- put_in(b, ov): (open door if needed) -> move to bread -> grasp -> move to
  interior -> release -> done when bread inside and released.
- press(button): move to button -> close gripper -> done when pressed.

## Subgoal verification

The env implements check_subgoal(name): oven_open / bread_in_oven /
button_pressed. Success = every subgoal in the task passes. The skills'
satisfied(env) reuses the same checks, so plan, execution, and evaluation all
agree on what 'done' means.

## Replanning / recovery

If a skill fails (budget exhausted, collision), the executor retries once. For
stronger recovery, ask the planner to revise the remaining plan given the
failure (the extension point). The two-level design makes replanning cheap:
only the plan changes, skills stay the same.

## Why this scales

- Skills are reusable, verifiable, and composable; new tasks = new plans, not
  new controllers.
- The planner only needs the catalog + scene text, so context stays small even
  as the skill set grows.
- A 3D Franka reuses the same Skill interface (IK + approach-height); the env
  just exposes the same query API (get_object_pos, get_container_interior,
  get_button_pos, check_subgoal).

## Files

- harness/skills/ (base, builtin, registry, executor, planning)
- harness/envs/kitchen.py (KitchenEnv)
- harness/tasks/specs_long.py (cook_bread)
- agent mode: agent.mode = "skills"
