# Core concepts

## Observation & action

- Obs: state vector + optional image/depth + text + info (harness/types.py).
- Action: kind (ee_delta | ee_pose | joint_position | discrete | noop | stop),
  a numeric value, an optional gripper channel, and a comment.
- ActionSpace / ObservationSpace: static descriptions (bounds, joint names, dims)
  used for prompts and clipping.

## Step / Episode

- StepResult: obs + reward + terminated (success) + truncated (timeout) + info.
- Episode: the recorded trajectory (actions, rewards, infos, observations) with
  a success flag and total reward.

## The control loop

The agent loop is a closed-loop ReAct: observe -> ask LLM -> parse -> step ->
record -> repeat. Modes:

- json  (default): LLM emits one JSON action per step.
- tools:          LLM calls named tools (grasp, move_to, list_objects, ...).
- code:           LLM writes a Python snippet using a skill library.
- plan:           LLM writes a plan first, then acts.

## Task vs Env vs Tool

- TaskSpec (tasks/): a declarative WHAT - which objects go where, what obstacles
  exist, and how success is measured. Generated procedurally with a difficulty.
- Env (envs/): the HOW - physics + success check + object-aware queries.
- Tool (tools/): a named capability (schema + description) the LLM can call;
  maps to env actions or perception queries.

## Object-aware query API (envs/base.py)

list_objects / get_object_pos / list_goals / get_goal_pos / list_obstacles /
get_ee_pos / is_grasped / grasped_object. Safe defaults return empty/None, so
simple envs need not implement them; multi-object envs (TabletopEnv) override.

## Trace / visualization

TraceRecorder (viz/recorder.py) captures per-step observation text, prompt,
response, action, reward, and frame. viz/html.py renders a synced animation +
trace viewer; viz/live.py prints a live trace or shows a matplotlib window.
