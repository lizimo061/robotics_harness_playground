# robolab (harness/envs/robolab.py)
> Purpose: drive NVIDIA RoboLab (Isaac Lab) tasks with the harness agent in-process.
> Requires: RoboLab venv on Linux + CUDA (Isaac Sim 5.x + Isaac Lab).

## Public API

- RoboLabEnv(task, num_envs=1, device='cuda:0', use_fabric=False, action_mode='ee_delta', seed)
- maps RoboLab's gym obs/action to Obs/Action/StepResult
- env_cfg.instruction -> the task description fed to the LLM

## Run

    python examples/run_robolab.py --task RubiksCubeTask --headless              # real LLM
    python examples/run_robolab.py --task RubiksCubeTask --scripted --headless  # offline

Hierarchical (recommended for real tasks): the LLM plans, a trained VLA executes.
An LLM emitting raw action vectors cannot do fine manipulation; delegating motion
to a policy via the run_policy tool is what makes RoboLab tasks tractable.

    python examples/serve_robolab.py --port 8000                 # policy server
    python examples/run_robolab.py --task BananaInBowlTask --headless \
        --policy-url http://localhost:8000 --policy-steps 60

See modules/policies.md and configs/robolab_policy_tool.yaml.

Run inside the RoboLab venv. The Isaac Sim AppLauncher must be created before
importing robolab/isaaclab (the example does this). The generic
python -m harness.cli path does NOT set up AppLauncher, so it is not usable for
RoboLab.

## How it works

- __init__ lazily imports robolab, calls auto_register_droid_envs(), resolves the
  task via get_envs(task=[...]), then create_env(task, use_fabric=False).
- obs: extracts a low-dim state (observation/proprio/policy keys) and image
  (image/rgb keys); text is env_cfg.instruction.
- action: pads/truncates Action.value to action_space; treats the last dim as the
  gripper when Action.gripper is set.
- success: read from info (success / is_success / goal_achieved).

## TODO - verify on the Linux box (write these, then delete this section)

1. create_env(use_fabric=False) returns a gymnasium-style env; confirm reset()/
  step() return shapes and the obs dict keys.
2. Confirm the action dim and gripper convention for your target robot.
3. Surface RoboLab's composable success predicates in check_subgoal (they are
  the natural mapping for skills-mode subgoal verification).
4. decide whether use_fabric=True (parallel) is needed and adapt _unwrap_*.

## Policy server (leaderboard path)

For the benchmark/leaderboard, run the harness as a standalone policy server:

    python examples/serve_robolab.py --port 8000          # the harness side
    # + examples/robolab_inference_client.py in your RoboLab policy repo

See modules/policy.md for the protocol and client hooks.

## Related
- modules/envs.md, modules/policy.md, modules/policies.md (policy-as-tool),
  guides/add-environment.md, examples/run_robolab.py.

## What the agent can see

`get_text_state()` reports the instruction, every scene object's position in the
robot's frame, and the arm's own pose; `list_objects()` / `get_object_pos()` back
the query tools. Object poses come from Isaac's ground-truth scene state, so this
is a **privileged-state** evaluation: it measures planning and grounding, not
perception. Say so when reporting a number from it -- a model scoring well here
has not been shown to perceive anything. For perception, use a VLM through the
vision path, or delegate motion to a VLA via `run_policy` (policy-as-tool).

Without these, a text model knows the instruction and its own arm pose and
nothing else. Measured consequence on a live DeepSeek run: it called `done` after
a single environment step, because flailing and stopping are indistinguishable
when there is no feedback.

## Absolute vs relative actions

RoboLab's controller consumes **relative** end-effector deltas. `move_to` emits
an absolute target, so the adapter converts it against the current pose and clips
to the action space's own per-step limit. Passing the target through unchanged
made every `move_to` a command to jump most of a metre; the controller saturates,
the arm lurches, and the resulting failures look like the model's when they are
the harness's.

## Task names

Task names must exist in the installed RoboLab registry (120 in v0.3.1). There is
no `PickCubeTask`, which this repo's config and examples used to name --
`RoboLabEnv` raises with the full list, so run a name through it once rather than
guessing.

## Frame conventions (the ones that fail silently)

RoboLab's differential IK **tracks `base_link`**, while poses are expressed in an
end-effector frame that shares that origin but is rotated by
`robolab.robots.droid.EEF_OFFSET_ROT = (0.5, -0.5, 0.5, -0.5)`
(`EEF_OFFSET_POS` is zero). A commanded orientation must therefore be un-offset
before it is sent:

    action_quat = target_eef_quat (x) R_offset^-1

RoboLab's own `examples/run_abs_ik_demo.py` does exactly this, and
`DroidIKActionCfg`'s docstring explains why the rotation is not baked into
`body_offset`. Skipping the conversion does not raise: it points the gripper
somewhere else. Symptom to recognise -- a commanded top-down grasp closes on empty
space at *every* approach depth, because the fingers were never above the object.

The measured orientation is converted the same way on the way in, so a pose read
back can be commanded again unchanged. Both directions must agree or the agent
cannot aim.

## Reading state after a reset

IsaacLab fills `root_pos_w` / `body_pos_w` during a sim step, so reading them
immediately after `reset()` can return the previous episode's values. Two symptoms
worth recognising: every one of the robot's 18 body poses reads back identical, and
an object left displaced by one episode is still reported there by the next
episode's reset -- which looks exactly like an episode-state leak. `reset()`
refreshes the scene's cached poses to avoid handing the agent a stale first
observation.

`reset(seed=)` also forwards the seed to IsaacLab. It used to be dropped, so every
trial reset from IsaacLab's advancing RNG and the per-trial seed controlled nothing.
