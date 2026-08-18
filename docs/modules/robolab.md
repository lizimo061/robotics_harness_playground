# robolab (harness/envs/robolab.py)
> Purpose: drive NVIDIA RoboLab (Isaac Lab) tasks with the harness agent in-process.
> Requires: RoboLab venv on Linux + CUDA (Isaac Sim 5.x + Isaac Lab).

## Public API

- RoboLabEnv(task, num_envs=1, device='cuda:0', use_fabric=False, action_mode='ee_delta', seed)
- maps RoboLab's gym obs/action to Obs/Action/StepResult
- env_cfg.instruction -> the task description fed to the LLM

## Run

    python examples/run_robolab.py --task PickCubeTask --headless              # real LLM
    python examples/run_robolab.py --task PickCubeTask --scripted --headless  # offline

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
