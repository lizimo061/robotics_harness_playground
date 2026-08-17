# robot (harness/robot/)
> Purpose: robot metadata (joints/limits/home) and Action constructors.
> Read when: adding a robot arm or building prompts around a specific arm.
> Key files: specs.py, actions.py

## Public API

- get_robot_spec(name) -> RobotSpec; ROBOT_SPECS dict
- RobotSpec (name, dof, gripper_dof, joint_names, joint_low, joint_high, ee_link, home_qpos)
- Predefined: FRANKA_PANDA (7 dof + 2 finger), UR5E (6 dof)
- actions: move_delta, move_to, set_gripper, set_joint_positions, noop, stop

## Extension points

- Add a RobotSpec and register it in ROBOT_SPECS.

## Related
- modules/envs.md (envs use action-space metadata), concepts.md.
