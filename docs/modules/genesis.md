# genesis (harness/envs/genesis.py)
> Purpose: Franka Panda in Genesis (genesis-world), driven via IK + kinematic grasp.
> Read when: using/debugging the Genesis backend, or adding a Genesis/3D task.
> Requires: pip install genesis-world (pulls in PyTorch)

## Public API

- GenesisFrankaEnv (alias GenesisEnv)
- __init__(task_spec=None, task='pick_place', seed, difficulty, max_episode_steps,
  backend='cpu', show_viewer=False, control_mode='ee_delta', camera_res=(320,320),
  robot_file='xml/franka_emika_panda/panda.xml')
- action space: ee_delta (3D, dim 3) + gripper, or joint_position (7 dof)
- object-aware queries + get_text_state + render (camera frame) + TaskSpec support

## Scene

plane + Franka Panda (MJCF) + one 0.04 m Box per object + goal markers + obstacle
boxes + a camera. Objects start at table height (z = 0.02 m).

## Control

- ee_delta / ee_pose -> franka.inverse_kinematics(link=hand, pos, quat) then
  control_dofs_position on arm dof 0..6.
- gripper -> control_dofs_position on finger dof 7..8.
- joint_position -> direct arm control (control_mode='joint_position').

Two non-obvious facts the env handles for you:

- control_dofs_position is a **PD target**, not a teleport. A single scene.step()
  barely moves the arm, so step() loops "substeps" physics substeps (default 60)
  per action - a move_to actually reaches its target before returning.
- The Franka wrist's reachable z band is roughly [0.12, 0.45] m (objects on the
  table at z=0.02 are BELOW it). _ik clamps into this band, and grasp uses a
  remembered offset so a carried object keeps its table height.

## Grasp model

Kinematic attachment: closing the fingers over an object within grasp_radius
attaches it to the end-effector (it moves with the EE); opening releases it.
Robust for LLM-driven control. For contact-based grasping see the Genesis SAP
example (examples/sap_coupling/franka_grasp_rigid_cube.py).

## Task integration

Consumes 3D TaskSpecs from harness.tasks (generate_task(name, dims=3)). Success
per kind: place/push/sort (xy goal distance + optional release), stack (top above
base + xy proximity), reach/reach_avoid (ee within target radius, no collision).

## Known issue (libigl version)

genesis-world 0.2.x breaks with libigl >= 2.6.0: igl.signed_distance now returns
4 values but Genesis unpacks 3 (ValueError: too many values to unpack). Fix:

    pip install "libigl==2.5.1"

(Genesis >= 0.2.2 / main branch has a patch; pinning libigl==2.5.1 is the
quickest fix for the PyPI 0.2.1 release.)

## Notes / verification

- API calls follow the official franka_cube example; adjust this file if your
  installed Genesis version differs (it is the single place).
- show_viewer=True opens the native Genesis GUI; the camera feeds the harness
  viz (html/live) the same way the tabletop env renders frames.
- Run: python examples/genesis_demo.py --task stack --show-viewer

## Visualization & video export

- --show-viewer opens the native Genesis GUI (real-time).
- --video PATH records every physics substep and writes an mp4/gif (via imageio);
  capture happens inside step(), and close() auto-saves the video.
- --viz PATH writes the harness HTML replay (animation + LLM trace).
- --scripted runs a deterministic pick-place with no API key - handy for testing
  the video / visualization pipeline offline.

    python examples/genesis_demo.py --task pick_place --scripted --video demo.mp4
    python examples/genesis_demo.py --task pick_place --show-viewer

The generic harness writer is harness/viz/video.write_video(frames, path, fps),
and frames_from_recorder(recorder) pulls frames from a TraceRecorder for any env.

## Related

- modules/envs.md, modules/tasks.md, guides/add-environment.md, guides/add-task.md.
