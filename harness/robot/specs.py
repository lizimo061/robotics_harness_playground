"""Robot specifications (metadata used for prompts and action spaces)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotSpec:
    name: str
    dof: int  # actuated arm joints
    gripper_dof: int  # extra gripper joints
    joint_names: tuple[str, ...]
    joint_low: tuple[float, ...]
    joint_high: tuple[float, ...]
    ee_link: str
    home_qpos: tuple[float, ...]

    @property
    def total_dof(self) -> int:
        return self.dof + self.gripper_dof


FRANKA_PANDA = RobotSpec(
    name="Franka Panda",
    dof=7,
    gripper_dof=2,
    joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
    joint_low=(-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973),
    joint_high=(2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973),
    ee_link="hand",
    home_qpos=(-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04),
)

UR5E = RobotSpec(
    name="UR5e",
    dof=6,
    gripper_dof=0,
    joint_names=("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"),
    joint_low=(-6.2832, -6.2832, -6.2832, -6.2832, -6.2832, -6.2832),
    joint_high=(6.2832, 6.2832, 6.2832, 6.2832, 6.2832, 6.2832),
    ee_link="ee_link",
    home_qpos=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)

ROBOT_SPECS: dict[str, RobotSpec] = {s.name: s for s in (FRANKA_PANDA, UR5E)}
ROBOT_SPECS["franka_panda"] = FRANKA_PANDA
ROBOT_SPECS["panda"] = FRANKA_PANDA
ROBOT_SPECS["franka"] = FRANKA_PANDA
ROBOT_SPECS["ur5e"] = UR5E


def get_robot_spec(name: str) -> RobotSpec:
    key = name.lower().replace("-", "_")
    if key not in ROBOT_SPECS:
        raise KeyError(f"Unknown robot '{name}'. Available: {sorted(ROBOT_SPECS)}")
    return ROBOT_SPECS[key]
