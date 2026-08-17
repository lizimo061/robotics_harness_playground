from harness.tasks.base import (
    TaskSpec,
    generate_task,
    available_tasks,
    register_task,
    register_task_3d,
    generate_curriculum,
)
from harness.tasks import specs  # noqa: F401  (2D generators)
from harness.tasks import specs3d  # noqa: F401  (3D Franka/Genesis generators)
from harness.tasks import specs_long  # noqa: F401  (long-horizon generators)

__all__ = [
    "TaskSpec",
    "generate_task",
    "available_tasks",
    "register_task",
    "register_task_3d",
    "generate_curriculum",
]
