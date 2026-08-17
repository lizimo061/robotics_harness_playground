"""Demo the harder-task + tool-use system.

    python examples/harder_tasks.py                  # offline demo (mock LLM, fixed stack task)
    python examples/harder_tasks.py --task sort      # list generators, then run sort with a real LLM

With DEEPSEEK_API_KEY set, the --task path drives a real LLM through the tool
interface (list_objects, move_to, grasp, ...) to solve a procedurally generated
harder task.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.agent import LLMController
from harness.config import LLMConfig
from harness.envs.tabletop import TabletopEnv
from harness.llm import get_llm
from harness.tasks import available_tasks, generate_task
from harness.tasks.base import TaskSpec


def run_fixed_stack() -> None:
    spec = TaskSpec(
        kind="stack",
        description="Stack the top block on the base block.",
        objects=[
            {"name": "base", "pos": [0.4, 0.4], "role": "base"},
            {"name": "top", "pos": [0.4, 0.7], "role": "top"},
        ],
        goals={},
        ee_start=[0.1, 0.1],
        params={"stack_radius": 0.1, "grasp_radius": 0.15},
    )
    env = TabletopEnv(task_spec=spec)
    script = [
        {"tool": "move_to", "args": {"x": 0.4, "y": 0.7}},
        {"tool": "grasp", "args": {}},
        {"tool": "move_to", "args": {"x": 0.4, "y": 0.4}},
        {"tool": "release", "args": {}},
        {"tool": "done", "args": {}},
    ]
    llm = get_llm(LLMConfig(provider="mock", extra={"script": script}))
    ep = LLMController(llm, mode="tools", max_steps=20).run(env)
    print(f"[offline] stack: success={ep.success}  steps={ep.steps}")


def run_generated(task: str, provider: str) -> None:
    spec = generate_task(task, seed=1, difficulty=0.7)
    print(f"[{task}] {spec.description}")
    env = TabletopEnv(task_spec=spec)
    llm = get_llm(LLMConfig(provider=provider))
    ep = LLMController(llm, mode="tools", max_steps=60, task_description=spec.description).run(env)
    print(f"[{task}] success={ep.success}  steps={ep.steps}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    print("available tasks:", available_tasks())
    print()
    run_fixed_stack()
    print()
    if args:
        task = args[0]
        provider = "deepseek"
        run_generated(task, provider)


if __name__ == "__main__":
    main()
