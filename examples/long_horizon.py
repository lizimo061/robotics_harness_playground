"""Solve a long-horizon task: put the bread into the oven, then press the button.

    python examples/long_horizon.py          # offline (gold plan via mock LLM)
    python examples/long_horizon.py --real   # real DeepSeek plans, then acts

Demonstrates the two-level hierarchy: the planner (or the task's gold plan)
decomposes the task into skills; each skill runs closed-loop with subgoal
verification.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.agent import LLMController
from harness.config import LLMConfig
from harness.envs.kitchen import KitchenEnv
from harness.llm import get_llm
from harness.tasks import generate_task


def main() -> None:
    real = "--real" in sys.argv
    task_spec = generate_task("cook_bread", seed=1)
    print("Task:", task_spec.description)
    print("gold plan:", [(s["skill"], s["args"]) for s in task_spec.steps])
    print()

    env = KitchenEnv(task_spec=task_spec)
    if real:
        llm = get_llm(LLMConfig(provider="deepseek"))
    else:
        # mock returns non-JSON -> the controller falls back to the gold plan
        llm = get_llm(LLMConfig(provider="mock", extra={"fallback": "not json"}))

    ctrl = LLMController(llm, mode="skills", max_steps=200, task_description=task_spec.description)
    ep = ctrl.run(env)
    print(f"success={ep.success}  steps={ep.steps}")
    print("plan used:", [s["skill"] for s in ep.metadata.get("plan", [])])
    for k, v in ep.metadata.items():
        if k.startswith("skill_"):
            print("  ", k, v)


if __name__ == "__main__":
    main()
