"""Minimal self-contained example (no config file, no API key).

Shows the whole pipeline in ~20 lines: build an environment, wire a mock LLM
through the controller, run one episode, and print the result.

    python examples/quickstart.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.agent import LLMController
from harness.config import EnvConfig, LLMConfig
from harness.envs import get_env
from harness.llm import get_llm

# A scripted "LLM" that solves the toy pick-and-place task offline.
script = [
    {"action": "move", "delta": [0.25, 0.25]},
    {"action": "move", "delta": [0.15, 0.15]},
    {"action": "close"},
    {"action": "move", "delta": [0.25, 0.25]},
    {"action": "move", "delta": [0.10, 0.10]},
    {"action": "stop"},
]

llm = get_llm(LLMConfig(provider="mock", extra={"script": script}))
env = get_env(EnvConfig(name="toy_tabletop", task="pick_and_place"))
controller = LLMController(llm, mode="json", max_steps=20)

episode = controller.run(env)
print(f"success={episode.success}  steps={episode.steps}  reward={episode.total_reward:.3f}")
print("actions:", [a.kind for a in episode.actions])
