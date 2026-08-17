"""PolicyAgent: a per-step, server-friendly agent.

LLMController.run() is a closed-loop controller that owns the environment and
drives a whole episode. Policy-server architectures (RoboLab, RoboDojo /
XPolicyLab) invert that: the simulator owns the env and asks the policy for ONE
action per step. PolicyAgent exposes that interface.

    agent = PolicyAgent(llm, action_space=ActionSpace(kind="joint_position", dim=8))
    agent.begin("Pick up the cube")            # build the system prompt
    vec = agent.act("ee=(0.1, ...) joint=...") # one action vector per call

It maintains the conversation across act() calls, so long-horizon reasoning
survives the server round-trips. Modes that need in-process env access (tools,
skills, code) belong to LLMController; PolicyAgent is the direct action-emission
path used behind a policy server.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from harness.agent.action_parser import parse_action
from harness.agent.prompts import build_system_prompt
from harness.llm.base import ChatMessage, LLMClient
from harness.types import ActionSpace


class PolicyAgent:
    def __init__(
        self,
        llm: LLMClient,
        *,
        action_space: Optional[ActionSpace] = None,
        temperature: Optional[float] = None,
        action_dim: int = 8,
        gripper_last: bool = True,
    ) -> None:
        self._llm = llm
        self._action_space = action_space
        self._temperature = temperature
        self._action_dim = action_dim
        self._gripper_last = gripper_last
        self._messages: list[ChatMessage] = []
        self._instruction = ""

    def begin(self, instruction: str, action_space: Optional[ActionSpace] = None) -> None:
        """Start a fresh session: set the instruction and reset the conversation."""
        self._instruction = instruction
        if action_space is not None:
            self._action_space = action_space
        if self._action_space is not None:
            self._action_dim = self._action_space.dim
        space = self._action_space or self._default_space()
        system = build_system_prompt(task=instruction, action_space=space, mode="json")
        self._messages = [ChatMessage.system(system)]

    def act(self, observation_text: str, image: Optional[np.ndarray] = None) -> np.ndarray:
        """Emit one action vector for the given observation.

        Returns a float32 array of self._action_dim (gripper in the last slot if
        gripper_last). A stop/noop reply maps to a zero vector (env handles
        termination via its own success signal).
        """
        if image is not None:
            from harness.perception.vision import encode_image

            self._messages.append(ChatMessage.user_vision(observation_text, encode_image(image)))
        else:
            self._messages.append(ChatMessage.user(observation_text))

        resp = self._llm.complete(self._messages, temperature=self._temperature)
        self._messages.append(ChatMessage.assistant(resp.content))

        action = parse_action(resp.content, self._action_space)
        return self._to_vector(action)

    # -- helpers --------------------------------------------------------- #
    def _to_vector(self, action) -> np.ndarray:
        dim = self._action_dim
        v = np.zeros(dim, dtype=np.float32)
        if action.value is not None:
            arr = np.asarray(action.value, dtype=np.float32).ravel()
            n = min(dim, arr.size)
            v[:n] = arr[:n]
        if action.gripper is not None:
            idx = dim - 1 if self._gripper_last else 0
            v[idx] = float(action.gripper)
        return v

    def _default_space(self) -> ActionSpace:
        return ActionSpace(
            kind="joint_position",
            dim=self._action_dim,
            low=-np.pi * np.ones(self._action_dim, dtype=np.float32),
            high=np.pi * np.ones(self._action_dim, dtype=np.float32),
            description=f"raw action vector of dim {self._action_dim}; last dim is the gripper (0 open, 1 closed).",
        )
