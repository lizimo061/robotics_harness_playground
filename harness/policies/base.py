"""Low-level policy backends the agent can delegate motor control to.

A Policy maps an observation to a raw action vector. It is the *executor* half
of the hierarchical setup that policy benchmarks (RoboLab, RoboDojo) are built
around::

    LLM (planner, tools mode) --run_policy("pick up the banana")--> Policy
                              <--feedback: steps, success, state--

Why this exists: an LLM emitting JSON deltas step-by-step cannot do fine
continuous manipulation, but it is good at decomposing a task and checking
progress. A trained VLA (pi0.5, GR00T, ...) is the reverse. Exposing the policy
as a tool lets each do the half it is good at.

The protocol deliberately mirrors harness.agent.policy.PolicyAgent, so an LLM
can itself stand in as the low-level policy (handy for smoke tests without a
GPU policy server), while a real VLA plugs in over HTTP via RemotePolicy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from harness.types import ActionSpace


class Policy(ABC):
    """Maps an observation to one raw action vector."""

    name: str = "policy"

    def begin(self, instruction: str, action_space: Optional[ActionSpace] = None) -> None:
        """Start a new sub-task with a fresh instruction. Default: no-op.

        Stateful policies (conversation history, action chunking) reset here.
        """
        return None

    @abstractmethod
    def act(self, observation_text: str, image: Optional[np.ndarray] = None) -> np.ndarray:
        """Return one action vector for the given observation."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear per-episode state. Default: no-op."""
        return None

    def close(self) -> None:
        """Release transport/compute resources. Default: no-op."""
        return None
