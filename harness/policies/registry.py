"""Policy factory: turn a config spec into a Policy.

Lets the executor policy be selected from YAML, e.g.::

    agent:
      mode: tools
      extra:
        policy:
          type: remote            # remote | llm | scripted
          base_url: http://localhost:8000
          action_dim: 8
"""
from __future__ import annotations

from typing import Any, Optional

from harness.policies.base import Policy


def get_policy(spec: Any, *, llm: Optional[Any] = None) -> Policy:
    """Build a Policy from a spec.

    ``spec`` is either an already-built Policy (returned as-is) or a dict with a
    ``type`` key. ``llm`` is required for ``type: llm``, which wraps the
    controller's own LLM as the low-level policy (a smoke-test path -- a real
    VLA should go through ``type: remote``).
    """
    if isinstance(spec, Policy):
        return spec
    if hasattr(spec, "act") and not isinstance(spec, dict):
        return spec  # duck-typed policy (e.g. PolicyAgent)
    if not isinstance(spec, dict):
        raise TypeError(f"policy spec must be a Policy or a dict, got {type(spec).__name__}")

    cfg = dict(spec)
    kind = str(cfg.pop("type", "remote")).strip().lower()

    if kind in ("remote", "server", "http"):
        from harness.policies.remote import RemotePolicy

        return RemotePolicy(**cfg)

    if kind in ("llm", "policy_agent"):
        if llm is None:
            raise ValueError("policy type 'llm' needs an LLM client; none was provided")
        from harness.agent.policy import PolicyAgent

        return PolicyAgent(llm, **cfg)

    if kind in ("scripted", "mock"):
        from harness.policies.scripted import ScriptedPolicy

        return ScriptedPolicy(**cfg)

    raise KeyError(f"Unknown policy type '{kind}'. Use remote, llm, or scripted.")
