"""Tool abstraction: a named, schema-described capability the agent can call.

A tool is the unit of "tool use". The agent is given a list of tool schemas and
calls them by name; the harness executes them against the environment and feeds
the resulting feedback back into the next LLM turn. This is higher-level than
raw action emission, which is what makes harder tasks tractable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from harness.types import Action


@dataclass
class ToolResult:
    """The outcome of running a tool.

    feedback: text returned to the agent (perception result, confirmation, ...)
    action:   optional low-level action to apply to the environment
    done:     True when the tool declares the episode finished
    """

    feedback: str = ""
    action: Optional[Action] = None
    done: bool = False


class Tool(ABC):
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON Schema for the arguments

    @abstractmethod
    def run(self, env, **args: Any) -> ToolResult:
        """Execute the tool against the environment and return feedback."""
        raise NotImplementedError

    def to_openai_function(self) -> dict:
        """Return an OpenAI function-calling schema (also usable for prompt-based tool use)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }

    def signature(self) -> str:
        props = (self.parameters or {}).get("properties", {})
        args = ", ".join(props.keys())
        return f"{self.name}({args})"

    def __repr__(self) -> str:
        return f"Tool({self.name})"
