"""Two contexts, read at two different rates.

A single growing transcript is the wrong shape for a physical loop. VoLoAgent's
observation is that the world does not pause for reasoning, so the prompt that
*watches* must be cheap enough to read at the motion timescale, while the prompt
that *decides* can be expensive because it is read rarely:

- :class:`MonitorContext` -- system line, active subgoal, the newest observation and
  the last few decisions. Bounded by construction, so its size does not grow with
  episode length.
- :class:`DeliberationContext` -- the full transcript, consulted at planning points
  (start, subgoal boundaries, recovery).

Both are *views* over one shared :class:`Transcript`. That matters: if the monitor
kept its own message list, the two would drift and a decision taken on the fast
clock would be invisible to the slow one. Appending happens in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from harness.llm.base import ChatMessage


@dataclass
class Turn:
    """One exchange, kept structured so a view can be rebuilt from parts."""

    observation: str = ""
    reply: str = ""
    #: short human-readable note of what the reply did, e.g. "move_to(0.43,-0.1)"
    decision: str = ""
    #: feedback returned to the agent after the decision ran
    feedback: str = ""
    #: True when this turn was a planning point rather than a monitor step
    deliberated: bool = False


@dataclass
class Transcript:
    """The single source of truth for an episode's conversation."""

    system: str = ""
    turns: list = field(default_factory=list)
    #: subgoals the agent committed to, in order, and where it is now
    subgoals: list = field(default_factory=list)
    subgoal_index: int = 0

    # -- mutation (one place, so views cannot diverge) --------------------- #
    def begin_turn(self, observation: str, *, deliberated: bool = False) -> Turn:
        turn = Turn(observation=observation, deliberated=deliberated)
        self.turns.append(turn)
        return turn

    def record_reply(self, reply: str, decision: str = "") -> None:
        if self.turns:
            self.turns[-1].reply = reply
            self.turns[-1].decision = decision

    def record_feedback(self, feedback: str) -> None:
        if self.turns:
            self.turns[-1].feedback = feedback

    def set_subgoals(self, subgoals) -> None:
        self.subgoals = [str(s) for s in subgoals]
        self.subgoal_index = 0

    def advance_subgoal(self) -> Optional[str]:
        if self.subgoal_index < len(self.subgoals):
            self.subgoal_index += 1
        return self.active_subgoal

    @property
    def active_subgoal(self) -> Optional[str]:
        if 0 <= self.subgoal_index < len(self.subgoals):
            return self.subgoals[self.subgoal_index]
        return None

    def recent_decisions(self, k: int) -> list:
        out = [t.decision for t in self.turns if t.decision]
        return out[-k:] if k > 0 else []


@dataclass
class MonitorContext:
    """The cheap, bounded prompt read at the motion timescale.

    Deliberately omits the full history. The point is not only cost: a long
    transcript buries the current observation under the model's own prior
    reasoning, which is how an agent ends up re-deciding what it already decided
    instead of reacting to what it can see now.
    """

    #: how many recent decisions to carry as short lines
    decisions: int = 4
    #: attach the newest frame when the caller has one and vision is on
    include_image: bool = True

    def messages(self, transcript: Transcript, observation: str, *,
                 image=None, extra: str = "") -> list:
        lines = []
        goal = transcript.active_subgoal
        if goal:
            done = transcript.subgoal_index
            total = len(transcript.subgoals)
            lines.append(f"Current subgoal ({done + 1}/{total}): {goal}")
        recent = transcript.recent_decisions(self.decisions)
        if recent:
            lines.append("Recent decisions: " + " -> ".join(recent))
        lines.append(observation)
        if extra:
            lines.append(extra)
        body = "\n".join(lines)

        system = transcript.system
        out = [ChatMessage.system(system)] if system else []
        if image is not None and self.include_image:
            from harness.perception.vision import encode_image

            out.append(ChatMessage.user_vision(body, encode_image(image)))
        else:
            out.append(ChatMessage.user(body))
        return out


@dataclass
class DeliberationContext:
    """The full prompt, consulted only at planning points."""

    #: cap on turns replayed; None keeps everything. A cap is a last resort against
    #: context overflow, not a routine economy -- deliberation is meant to be rich.
    max_turns: Optional[int] = None

    def messages(self, transcript: Transcript, observation: str, *,
                 image=None, extra: str = "") -> list:
        out = [ChatMessage.system(transcript.system)] if transcript.system else []
        turns = transcript.turns
        if self.max_turns is not None and len(turns) > self.max_turns:
            turns = turns[-self.max_turns:]
        for t in turns:
            if t.observation:
                out.append(ChatMessage.user(t.observation))
            if t.reply:
                out.append(ChatMessage.assistant(t.reply))
            if t.feedback:
                out.append(ChatMessage.user("Tool result: " + t.feedback))
        if transcript.subgoals:
            done = transcript.subgoal_index
            plan = "\n".join(
                f"{'x' if i < done else ('>' if i == done else ' ')} {s}"
                for i, s in enumerate(transcript.subgoals))
            out.append(ChatMessage.user("Subgoals so far:\n" + plan))
        body = observation if not extra else observation + "\n" + extra
        if image is not None:
            from harness.perception.vision import encode_image

            out.append(ChatMessage.user_vision(body, encode_image(image)))
        else:
            out.append(ChatMessage.user(body))
        return out


def estimate_prompt_chars(messages) -> int:
    """Rough size of a message list, for asserting the monitor stays bounded.

    Counts text only: an attached image is a fixed cost per turn either way, and
    what we care about is that history does not accumulate.
    """
    total = 0
    for m in messages:
        if isinstance(m.content, str):
            total += len(m.content)
        else:
            for blk in m.content:
                if blk.get("type") == "text":
                    total += len(blk.get("text", ""))
    return total
