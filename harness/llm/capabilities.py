"""Per-model request shape and pricing.

Frontier models have diverged on what a request may legally contain, so the
differences live here as data rather than as branches at every call site.

The immediate reason this module exists: current Claude models **reject a
non-default `temperature` with HTTP 400**. A client that always sends one
cannot call them at all. Capability lookup lets each client omit what the
target model does not accept.

Pricing is only recorded where it has been verified against the vendor's
published table. Unknown prices stay ``None`` and ``estimate_cost`` returns
``None`` rather than inventing a number -- a wrong cost on a leaderboard is
worse than a blank cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Cache multipliers relative to the input price (Anthropic).
_CACHE_READ_MULT = 0.1
_CACHE_WRITE_MULT = 1.25


@dataclass(frozen=True)
class ModelCaps:
    """What a model's request may contain, and what its tokens cost.

    sampling_params: whether temperature / top_p / top_k are accepted at all.
    thinking:        "none" | "adaptive" | "budget"
    effort:          whether output_config.effort is supported.
    price_in/out:    USD per 1M tokens; None means "unknown, do not guess".
    price_cache_read: USD per 1M cached-read tokens. Providers differ sharply
        here -- Anthropic discounts ~10x, DeepSeek ~31x -- so a single global
        multiplier misprices badly. Falls back to _CACHE_READ_MULT when unset.
    vision:          whether the model accepts image blocks at all.
    """

    sampling_params: bool = True
    thinking: str = "none"
    effort: bool = False
    max_output: int = 8192
    context: int = 200_000
    price_in: Optional[float] = None
    price_out: Optional[float] = None
    price_cache_read: Optional[float] = None
    #: Defaults to False on purpose, unlike every other field here. The rest of
    #: this dataclass is permissive so an unlisted model still runs; vision is
    #: the opposite, because a wrong True is silent. A text-only model handed an
    #: image either 400s or -- worse -- answers from the text alone, and the run
    #: still looks like a vision evaluation. False makes an unverified model
    #: refuse a vision run instead of producing a publishable-looking zero.
    vision: bool = False


# Anthropic, verified against the published model table.
# Sampling params are rejected from Opus 4.7 / Sonnet 5 onward.
#
# vision=True throughout: every Claude model the API has served since Claude 3
# accepts base64 image blocks alongside text, and this harness sends exactly
# that shape (see ChatMessage.to_anthropic). No Claude entry here is text-only.
_ANTHROPIC: dict[str, ModelCaps] = {
    "claude-fable-5": ModelCaps(
        sampling_params=False, thinking="adaptive", effort=True, vision=True,
        max_output=128_000, context=1_000_000, price_in=10.0, price_out=50.0,
    ),
    "claude-mythos-5": ModelCaps(
        sampling_params=False, thinking="adaptive", effort=True, vision=True,
        max_output=128_000, context=1_000_000, price_in=10.0, price_out=50.0,
    ),
    "claude-opus-5": ModelCaps(
        sampling_params=False, thinking="adaptive", effort=True, vision=True,
        max_output=128_000, context=1_000_000, price_in=5.0, price_out=25.0,
    ),
    "claude-opus-4-8": ModelCaps(
        sampling_params=False, thinking="adaptive", effort=True, vision=True,
        max_output=128_000, context=1_000_000, price_in=5.0, price_out=25.0,
    ),
    "claude-opus-4-7": ModelCaps(
        sampling_params=False, thinking="adaptive", effort=True, vision=True,
        max_output=128_000, context=1_000_000, price_in=5.0, price_out=25.0,
    ),
    "claude-opus-4-6": ModelCaps(
        sampling_params=True, thinking="adaptive", effort=True, vision=True,
        max_output=128_000, context=1_000_000, price_in=5.0, price_out=25.0,
    ),
    "claude-sonnet-5": ModelCaps(
        sampling_params=False, thinking="adaptive", effort=True, vision=True,
        max_output=128_000, context=1_000_000, price_in=3.0, price_out=15.0,
    ),
    "claude-sonnet-4-6": ModelCaps(
        sampling_params=True, thinking="adaptive", effort=True, vision=True,
        max_output=128_000, context=1_000_000, price_in=3.0, price_out=15.0,
    ),
    "claude-haiku-4-5": ModelCaps(
        sampling_params=True, thinking="budget", vision=True,
        max_output=64_000, context=200_000, price_in=1.0, price_out=5.0,
    ),
}

# DeepSeek, from the vendor's published pricing table.
#
# Two things make these prices awkward, and both are handled by recording the
# PEAK rate: off-peak is exactly half of peak (peak = 01:00-04:00 and
# 06:00-10:00 UTC), and cached input is ~31x cheaper than fresh input rather
# than Anthropic's ~10x. Peak is the conservative choice -- for a cost
# comparison, overstating is far safer than understating, and an off-peak run
# simply comes in under the reported figure.
#
# vision stays at its default False for both: DeepSeek's chat-completions models
# are text-only as far as this repo has verified, and nothing here has ever been
# observed accepting an image block. If a served v4 model does take images, the
# fix is one flag plus the evidence -- not an optimistic default.
_DEEPSEEK: dict[str, ModelCaps] = {
    "deepseek-v4-pro": ModelCaps(
        max_output=384_000, context=1_000_000,
        price_in=1.32, price_out=3.96, price_cache_read=0.044,
    ),
    "deepseek-v4-flash": ModelCaps(
        max_output=384_000, context=1_000_000,
        price_in=0.44, price_out=1.32, price_cache_read=0.014,
    ),
}

# Alias names. `deepseek-chat` was observed resolving to deepseek-v4-flash
# against the live API (the response's `model` field), so its limits are set
# to the served model's rather than the stale 64K/8K of an older generation.
#
# Prices stay None deliberately: an alias can be repointed server-side at any
# time, and estimate_cost() prices the model named in the *response*, so a
# served v4 model is billed correctly without trusting the alias.
#
# vision=False is explicit rather than inherited, because these two aliases are
# the ones this repo's configs actually name: DeepSeek's chat and reasoner
# endpoints take text only. Sending them a frame does not error loudly -- the
# model answers from the text and the run reads like a failed vision eval.
_OTHER: dict[str, ModelCaps] = {
    "deepseek-chat": ModelCaps(max_output=384_000, context=1_000_000, vision=False),
    "deepseek-reasoner": ModelCaps(max_output=384_000, context=1_000_000, vision=False),
}

_DEFAULT = ModelCaps()


def get_caps(model: str) -> ModelCaps:
    """Look up a model's capabilities; longest matching prefix wins.

    Unknown models get a permissive default (sampling allowed, no pricing), so
    a new or self-hosted model still runs -- it just reports no cost.
    """
    name = (model or "").strip().lower()
    if not name:
        return _DEFAULT
    table = {**_ANTHROPIC, **_DEEPSEEK, **_OTHER}
    if name in table:
        return table[name]
    matches = [k for k in table if name.startswith(k)]
    if matches:
        return table[max(matches, key=len)]
    return _DEFAULT


def supports_vision(model: str) -> bool:
    """Whether `model` is known to accept image input.

    Unknown models answer False. That is not a claim that they are blind -- it is
    a refusal to assume they can see. Vision support cannot be probed cheaply and
    a wrong "yes" is invisible: a text-only model handed a rendered frame either
    rejects the request or replies from the text alone, and the resulting zero is
    indistinguishable from genuine incapability. So the burden of proof sits on
    the model: add it to the tables above, with the evidence, to run vision.
    """
    return get_caps(model).vision


class VisionUnsupportedError(ValueError):
    """A run that needs pixels was pointed at a model that cannot receive them.

    Raised instead of warned. The failure this closes is silent by construction:
    ``use_vision=True`` on deepseek-chat, or the ``perception`` tier (which
    withdraws the ground-truth object queries and leaves only detect/point_at
    over the image), produces a complete-looking episode, a real success rate and
    a real cost -- all of it measuring a model that was never shown the scene.
    """


def check_vision_config(
    model: str,
    *,
    use_vision: bool = False,
    tier: str = "privileged",
) -> None:
    """Raise VisionUnsupportedError if this configuration needs a sighted model.

    A configuration needs vision when ``use_vision`` is set (the camera frame is
    attached to every observation) or when the tier is anything but
    ``privileged`` (ground-truth queries are withdrawn, so the only way to locate
    an object is to look at it). Either paired with a text-only model is a
    meaningless evaluation, so it is refused up front rather than reported.
    """
    tier = str(tier or "privileged")
    reasons = []
    if use_vision:
        reasons.append("use_vision=True attaches the camera frame to every observation")
    if tier != "privileged":
        reasons.append(
            f"tier={tier!r} withdraws the ground-truth object queries, leaving "
            "detect/point_at over the image as the only way to locate anything"
        )
    if not reasons:
        return
    if supports_vision(model):
        return
    # "recorded as text-only" and "never recorded at all" are different claims,
    # and only the second one is the harness's ignorance rather than the model's
    # limitation. get_caps falls back to _DEFAULT, so identity tells them apart --
    # and a suffixed id that prefix-matches a real entry counts as recorded.
    known = get_caps(model) is not _DEFAULT
    verdict = (
        # Phrased as a fact about the table, not about the vendor: the table is
        # what the harness can defend, and every entry cites its evidence.
        f"model {model!r} is recorded as text-only (no image input)"
        if known
        else f"model {model!r} is not in the capability table, so its image support is unverified"
    )
    raise VisionUnsupportedError(
        f"{verdict}, but this run requires vision: "
        + "; ".join(reasons)
        + ". A blind model scores zero here for the wrong reason, and the result "
        "looks publishable. Fix one of: (a) run a vision-capable model "
        "(any Anthropic claude-* model in harness/llm/capabilities.py); "
        "(b) drop to tier='privileged' with use_vision=False for a text-only, "
        "ground-truth run; (c) if you know this model can see, add vision=True "
        "to its ModelCaps entry along with the evidence; (d) for offline plumbing "
        "tests only, use the mock provider or pass allow_blind_vision=True."
    )


@dataclass
class TokenUsage:
    """Provider-agnostic token counts.

    Normalises the two wire shapes the harness sees: Anthropic's
    ``input_tokens``/``output_tokens`` (+ explicit cache fields) and the
    OpenAI-compatible ``prompt_tokens``/``completion_tokens``.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }

    @classmethod
    def from_raw(cls, usage: Optional[dict]) -> "TokenUsage":
        if not usage:
            return cls()

        def _i(*keys) -> int:
            for k in keys:
                v = usage.get(k)
                if isinstance(v, (int, float)):
                    return int(v)
            return 0

        cached = _i("cache_read_input_tokens")
        if not cached:
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                cached = int(details.get("cached_tokens") or 0)

        fresh = _i("input_tokens", "prompt_tokens")

        # DeepSeek splits the prompt into its own hit/miss fields, and
        # prompt_tokens is their SUM -- counting both would double-bill the
        # cached half at the full input rate. Prefer the explicit miss count.
        if "prompt_cache_hit_tokens" in usage or "prompt_cache_miss_tokens" in usage:
            cached = cached or _i("prompt_cache_hit_tokens")
            miss = _i("prompt_cache_miss_tokens")
            if miss or cached:
                fresh = miss

        return cls(
            input_tokens=fresh,
            output_tokens=_i("output_tokens", "completion_tokens"),
            cache_read_tokens=cached,
            cache_write_tokens=_i("cache_creation_input_tokens"),
        )


def estimate_cost(model: str, usage: Any) -> Optional[float]:
    """USD for one call, or None when the model's pricing is unknown.

    Cache reads bill at ~0.1x input and writes at ~1.25x, so a harness that
    re-sends the whole conversation every step cannot price itself from
    input+output alone.
    """
    caps = get_caps(model)
    if caps.price_in is None or caps.price_out is None:
        return None
    u = usage if isinstance(usage, TokenUsage) else TokenUsage.from_raw(usage)
    per_token_in = caps.price_in / 1_000_000
    per_token_out = caps.price_out / 1_000_000
    # Prefer the vendor's own cached-read rate; the multiplier is a fallback.
    if caps.price_cache_read is not None:
        per_token_cached = caps.price_cache_read / 1_000_000
    else:
        per_token_cached = per_token_in * _CACHE_READ_MULT
    return (
        u.input_tokens * per_token_in
        + u.output_tokens * per_token_out
        + u.cache_read_tokens * per_token_cached
        + u.cache_write_tokens * per_token_in * _CACHE_WRITE_MULT
    )
