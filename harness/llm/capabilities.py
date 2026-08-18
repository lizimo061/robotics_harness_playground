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
    """

    sampling_params: bool = True
    thinking: str = "none"
    effort: bool = False
    max_output: int = 8192
    context: int = 200_000
    price_in: Optional[float] = None
    price_out: Optional[float] = None
    price_cache_read: Optional[float] = None


# Anthropic, verified against the published model table.
# Sampling params are rejected from Opus 4.7 / Sonnet 5 onward.
_ANTHROPIC: dict[str, ModelCaps] = {
    "claude-fable-5": ModelCaps(
        sampling_params=False, thinking="adaptive", effort=True,
        max_output=128_000, context=1_000_000, price_in=10.0, price_out=50.0,
    ),
    "claude-mythos-5": ModelCaps(
        sampling_params=False, thinking="adaptive", effort=True,
        max_output=128_000, context=1_000_000, price_in=10.0, price_out=50.0,
    ),
    "claude-opus-5": ModelCaps(
        sampling_params=False, thinking="adaptive", effort=True,
        max_output=128_000, context=1_000_000, price_in=5.0, price_out=25.0,
    ),
    "claude-opus-4-8": ModelCaps(
        sampling_params=False, thinking="adaptive", effort=True,
        max_output=128_000, context=1_000_000, price_in=5.0, price_out=25.0,
    ),
    "claude-opus-4-7": ModelCaps(
        sampling_params=False, thinking="adaptive", effort=True,
        max_output=128_000, context=1_000_000, price_in=5.0, price_out=25.0,
    ),
    "claude-opus-4-6": ModelCaps(
        sampling_params=True, thinking="adaptive", effort=True,
        max_output=128_000, context=1_000_000, price_in=5.0, price_out=25.0,
    ),
    "claude-sonnet-5": ModelCaps(
        sampling_params=False, thinking="adaptive", effort=True,
        max_output=128_000, context=1_000_000, price_in=3.0, price_out=15.0,
    ),
    "claude-sonnet-4-6": ModelCaps(
        sampling_params=True, thinking="adaptive", effort=True,
        max_output=128_000, context=1_000_000, price_in=3.0, price_out=15.0,
    ),
    "claude-haiku-4-5": ModelCaps(
        sampling_params=True, thinking="budget",
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
_OTHER: dict[str, ModelCaps] = {
    "deepseek-chat": ModelCaps(max_output=384_000, context=1_000_000),
    "deepseek-reasoner": ModelCaps(max_output=384_000, context=1_000_000),
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
