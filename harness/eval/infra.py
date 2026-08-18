"""Post-hoc classification of infrastructure failures.

Adapted from SWE-bench's ``infra_failure.py``. Two properties are deliberate
and worth preserving exactly:

1. Classification is **read-only and post-hoc**. It never decides whether an
   episode runs, so it cannot drop an episode from a sweep.
2. It is **advisory**: a flagged episode stays in the failure set, and the
   scoring denominator is unchanged. Infra counts are reported as a separate
   dimension, never netted out of the success rate.

Why that matters here more than it does for a coding benchmark: a physics
simulator is flakier than a container running pytest. Out-of-memory on a 16GB
GPU, a missing display, a wedged sim process -- all produce a zero that says
nothing about the model. Silently counting them as capability failures makes a
model look worse on a bad machine day; silently excluding them inflates it.
Reporting them separately is the only honest option.
"""
from __future__ import annotations

import re
from typing import Optional

# Cannot plausibly be caused by the policy's actions.
TIER_ENVIRONMENT = "environment"
# Could come from either a broken environment or a genuinely bad rollout, so it
# is reported apart from confirmed environment faults rather than merged in.
TIER_AMBIGUOUS = "ambiguous"

#: (reason, tier, pattern) -- first match wins, evaluated multiline.
INFRA_FAILURE_SIGNATURES: tuple[tuple[str, str, str], ...] = (
    # --- simulator / GPU, the common ones for an Isaac Sim suite ---
    ("out_of_memory", TIER_ENVIRONMENT,
     r"CUDA out of memory|Cannot allocate memory|OutOfMemoryError|^Killed$"),
    ("display_unavailable", TIER_ENVIRONMENT,
     r"cannot open display|Missing X server|unable to open X display|no EGL"),
    ("gpu_unavailable", TIER_ENVIRONMENT,
     r"no CUDA-capable device|CUDA driver version is insufficient|failed to initialize NVML"),
    ("sim_startup_failed", TIER_ENVIRONMENT,
     r"Failed to create simulation|AppLauncher.*failed|Isaac Sim.*failed to start"),
    ("container_unavailable", TIER_ENVIRONMENT,
     r"Error response from daemon|Cannot connect to the Docker daemon"),
    ("network_unreachable", TIER_ENVIRONMENT,
     r"Could not resolve host|Temporary failure in name resolution"),
    # --- provider-side, also not the model's capability ---
    ("provider_rate_limited", TIER_ENVIRONMENT, r"rate.?limit|429 Too Many Requests"),
    ("provider_unavailable", TIER_ENVIRONMENT, r"HTTP 5\d\d|overloaded_error|service unavailable"),
    # --- could be either ---
    ("missing_asset", TIER_AMBIGUOUS,
     r"Could not open asset|Unresolved reference|ModuleNotFoundError|No such file or directory"),
    ("physics_instability", TIER_AMBIGUOUS,
     r"\bnan\b.*(position|velocity)|solver diverged|physics.*unstable"),
    ("episode_timed_out", TIER_AMBIGUOUS, r"Timeout error: \d+ seconds exceeded|timed out after"),
)

_COMPILED = tuple(
    (reason, tier, re.compile(pat, re.MULTILINE | re.IGNORECASE))
    for reason, tier, pat in INFRA_FAILURE_SIGNATURES
)


def classify_infra_failure(text: Optional[str]) -> Optional[dict]:
    """Return ``{"reason", "tier"}`` for the first matching signature, else None.

    ``text`` is whatever the run captured -- an exception traceback, a sim log,
    stderr. None/empty yields None (i.e. no evidence of an infra fault, which
    is not the same as evidence of none).
    """
    if not text:
        return None
    for reason, tier, rx in _COMPILED:
        if rx.search(text):
            return {"reason": reason, "tier": tier}
    return None


def summarize_infra(records) -> dict:
    """Count infra-flagged episodes by tier and reason, without reweighting.

    The returned counts sit beside the success rate; they never modify it.
    """
    env = amb = 0
    reasons: dict[str, int] = {}
    for r in records:
        info = r.get("infra_failure") if hasattr(r, "get") else None
        if not info:
            continue
        tier = info.get("tier")
        if tier == TIER_ENVIRONMENT:
            env += 1
        elif tier == TIER_AMBIGUOUS:
            amb += 1
        reason = info.get("reason") or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    if not (env or amb):
        return {}
    return {
        "environment_failures": env,
        "ambiguous_failures": amb,
        "reasons": reasons,
        "denominator_unchanged": True,  # stated explicitly so readers need not infer it
    }
