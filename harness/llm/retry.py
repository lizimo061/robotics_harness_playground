"""Retry helper with exponential backoff (for transient LLM errors)."""
from __future__ import annotations

import random
import time

from harness.utils.logging import get_logger

log = get_logger("harness.llm.retry")


def with_retries(fn, *, retries: int = 3, base_delay: float = 0.5, max_delay: float = 8.0, exceptions=(Exception,)):
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except exceptions as e:  # noqa: PERF203
            last = e
            if attempt == retries:
                break
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 0.2)
            log.warning("attempt %d/%d failed (%s); retrying in %.2fs", attempt + 1, retries, e, delay)
            time.sleep(delay)
    raise last  # type: ignore[misc]
