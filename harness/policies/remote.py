"""HTTP client for a policy server (the executor behind run_policy).

Speaks the same protocol harness.serving exposes, so any of these work:

- the harness's own policy server (examples/serve_robolab.py)
- a real VLA (pi0.5, GR00T, ...) wrapped behind that protocol

    POST /begin -> {"instruction", "env_id"}
    POST /act   -> {"instruction", "observation_text", "image_b64"?, "env_id"}
                -> {"action": [float, ...]}
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from harness.llm.retry import with_retries
from harness.policies.base import Policy
from harness.types import ActionSpace
from harness.utils.logging import get_logger

log = get_logger("harness.policies.remote")

_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class PolicyServerError(RuntimeError):
    """The policy server returned an unusable response."""


class TransientPolicyServerError(PolicyServerError):
    """A retryable policy-server error (rate limit, 5xx, network blip)."""


class RemotePolicy(Policy):
    """Query a policy server for one action per step."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        action_dim: int = 8,
        env_id: int = 0,
        timeout: float = 120.0,
        retries: int = 2,
        send_images: bool = True,
    ) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._action_dim = action_dim
        self._env_id = env_id
        self._timeout = timeout
        self._retries = retries
        self._send_images = send_images
        self._instruction = ""

    @property
    def name(self) -> str:
        return f"remote_policy({self._base_url})"

    # -- Policy protocol -------------------------------------------------- #
    def begin(self, instruction: str, action_space: Optional[ActionSpace] = None) -> None:
        self._instruction = instruction or ""
        if action_space is not None and action_space.dim:
            self._action_dim = action_space.dim
        payload = {"instruction": self._instruction, "env_id": self._env_id}
        try:
            self._post("/begin", payload)
        except PolicyServerError as e:
            # /begin is advisory: act() carries the instruction anyway, so a
            # server that only implements /act stays usable.
            log.warning("policy server /begin failed (%s); continuing", e)

    def act(self, observation_text: str, image: Optional[np.ndarray] = None) -> np.ndarray:
        payload = {
            "instruction": self._instruction,
            "observation_text": observation_text or "",
            "env_id": self._env_id,
        }
        if image is not None and self._send_images:
            from harness.perception.vision import encode_image

            payload["image_b64"] = encode_image(image)

        data = self._post("/act", payload)
        vec = data.get("action")
        if vec is None:
            raise PolicyServerError(f"no 'action' in policy response: {str(data)[:200]}")
        return np.asarray(vec, dtype=np.float32).ravel()

    def reset(self) -> None:
        self._instruction = ""

    # -- transport -------------------------------------------------------- #
    def _post(self, path: str, payload: dict) -> dict:
        import httpx

        url = f"{self._base_url}{path}"

        def call() -> dict:
            try:
                resp = httpx.post(url, json=payload, timeout=self._timeout)
            except httpx.TransportError as e:
                raise TransientPolicyServerError(f"network error: {e}") from e
            if resp.status_code in _TRANSIENT_STATUS:
                raise TransientPolicyServerError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 400:
                raise PolicyServerError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            try:
                return resp.json()
            except ValueError as e:
                raise PolicyServerError(f"non-JSON response: {resp.text[:200]}") from e

        return with_retries(call, retries=self._retries, exceptions=(TransientPolicyServerError,))
