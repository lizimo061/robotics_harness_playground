"""Detection as a tool, so perception can be measured instead of assumed.

The harness's query API (``get_object_pos``) reads simulator ground truth. That is
useful and fast, but it makes every result a *privileged-state* measurement: it scores
planning and grounding, never perception. CaP-X's central finding is that scores
depend on exactly this scaffolding -- they improve with human-crafted abstractions and
degrade as those are removed -- so the difference has to be a number, not a caveat.

This module supplies the other side of that comparison: a :class:`Detector` protocol
mirroring the existing :class:`~harness.policies.base.Policy` seam, with

- :class:`RemoteDetector` -- an HTTP client, so a real model (GroundingDINO, SAM,
  Molmo-style pointing) can live out of process on its own GPU;
- :class:`OracleDetector` -- projects ground-truth poses into the image and adds
  configurable noise and dropout. Not a stand-in for a real detector, and not
  presented as one: it exists so the *plumbing and the tier switch* can be tested and
  so the sensitivity of a task to detection error can be measured without a GPU.

An honest report says which detector produced a number. ``Detection.source`` carries
that, and the tier is recorded in the run config.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from harness.utils.logging import get_logger

log = get_logger("harness.perception.detect")


@dataclass
class Detection:
    """One detected object.

    ``pixel`` is (x, y) in image coordinates -- what a real detector returns.
    ``position`` is the 3-D pose *if* the backend can supply one; a camera-only
    detector leaves it None, and an agent that needs metric coordinates then has to
    ask for them or work in pixels.
    """

    name: str = ""
    pixel: Optional[tuple] = None
    position: Optional[np.ndarray] = None
    confidence: float = 0.0
    #: which backend produced this, so a report can say what it measured
    source: str = ""
    extra: dict = field(default_factory=dict)

    def describe(self) -> str:
        bits = [self.name]
        if self.pixel is not None:
            bits.append(f"at pixel ({int(self.pixel[0])}, {int(self.pixel[1])})")
        if self.position is not None:
            p = np.asarray(self.position).ravel()
            bits.append("at (" + ", ".join(f"{float(v):.3f}" for v in p) + ")")
        if self.confidence:
            bits.append(f"conf {self.confidence:.2f}")
        return " ".join(bits)


@runtime_checkable
class Detector(Protocol):
    """Open-vocabulary detection over an image."""

    name: str

    def detect(self, image, query: str) -> list:
        """Return Detections for `query`, best first. Empty list means not found."""
        ...


class OracleDetector:
    """Ground truth, optionally degraded on purpose.

    Two uses, both legitimate, neither of them "pretending to be a detector":

    - **plumbing** -- exercise the tools and the tier switch without a GPU;
    - **sensitivity** -- with ``noise_m`` / ``dropout`` set, measure how much a task
      depends on detection accuracy. A task that collapses at 2cm of noise was being
      carried by the privileged state, which is worth knowing before believing a
      leaderboard built on it.

    At its defaults (no noise, no dropout) it is exactly the privileged tier, and the
    ``source`` field says so.
    """

    def __init__(self, env=None, *, noise_m: float = 0.0, dropout: float = 0.0,
                 seed: int = 0, image_size: int = 256) -> None:
        self.env = env
        self._noise = float(noise_m)
        self._dropout = float(dropout)
        self._rng = np.random.default_rng(seed)
        self._image_size = int(image_size)
        self.name = ("oracle" if not (noise_m or dropout)
                     else f"oracle(noise={noise_m}m,dropout={dropout})")

    def bind(self, env) -> "OracleDetector":
        self.env = env
        return self

    def detect(self, image, query: str) -> list:
        env = self.env
        if env is None:
            return []
        query_l = str(query).strip().lower()
        names = list(getattr(env, "list_objects", lambda: [])() or [])
        matches = [n for n in names if query_l and query_l in str(n).lower()]
        if not matches:
            # an open-vocabulary detector fails by returning nothing, not by raising
            return []
        out = []
        for name in matches:
            if self._dropout and float(self._rng.random()) < self._dropout:
                continue  # a real detector misses things; so does this one, on purpose
            pos = getattr(env, "get_object_pos", lambda _n: None)(name)
            if pos is None:
                continue
            pos = np.asarray(pos, dtype=np.float32).ravel().copy()
            if self._noise:
                pos[:min(3, pos.size)] += self._rng.normal(0.0, self._noise,
                                                           size=min(3, pos.size))
            out.append(Detection(name=str(name), pixel=self._project(pos, image),
                                 position=pos, confidence=1.0, source=self.name))
        return out

    def _project(self, pos, image) -> Optional[tuple]:
        """A crude planar projection, so pixel-space tools have something to consume.

        Deliberately not a calibrated camera model: the pixel coordinate is only used
        to give the agent a perception-shaped interface. Anything that needs true
        geometry should use `position`.
        """
        size = self._image_size
        if image is not None and getattr(image, "ndim", 0) >= 2:
            size = int(image.shape[0])
        p = np.asarray(pos, dtype=np.float32).ravel()
        if p.size < 2:
            return None
        return (float(np.clip(p[0], 0, 1) * size if abs(p[0]) <= 1 else p[0]),
                float(np.clip(p[1], 0, 1) * size if abs(p[1]) <= 1 else p[1]))


class RemoteDetector:
    """An HTTP detector, so the model can live in its own process.

    Mirrors RemotePolicy: POST an image and a text query, receive detections. Kept
    deliberately thin -- a GroundingDINO or Molmo server has its own environment, and
    forcing it into this one is how a harness becomes unusable.

    Wire format::

        POST {base_url}/detect
        {"query": "the red cube", "image": "<base64 png>"}
        -> {"detections": [{"name": ..., "pixel": [x, y], "position": [x,y,z],
                            "confidence": 0.9}]}
    """

    def __init__(self, base_url: str, *, timeout: float = 20.0,
                 name: str = "remote") -> None:
        self._base = base_url.rstrip("/")
        self._timeout = float(timeout)
        self.name = name

    def detect(self, image, query: str) -> list:
        import httpx

        from harness.perception.vision import encode_image

        payload = {"query": str(query)}
        if image is not None:
            payload["image"] = encode_image(image)
        try:
            resp = httpx.post(f"{self._base}/detect", json=payload, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # noqa: BLE001 - a detector outage is not a task failure
            log.warning("detector request failed (%s: %s)", type(e).__name__, e)
            return []
        out = []
        for d in (data.get("detections") or []):
            pos = d.get("position")
            out.append(Detection(
                name=str(d.get("name") or query),
                pixel=tuple(d["pixel"]) if d.get("pixel") else None,
                position=None if pos is None else np.asarray(pos, dtype=np.float32),
                confidence=float(d.get("confidence") or 0.0),
                source=self.name,
                extra={k: v for k, v in d.items()
                       if k not in ("name", "pixel", "position", "confidence")},
            ))
        return out


def get_detector(spec, env=None):
    """Build a detector from a config value.

    Accepts ``None``, ``"oracle"``, ``{"type": "oracle", "noise_m": 0.01}``, or
    ``{"type": "remote", "base_url": "http://localhost:8100"}``.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"type": spec}
    if isinstance(spec, str) or not hasattr(spec, "get"):
        raise ValueError(f"cannot build a detector from {spec!r}")
    kind = str(spec.get("type") or "oracle").lower()
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    if kind == "oracle":
        return OracleDetector(env, **kwargs)
    if kind == "remote":
        return RemoteDetector(**kwargs)
    raise ValueError(f"unknown detector type {kind!r}; use 'oracle' or 'remote'")
