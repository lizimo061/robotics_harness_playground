"""Run lock: what actually ran, so a resume cannot silently change it.

A results file is only trustworthy if you can tell what produced it. The lock
records the resolved inputs -- harness version and git SHA, whether the install
was editable, the task set and its content digest, every agent/model and its
kwargs, the seed list, and the invocation -- and a resume compares it before
appending to an existing run.

The comparison deliberately excludes the timestamp and the invocation string:
those differ on every resume by construction, while everything that changes the
*meaning* of a number must match.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

LOCKFILE_VERSION = "1.0"


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - not a git checkout, or no git
        return None


def _is_editable_install() -> bool:
    """True when running from a working tree rather than an installed wheel.

    Harbor flags this because a run from a dirty local checkout is not
    reproducible from the recorded version alone -- the version string does not
    identify the code that ran.
    """
    return (Path(__file__).resolve().parents[2] / ".git").exists()


def digest_of(obj: Any) -> str:
    """Stable sha256 over a JSON-able object (sorted keys, so order-free)."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


@dataclass
class RunLock:
    lockfile_version: str = LOCKFILE_VERSION
    created_at: str = ""
    invocation: str = ""

    harness_version: str = ""
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    is_editable_installation: bool = False
    python: str = ""

    tasks: list = field(default_factory=list)
    task_digest: str = ""
    agents: list = field(default_factory=list)
    seeds: list = field(default_factory=list)
    episodes_per_cell: int = 1
    concurrency: int = 1

    #: fields excluded from equality: they change on every resume by design.
    VOLATILE = ("created_at", "invocation")

    @classmethod
    def capture(
        cls,
        *,
        tasks: list,
        agents: list,
        seeds: list,
        episodes_per_cell: int = 1,
        concurrency: int = 1,
        created_at: str = "",
    ) -> "RunLock":
        from harness import __version__ as version  # local import: avoids a cycle

        status = _git("status", "--porcelain")
        return cls(
            created_at=created_at,
            invocation=" ".join(sys.argv[:1] + sys.argv[1:]),
            harness_version=str(version),
            git_commit=_git("rev-parse", "HEAD"),
            git_dirty=(bool(status) if status is not None else None),
            is_editable_installation=_is_editable_install(),
            python=sys.version.split()[0],
            tasks=list(tasks),
            task_digest=digest_of(sorted(tasks)),
            agents=list(agents),
            seeds=list(seeds),
            episodes_per_cell=episodes_per_cell,
            concurrency=concurrency,
        )

    def comparable(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in self.VOLATILE}

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return isinstance(other, RunLock) and self.comparable() == other.comparable()

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return p

    @classmethod
    def read(cls, path) -> Optional["RunLock"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        known = {f for f in cls.__dataclass_fields__}  # tolerate added/removed fields
        return cls(**{k: v for k, v in data.items() if k in known})

    def describe_mismatch(self, other: "RunLock") -> list[str]:
        """Human-readable list of what differs, for a refused resume."""
        a, b = self.comparable(), other.comparable()
        return [f"{k}: existing={b.get(k)!r} requested={a.get(k)!r}"
                for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


class LockMismatch(RuntimeError):
    """A resume was attempted against a run configured differently."""
