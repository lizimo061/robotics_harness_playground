"""Local Claude Code CLI as an LLM backend (a testing convenience, not a peer).

This driver shells out to the ``claude`` CLI in headless print mode instead of
POSTing to a paid HTTP endpoint, so the harness can be exercised end-to-end --
runner, agent loop, parsers, reports -- on a workstation that already has the
CLI authenticated, without spending API budget per sweep.

Be honest about what this is NOT:

* It is not the Messages API. The CLI wraps the model in Claude Code's own
  coding-agent system prompt, its own context assembly, and its own model
  routing. The token distribution you get back is that agent's, not the raw
  model's.
* Latency is process-spawn + agent-harness latency (seconds of fixed overhead
  per call), so anything timing-sensitive measured through here is measuring
  the CLI, not the model.
* ``--model`` is a request, not a guarantee: the CLI may route side work to
  small helper models, and the effective model can change with the installed
  CLI version.
* Therefore: do not publish numbers produced by this backend as
  model-comparison results. Use ``provider: claude`` (the HTTP API) for
  anything anyone will cite.

It is also text-only. The CLI's prompt channel takes a string; there is no
place to attach the raw base64 frames that ``ChatMessage.user_vision`` builds,
so vision messages are refused rather than quietly flattened to their captions
(see ``on_image``).
"""
from __future__ import annotations

import json
import contextlib
import shutil
import subprocess
import tempfile
from typing import Any, Optional

from harness.llm.base import (
    ChatMessage,
    LLMClient,
    LLMError,
    LLMResponse,
    TransientLLMError,
)
from harness.llm.retry import with_retries
from harness.utils.logging import get_logger

log = get_logger("harness.llm.claude_code")

# Statuses the CLI surfaces in api_error_status that are worth another attempt.
_TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Substrings in stderr/stdout that mean "the invocation was fine, the call
# wasn't" -- these are the only non-status signals we treat as retryable.
_TRANSIENT_MARKERS = (
    "overloaded",
    "rate limit",
    "rate_limit",
    "socket hang up",
    "econnreset",
    "econnrefused",
    "etimedout",
    "timed out",
    "temporarily unavailable",
    "internal server error",
    "service unavailable",
)

# Which of the CLI's usage keys we forward. Deliberately a whitelist: the raw
# usage blob carries per-iteration arrays and tier metadata that would bloat
# every result file, and we would rather report nothing than invent a number.
_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}

_STDERR_LIMIT = 800


def _truncate(text: str, limit: int = _STDERR_LIMIT) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars]"


def _block_type(block: Any) -> str:
    return block.get("type", "") if isinstance(block, dict) else ""


def _is_image_block(block: Any) -> bool:
    return _block_type(block) in ("image", "image_url", "input_image")


def _message_has_image(msg: ChatMessage) -> bool:
    return (not isinstance(msg.content, str)
            and any(_is_image_block(b) for b in msg.content))


def _spill_image(block: Any, directory: str, index: int) -> Optional[str]:
    """Write one image block to a PNG in `directory`; return its absolute path.

    Accepts the two shapes this repo produces: an OpenAI-style
    ``image_url.url`` data URI, and an Anthropic-style ``source.data`` base64
    payload. Anything else is skipped rather than guessed at.
    """
    import base64
    import os

    data_b64 = None
    if _block_type(block) == "image_url":
        url = (block.get("image_url") or {}).get("url", "")
        if isinstance(url, str) and "base64," in url:
            data_b64 = url.split("base64,", 1)[1]
    else:
        source = block.get("source") or {}
        if isinstance(source, dict):
            data_b64 = source.get("data")
    if not data_b64:
        return None
    try:
        raw = base64.b64decode(data_b64)
    except Exception as e:  # noqa: BLE001 - a bad payload is not worth killing a run
        log.warning("could not decode an image block: %s", e)
        return None
    path = os.path.join(directory, f"frame_{index}.png")
    with open(path, "wb") as fh:
        fh.write(raw)
    return path


class ClaudeCodeClient(LLMClient):
    """Drives the local ``claude`` CLI as a plain text-completion backend.

    Configure through the LLM config's extra mapping::

        provider: claude_code
        model: sonnet              # optional; passed to --model as-is
        timeout: 180               # seconds; a hung CLI must not wedge a sweep
        extra:
          cli_path: /usr/bin/claude   # default: whatever is on PATH
          on_image: error             # error (default) | warn
          retries: 1                  # extra attempts on TransientLLMError
          cwd: /tmp                   # where to spawn the CLI
          system_prompt: "..."        # replaces the CLI's agent system prompt
          extra_args: ["--effort", "low"]
    """

    def __init__(
        self,
        *,
        model: str = "",
        timeout: float = 180.0,
        extra: Optional[dict] = None,
        **_: Any,
    ) -> None:
        extra = dict(extra or {})
        self._model = model
        # CLI calls are slow (process spawn + agent turn), so a failed attempt
        # is expensive. Retry once by default rather than the 3 the HTTP
        # clients use -- a wedged sweep is worse than a lost episode.
        self._retries = int(extra.pop("retries", 1))
        self._on_image = str(extra.pop("on_image", "error")).lower()
        #: Spill image blocks to disk so the CLI can read them. Off by default:
        #: it grants the Read tool for those calls, which is a real capability
        #: increase and should be asked for rather than assumed.
        self._vision = bool(extra.pop("vision", False))
        # Pop both spellings unconditionally so neither lingers in _ignored_extra.
        cli_path, binary = extra.pop("cli_path", ""), extra.pop("binary", "")
        self._cli_path = str(cli_path or binary or "claude")
        # Default to a scratch dir, not the caller's cwd: the CLI auto-discovers
        # CLAUDE.md and project settings from wherever it starts, and dragging
        # this repo's instructions into every completion would silently change
        # the prompt under test.
        self._cwd = str(extra.pop("cwd", "") or tempfile.gettempdir())
        self._system_prompt = str(extra.pop("system_prompt", "") or "")
        self._extra_args = [str(a) for a in (extra.pop("extra_args", None) or [])]
        self._timeout = float(timeout or 180.0)
        # Anything left over is ignored on purpose: temperature/max_tokens have
        # no CLI equivalent, and silently pretending otherwise would be a lie.
        self._ignored_extra = extra

    @property
    def name(self) -> str:
        return f"claude-code({self._model or 'default'})"

    # ---- prompt assembly -------------------------------------------------

    def _flatten(self, messages: list[ChatMessage], *, image_dir=None) -> str:
        """Render the message list as one legibly-role-labelled transcript.

        With ``image_dir`` set, image blocks are written into it as PNGs and
        referenced by absolute path, which is how the CLI actually accepts
        pixels (verified: it reads the file with its Read tool and answers about
        the contents). Without it, images are refused -- see _render_content.
        """
        parts: list[str] = []
        for msg in messages:
            body = self._render_content(msg, image_dir=image_dir).strip()
            if not body:
                # A bare "Assistant:" label with nothing under it reads as a
                # cue to the model rather than as history; drop it.
                continue
            label = _ROLE_LABELS.get(msg.role, msg.role.capitalize() or "User")
            parts.append(f"{label}:\n{body}")
        return "\n\n".join(parts)

    def _render_content(self, msg: ChatMessage, *, image_dir=None) -> str:
        if isinstance(msg.content, str):
            return msg.content

        texts: list[str] = []
        images = 0
        written: list[str] = []
        for block in msg.content:
            if _block_type(block) == "text":
                texts.append(str(block.get("text", "")))
            elif _is_image_block(block):
                images += 1
                if image_dir is not None:
                    path = _spill_image(block, image_dir, len(written))
                    if path is not None:
                        written.append(path)

        if written:
            # The CLI reads pixels from a file, not from base64 in the prompt.
            # Naming the path explicitly (rather than relying on @-syntax) keeps
            # this working whether or not file-reference expansion is enabled.
            for path in written:
                texts.append(f"[image: {path}] -- read this file to see the "
                             f"current camera view.")
            return "\n".join(t for t in texts if t)

        if images:
            # Dropping pixels quietly is the exact bug class this harness has
            # been stamping out: a vision run that silently becomes a text run
            # still produces plausible-looking numbers.
            if image_dir is not None:
                # Vision was requested and the spill failed: the payload was not
                # decodable. Say that, rather than blaming the CLI for being
                # text-only when it is not.
                detail = (
                    f"{self.name}: vision is on, but none of the {images} image "
                    f"block(s) in a {msg.role!r} message could be decoded to a PNG "
                    f"(expected a base64 data URI or an Anthropic source.data "
                    f"payload). Refusing to continue text-only, because the run "
                    f"would look like a vision run."
                )
            else:
                detail = (
                    f"{self.name} cannot send images: the Claude Code CLI takes a text "
                    f"prompt only, so {images} image block(s) in a {msg.role!r} message "
                    f"have nowhere to go. Use provider 'claude' (the Messages API) for "
                    f"vision runs, set extra.vision=true to spill frames to disk for the "
                    f"CLI to read, or set extra.on_image='warn' to accept a text-only "
                    f"degradation."
                )
            if self._on_image == "warn":
                log.warning("%s (continuing text-only)", detail)
                texts.append(f"[{images} image(s) omitted: CLI backend is text-only]")
            else:
                raise LLMError(detail)

        return "\n".join(t for t in texts if t)

    # ---- invocation ------------------------------------------------------

    def _argv(self, *, tools: str = "") -> list[str]:
        argv = [
            self._cli_path,
            "--print",  # headless: one prompt in, one answer out, no session
            "--output-format",
            "json",  # structured result envelope we can parse
            "--tools",
            # "" disables every built-in tool. Vision needs exactly one exception:
            # with no tools the model cannot open the frame, and -- measured -- it
            # does not error, it emits a transcript of the Read call it wanted to
            # make and returns that as its answer. A silent wrong answer, so the
            # image path grants Read and nothing else.
            tools,
            "--disable-slash-commands",  # a prompt starting with '/' is text, not a skill
            "--strict-mcp-config",  # ignore the user's MCP servers
            "--no-session-persistence",  # don't litter ~/.claude with sweep sessions
        ]
        if self._model:
            argv += ["--model", self._model]
        if self._system_prompt:
            argv += ["--system-prompt", self._system_prompt]
        argv += self._extra_args
        return argv

    def _run(self, prompt: str, timeout: float, *, tools: str = "") -> dict[str, Any]:
        argv = self._argv(tools=tools)
        try:
            proc = subprocess.run(  # noqa: S603 - argv list, no shell
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self._cwd,
            )
        except FileNotFoundError as e:
            # No amount of retrying installs the CLI.
            hint = "" if shutil.which(self._cli_path) else " (not on PATH)"
            raise LLMError(f"claude CLI not found at {self._cli_path!r}{hint}: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise TransientLLMError(
                f"claude CLI timed out after {timeout:.0f}s; stderr: {_truncate(_as_text(e.stderr))}"
            ) from e

        stdout, stderr = _as_text(proc.stdout), _as_text(proc.stderr)
        data = _loads(stdout)

        if proc.returncode != 0:
            raise _classify(
                f"claude CLI exited {proc.returncode}",
                data=data,
                stderr=stderr,
                stdout=stdout,
            )

        if data is None:
            # Exit 0 but garbage on stdout: the invocation itself was well
            # formed, so a partial/interleaved write is plausibly a one-off.
            raise TransientLLMError(
                "claude CLI returned unparseable stdout (expected --output-format json); "
                f"stdout: {_truncate(stdout, 300)}; stderr: {_truncate(stderr, 300)}"
            )

        if data.get("is_error"):
            raise _classify(
                "claude CLI reported an error",
                data=data,
                stderr=stderr,
                stdout=stdout,
            )

        return data

    # ---- public API ------------------------------------------------------

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # Images go to disk for the CLI to read; the directory lives only as long
        # as the call, so a sweep does not accumulate frames.
        has_images = any(_message_has_image(m) for m in messages)
        if has_images and self._on_image == "error" and not self._vision:
            raise LLMError(
                f"{self.name} received image blocks but vision is off. Set "
                f"extra.vision=true to spill frames to disk and let the CLI read "
                f"them (this grants the Read tool for those calls), or use provider "
                f"'claude' for the Messages API path."
            )
        image_dir = None
        stack = contextlib.ExitStack()
        with stack:
            if has_images and self._vision:
                image_dir = stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="harness-frames-"))
            prompt = self._flatten(messages, image_dir=image_dir)
            if not prompt.strip():
                raise LLMError("refusing to invoke the claude CLI with an empty prompt")
            return self._complete_with(prompt, timeout,
                                       tools="Read" if image_dir else "")

    def _complete_with(self, prompt: str, timeout, tools: str) -> LLMResponse:
        if self._ignored_extra:
            log.debug("claude_code backend ignores unsupported knobs: %s",
                      sorted(self._ignored_extra))

        eff_timeout = float(timeout or self._timeout)
        data = with_retries(
            lambda: self._run(prompt, eff_timeout, tools=tools),
            retries=self._retries,
            exceptions=(TransientLLMError,),
        )

        content = data.get("result")
        if not isinstance(content, str):
            # A success envelope with no text is a parse failure, not an empty
            # completion -- returning "" here would look like a refusal.
            raise TransientLLMError(
                f"claude CLI response has no string 'result' field: {_truncate(json.dumps(data), 300)}"
            )

        return LLMResponse(
            content=content,
            model=_reported_model(data) or self._model,
            usage=_usage(data),
            raw=data,
            finish_reason=str(data.get("stop_reason") or ""),
        )


# ---- helpers -------------------------------------------------------------


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value if isinstance(value, str) else ""


def _loads(stdout: str) -> Optional[dict[str, Any]]:
    """Best-effort parse of the CLI's JSON envelope; None if it isn't one."""
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _classify(prefix: str, *, data: Optional[dict], stderr: str, stdout: str) -> LLMError:
    """Pick LLMError vs TransientLLMError, keeping stderr in the message.

    Default is non-retryable: the usual cause of a failed CLI invocation is a
    bad flag, a missing model, or dead auth, and re-spawning the process three
    times just burns wall-clock on a long sweep. Retry only on evidence.
    """
    status = (data or {}).get("api_error_status")
    result = (data or {}).get("result")
    haystack = " ".join(
        p for p in (stderr, stdout if data is None else "", result if isinstance(result, str) else "") if p
    ).lower()

    detail = f"{prefix}"
    if isinstance(status, int):
        detail += f" (api_error_status={status})"
    if isinstance(result, str) and result.strip():
        detail += f"; result: {_truncate(result, 300)}"
    detail += f"; stderr: {_truncate(stderr) or '<empty>'}"

    retryable = False
    if isinstance(status, int):
        retryable = status in _TRANSIENT_STATUS
    elif any(marker in haystack for marker in _TRANSIENT_MARKERS):
        retryable = True

    return TransientLLMError(detail) if retryable else LLMError(detail)


def _reported_model(data: dict[str, Any]) -> str:
    """Recover the model the CLI actually used.

    The JSON envelope has no single ``model`` field -- it has ``modelUsage``,
    keyed by every model the turn touched, which includes small helper models
    the CLI uses for side tasks. The main model is the one whose token counts
    match the top-level ``usage`` totals; if that is ambiguous, fall back to
    whichever model did the most work.
    """
    model_usage = data.get("modelUsage")
    if not isinstance(model_usage, dict) or not model_usage:
        return ""

    out = (data.get("usage") or {}).get("output_tokens")
    if isinstance(out, int):
        matches = [
            name
            for name, u in model_usage.items()
            if isinstance(u, dict) and u.get("outputTokens") == out
        ]
        if len(matches) == 1:
            return str(matches[0])

    def work(name: str) -> int:
        u = model_usage.get(name)
        if not isinstance(u, dict):
            return 0
        return int(u.get("outputTokens") or 0) + int(u.get("inputTokens") or 0)

    return str(max(model_usage, key=work))


def _usage(data: dict[str, Any]) -> dict[str, Any]:
    """Forward the CLI's token/cost numbers, or nothing at all."""
    raw = data.get("usage")
    usage: dict[str, Any] = {}
    if isinstance(raw, dict):
        for key in _USAGE_KEYS:
            val = raw.get(key)
            if isinstance(val, (int, float)):
                usage[key] = int(val)
    cost = data.get("total_cost_usd")
    if isinstance(cost, (int, float)):
        usage["total_cost_usd"] = float(cost)
    return usage
