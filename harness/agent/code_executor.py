"""Restricted execution of LLM-generated Python (Code-as-Policies).

SECURITY NOTE: executing model-generated code is inherently unsafe. This module
runs the snippet in-process with a strict builtins allowlist, an AST import
blocklist, and a signal-based timeout. It is a best-effort guardrail, not a hard
security boundary: do not use "code" mode with untrusted model output in a
privileged environment. Prefer "json" mode for untrusted models.
"""
from __future__ import annotations

import ast
import contextlib
import io
import signal
import traceback
from typing import Any

_ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
    "list", "max", "min", "range", "round", "set", "str", "sum", "tuple", "zip",
    "print", "True", "False", "None",
}

_BLOCKED_ROOTS = {
    "os", "sys", "subprocess", "importlib", "shutil", "socket", "pathlib",
    "builtins", "io", "open", "eval", "exec", "compile", "__import__",
    "globals", "locals",
}


def _check_ast(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _BLOCKED_ROOTS:
                    raise ValueError(f"import '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _BLOCKED_ROOTS:
                raise ValueError(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _BLOCKED_ROOTS:
                raise ValueError(f"call to '{f.id}' is not allowed")


class _Timeout(Exception):
    pass


def _handler(signum, frame):
    raise _Timeout("code execution timed out")


def execute_code(code: str, namespace: dict[str, Any], timeout: float = 10.0) -> dict:
    """Execute a snippet with restricted builtins and a timeout.

    Returns {"ok": bool, "error": str, "output": str}.
    """
    try:
        _check_ast(code)
    except (SyntaxError, ValueError) as e:
        return {"ok": False, "error": str(e), "output": ""}

    import builtins

    safe_builtins = {k: getattr(builtins, k) for k in _ALLOWED_BUILTINS}
    ns = dict(namespace)
    ns["__builtins__"] = safe_builtins

    buf = io.StringIO()
    use_alarm = hasattr(signal, "SIGALRM")
    old_handler = None
    try:
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(max(1, int(timeout)))
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<llm>", "exec"), ns)
        except _Timeout:
            return {"ok": False, "error": f"timed out after {timeout}s", "output": buf.getvalue()}
        except Exception:
            return {"ok": False, "error": traceback.format_exc(), "output": buf.getvalue()}
        return {"ok": True, "error": "", "output": buf.getvalue()}
    finally:
        if use_alarm and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
