"""Serving: expose a PolicyAgent over HTTP for policy-server architectures.

Policy benchmarks (RoboLab, RoboDojo / XPolicyLab) run the policy as a
standalone server that receives observations and returns actions. This module
wraps harness.agent.policy.PolicyAgent behind a tiny stdlib HTTP server with
per-env sessions.

    manager = PolicySessionManager(llm, action_dim=8)
    serve(manager, port=8000)   # blocks

Endpoints:
    GET  /health -> {"ok": true}
    POST /begin  -> {"instruction": str, "env_id": int, "action_space": {...}?}
    POST /act    -> {"instruction": str, "observation_text": str,
                     "image_b64": str?, "env_id": int}
                 -> {"action": [float, ...]}
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np

from harness.agent.policy import PolicyAgent
from harness.types import ActionSpace


class PolicySessionManager:
    """Holds one PolicyAgent per env_id and auto-begins on instruction change."""

    def __init__(self, llm, *, action_space: Optional[ActionSpace] = None, action_dim: int = 8, temperature: Optional[float] = None) -> None:
        self._llm = llm
        self._action_space = action_space
        self._action_dim = action_dim
        self._temperature = temperature
        self._sessions: dict = {}
        self._instructions: dict = {}

    def begin(self, env_id: int, instruction: str, action_space: Optional[ActionSpace] = None) -> None:
        agent = self._sessions.get(env_id)
        if agent is None:
            agent = PolicyAgent(
                self._llm,
                action_space=action_space or self._action_space,
                action_dim=self._action_dim,
                temperature=self._temperature,
            )
            self._sessions[env_id] = agent
        agent.begin(instruction, action_space=action_space or self._action_space)
        self._instructions[env_id] = instruction

    def act(self, env_id: int, instruction: str, observation_text: str, image: Optional[np.ndarray] = None) -> np.ndarray:
        if env_id not in self._sessions or self._instructions.get(env_id) != instruction:
            self.begin(env_id, instruction)
        return self._sessions[env_id].act(observation_text, image=image)


def make_handler(manager: PolicySessionManager):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, code=404)

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                path = self.path.rstrip("/")
                if path == "/begin":
                    env_id = int(body.get("env_id", 0))
                    instruction = str(body.get("instruction", ""))
                    manager.begin(env_id, instruction)
                    self._json({"ok": True})
                elif path == "/act":
                    env_id = int(body.get("env_id", 0))
                    instruction = str(body.get("instruction", ""))
                    obs_text = str(body.get("observation_text", ""))
                    image = self._decode_image(body.get("image_b64"))
                    vec = manager.act(env_id, instruction, obs_text, image=image)
                    self._json({"action": np.asarray(vec, dtype=np.float32).tolist()})
                else:
                    self._json({"error": "not found"}, code=404)
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, code=500)

        def _json(self, obj, code: int = 200) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        @staticmethod
        def _decode_image(b64):
            if not b64:
                return None
            import base64
            import io

            from PIL import Image

            return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))

        def log_message(self, *args) -> None:  # quiet the default stderr logging
            pass

    return Handler


def serve(manager: PolicySessionManager, port: int = 8000, host: str = "0.0.0.0") -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(manager))
    print(f"harness policy server listening on {host}:{port}")  # noqa: T201
    httpd.serve_forever()
