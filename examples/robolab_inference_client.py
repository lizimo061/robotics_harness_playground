"""RoboLab inference client that talks to the harness policy server.

Drop this into your RoboLab policy repo (or import it). It subclasses
robolab.eval.InferenceClient and implements the 4 required hooks to call the
server started by examples/serve_robolab.py.

Usage in your policy repo's run script::

    from my_policy.inference_client import HarnessPolicyClient
    client = HarnessPolicyClient(remote_host="localhost", remote_port=8000)

Reference: RoboLab docs/policy.md
"""
from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import requests

from robolab.eval import InferenceClient  # only importable inside the RoboLab venv


class HarnessPolicyClient(InferenceClient):
    open_loop_horizon = 1  # the LLM re-queries every step

    def __init__(self, remote_host: str = "localhost", remote_port: int = 8000, action_dim: int = 8) -> None:
        super().__init__()
        self.host = remote_host
        self.port = remote_port
        self.action_dim = action_dim

    # -- required hooks --------------------------------------------------- #
    def _extract_observation(self, raw_obs, *, env_id: int = 0) -> dict:
        # default DROID registration: image_obs + proprio_obs. TODO(verify)
        img = raw_obs.get("image_obs", {})
        keys = list(img.keys())
        image = img[keys[0]][env_id].cpu().numpy() if keys else None
        proprio = raw_obs.get("proprio_obs", {})
        joint = proprio.get("arm_joint_pos")
        gripper = proprio.get("gripper_pos")
        return {
            "image": image,
            "joint_pos": joint[env_id].cpu().numpy() if joint is not None else None,
            "gripper_pos": gripper[env_id].cpu().numpy() if gripper is not None else None,
        }

    def _pack_request(self, extracted_obs, instruction):
        req = {"instruction": instruction, "observation_text": self._obs_to_text(extracted_obs, instruction)}
        if extracted_obs.get("image") is not None:
            req["image_b64"] = self._encode(extracted_obs["image"])
        return req

    def _query_server(self, request):
        return requests.post(f"http://{self.host}:{self.port}/act", json=request, timeout=120).json()

    def _unpack_response(self, response) -> np.ndarray:
        # must return (horizon, action_dim); horizon is 1
        return np.asarray(response["action"], dtype=np.float32).reshape(1, self.action_dim)

    # -- helpers ---------------------------------------------------------- #
    def _obs_to_text(self, o, instruction: str) -> str:
        parts = [f"Instruction: {instruction}"]
        if o.get("joint_pos") is not None:
            parts.append("joint_pos=" + np.array2string(np.asarray(o["joint_pos"]), precision=3))
        if o.get("gripper_pos") is not None:
            parts.append("gripper=" + np.array2string(np.asarray(o["gripper_pos"]), precision=3))
        return "\n".join(parts)

    @staticmethod
    def _encode(img) -> str:
        from PIL import Image

        buf = BytesIO()
        Image.fromarray(np.asarray(img).astype(np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
