# policy + serving (harness/agent/policy.py, harness/serving.py)
> Purpose: a per-step, server-friendly agent + HTTP serving for policy-server benchmarks.
> Read when: connecting the harness as a policy to RoboLab / RoboDojo / XPolicyLab.

## Public API

- PolicyAgent(llm, action_space, action_dim, gripper_last)
  - begin(instruction, action_space) -> reset the conversation
  - act(observation_text, image) -> action vector (np.ndarray)
- PolicySessionManager(llm, action_dim) - one PolicyAgent per env_id, auto-begins
  on instruction change
- make_handler(manager) -> BaseHTTPRequestHandler; serve(manager, port) -> blocks

## Why a separate agent

LLMController.run() owns the env and drives a whole episode. Policy servers invert
that: the simulator owns the env and asks for ONE action per step. PolicyAgent is
the direct action-emission path (json mode) that survives server round-trips;
tools/skills/code modes need in-process env access and stay in LLMController.

## HTTP protocol

- GET /health -> {"ok": true}
- POST /begin -> {"instruction", "env_id"} -> begin a session
- POST /act -> {"instruction", "observation_text", "image_b64"?, "env_id"}
  -> {"action": [float, ...]}

## End-to-end (RoboLab)

1. python examples/serve_robolab.py --port 8000   (DEEPSEEK_API_KEY set)
2. In your RoboLab policy repo, subclass InferenceClient with examples/
   robolab_inference_client.py, pointing at localhost:8000.

## Related
- modules/agent.md, modules/robolab.md, examples/serve_robolab.py.
