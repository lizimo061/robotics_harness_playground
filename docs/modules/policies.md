# policies (harness/policies/) + policy-as-tool (harness/tools/policy_tool.py)
> Purpose: delegate low-level motor control to a trained policy the LLM calls as a tool.
> Read when: running RoboLab (or any continuous-control env) with an LLM planner
> on top of a VLA executor.
> Key files: policies/base.py, policies/remote.py, tools/policy_tool.py

## The idea

An LLM emitting JSON deltas step-by-step cannot do fine continuous manipulation.
A trained VLA (pi0.5, GR00T, ...) can, but it does not plan or verify well. So
split the work:

```
LLM (planner, tools mode) --run_policy("pick up the banana")--> Policy (executor)
                          <--steps, success, resulting state---
```

The LLM decomposes the task, delegates each sub-instruction to the policy, reads
back what happened, and retries or moves on. This is the hierarchical setup
RoboLab's server-client architecture is built for.

## Public API

- Policy (policies/base.py) - begin(instruction, action_space) / act(observation_text, image) -> vector / reset / close
- RemotePolicy(base_url, action_dim, env_id, timeout, retries) - HTTP client for a policy server
- ScriptedPolicy(actions | callable, action_dim) - offline policy for tests/demos
- get_policy(spec, llm=None) - factory; spec is a Policy or {"type": remote|llm|scripted, ...}
- RunPolicyTool(policy, default_steps, max_steps, use_vision, stop_on_success)
- get_policy_tools(policy, **opts) - policy-centric toolset (run_policy + perception + done)
- get_default_tools(policy=None, **opts) - standard toolset; prepends run_policy when given a policy

## The run_policy tool

```json
{"tool": "run_policy", "args": {"instruction": "pick up the banana", "steps": 60}}
```

It runs the policy closed-loop for up to `steps` env steps, stopping early on
success or termination, and returns steps run + success + the resulting state.

`RunPolicyTool.closed_loop = True`: unlike the single-action tools in builtin.py,
it drives the env itself (a policy needs many steps per sub-instruction). The
controller passes an `on_step` callback so those inner steps still land in the
recorded Episode, while the trace shows ONE entry per tool call (with a `steps`
count). Budgets are clamped to `max_steps`, and a policy exception is reported
back to the LLM as tool feedback rather than killing the episode.

## Enabling it

Python:

```python
LLMController(llm, mode="tools",                      # policy-as-tool needs tools mode
              policy={"type": "remote", "base_url": "http://localhost:8000", "action_dim": 8},
              policy_options={"default_steps": 60})
```

YAML (see configs/robolab_policy_tool.yaml):

```yaml
agent:
  mode: tools
  extra:
    policy: {type: remote, base_url: "http://localhost:8000", action_dim: 8}
    policy_options: {default_steps: 60, max_steps: 400}
```

RoboLab end-to-end:

```bash
# 1. start a policy server (pi0.5/GR00T behind the harness protocol, or the harness's own)
python examples/serve_robolab.py --port 8000
# 2. run the task with the LLM planning on top of it
python examples/run_robolab.py --task BananaInBowlTask --headless --policy-url http://localhost:8000
```

## Which toolset

With a policy, the controller defaults to `get_policy_tools`: run_policy +
perception + done. The hand-written ee/joint action tools are dropped on purpose
-- their 2D/3D coordinate model does not match a RoboLab action vector, and
mixing the two invites the LLM to fight the policy for control. Pass `tools=`
explicitly to override.

## Policy vs PolicyAgent

- `harness/policies/` = low-level executors the agent delegates TO (this doc).
- `harness/agent/policy.py` PolicyAgent = the LLM exposed AS a policy, for when
  a benchmark owns the env and asks the harness for one action per step
  (modules/policy.md). It satisfies the Policy protocol, so `{"type": "llm"}`
  makes the LLM its own executor -- a smoke-test path, not a real VLA.

## Related
- modules/tools.md, modules/policy.md (serving), modules/robolab.md, modules/agent.md.
