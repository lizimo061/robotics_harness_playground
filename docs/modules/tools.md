# tools (harness/tools/)
> Purpose: named, schema-described capabilities the LLM can call (tool use).
> Read when: adding a tool, or understanding the tools agent mode.
> Key files: base.py (Tool/ToolResult), builtin.py, registry.py

## Public API

- Tool (name, description, parameters, closed_loop, run(env, **args) -> ToolResult, to_openai_function, signature)
- ToolResult (feedback, action, done, steps, success)
- get_default_tools(policy=None, **opts) -> list[Tool]
- get_policy_tools(policy, **opts) -> list[Tool]  (see modules/policies.md)
- ToolRegistry(tools).get(name) / register / tools()
- parse_tool_call(text) -> (name, args)  (tolerates {"tool":..} / {"name":..} / fences)

## Built-in tools

- Action: move_to(x,y,z), move_delta(dx,dy,dz), grasp(), release(), set_joints(positions)
- Perception: list_objects, get_object_position(name), list_goals, list_obstacles,
  get_end_effector_position, is_grasped
- Control: done()
- Delegation: run_policy(instruction, steps) - only when a policy is configured
  (modules/policies.md)

## How tools map to the env

Action tools return ToolResult.action (a low-level Action); the agent steps the
env with it. Perception tools return ToolResult.feedback (text) using the env's
object-aware query API. done() sets ToolResult.done.

Closed-loop tools (`closed_loop = True`, e.g. run_policy) instead drive the env
themselves for many steps and report `steps`/`success`; the agent hands them an
`on_step` callback so the inner steps still land in the Episode. A tool that
raises is reported back to the LLM as feedback rather than ending the episode.

## Extension points

- Subclass Tool, implement run(), add to get_default_tools().
- The parameters dict is a JSON Schema; it is also used for OpenAI function
  calling and for the prompt signature.

## Related
- guides/add-tool.md, modules/agent.md (tools mode), modules/envs.md (query API),
  modules/policies.md (policy-as-tool).
