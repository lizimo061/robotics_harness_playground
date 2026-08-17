# tools (harness/tools/)
> Purpose: named, schema-described capabilities the LLM can call (tool use).
> Read when: adding a tool, or understanding the tools agent mode.
> Key files: base.py (Tool/ToolResult), builtin.py, registry.py

## Public API

- Tool (name, description, parameters, run(env, **args) -> ToolResult, to_openai_function, signature)
- ToolResult (feedback, action, done)
- get_default_tools() -> list[Tool]
- ToolRegistry(tools).get(name) / register / tools()
- parse_tool_call(text) -> (name, args)  (tolerates {"tool":..} / {"name":..} / fences)

## Built-in tools

- Action: move_to(x,y,z), move_delta(dx,dy,dz), grasp(), release(), set_joints(positions)
- Perception: list_objects, get_object_position(name), list_goals, list_obstacles,
  get_end_effector_position, is_grasped
- Control: done()

## How tools map to the env

Action tools return ToolResult.action (a low-level Action); the agent steps the
env with it. Perception tools return ToolResult.feedback (text) using the env's
object-aware query API. done() sets ToolResult.done.

## Extension points

- Subclass Tool, implement run(), add to get_default_tools().
- The parameters dict is a JSON Schema; it is also used for OpenAI function
  calling and for the prompt signature.

## Related
- guides/add-tool.md, modules/agent.md (tools mode), modules/envs.md (query API).
