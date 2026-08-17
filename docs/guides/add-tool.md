# Add a tool

1. Subclass Tool (harness/tools/base.py):

       class MyTool(Tool):
           name = "my_tool"
           description = "What it does, in one sentence."
           parameters = {"type": "object", "properties": {...}, "required": [...]}
           def run(self, env, **args):
               return ToolResult(feedback="...", action=None, done=False)

2. Add it to get_default_tools() in harness/tools/registry.py.

## Contract

- Action tools: return ToolResult.action (an Action) so the agent steps the env.
- Perception tools: return ToolResult.feedback (text) read from the env's
  object-aware query API.
- Control tools: set ToolResult.done=True (e.g. done()).
- parameters is a JSON Schema; it also feeds OpenAI function calling and the prompt.

## Test

Add a case to tests/test_tools.py: parse a call, run the tool against a
TabletopEnv, and assert the feedback/action.
