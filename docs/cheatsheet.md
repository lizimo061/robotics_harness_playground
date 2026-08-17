# Cheatsheet

## Run a task

```bash
python -m harness.cli configs/toy_pick_place.yaml          # offline mock
python -m harness.cli configs/deepseek_pick_place.yaml     # real DeepSeek
python examples/visualize.py configs/... --console         # live trace
python examples/harder_tasks.py                            # tools demo
```

## Config (minimal)

```yaml
llm:   { provider: deepseek, model: deepseek-chat }
env:   { name: tabletop, task: stack, params: { difficulty: 0.7 } }
agent: { mode: tools, max_steps: 60 }
eval:  { episodes: 3 }
viz:   { backend: html, output: logs/viz.html }
```

## Programmatic

```python
from harness.config import LLMConfig, EnvConfig
from harness.llm import get_llm
from harness.envs import get_env
from harness.agent import LLMController

llm = get_llm(LLMConfig(provider='mock', extra={'script': [...]}))
env = get_env(EnvConfig(name='tabletop', task='stack'))
ep = LLMController(llm, mode='tools').run(env)
print(ep.success, ep.steps)
```

## Generate a task

```python
from harness.tasks import generate_task, available_tasks
spec = generate_task('pick_place_obstacle', seed=1, difficulty=0.8)
print(spec.description, spec.objects, spec.obstacles)
```

## Tools

```python
from harness.tools import get_default_tools, ToolRegistry, parse_tool_call
reg = ToolRegistry(get_default_tools())
name, args = parse_tool_call('{"tool": "grasp", "args": {}}')
result = reg.get(name).run(env, **args)   # ToolResult(feedback, action, done)
```

## Public API surface

harness: run_eval, load_config, HarnessConfig, Obs, Action, StepResult, Episode.
harness.llm: get_llm, ChatMessage, LLMClient, LLMResponse.
harness.envs: get_env, Env.
harness.agent: LLMController, Agent.
harness.tasks: generate_task, available_tasks, TaskSpec.
harness.tools: get_default_tools, ToolRegistry, parse_tool_call, Tool.
harness.viz: TraceRecorder, save_html, ConsoleTracer, MatplotlibViewer.
