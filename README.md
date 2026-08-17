# Robotics Harness

A modular Python framework for controlling **simulated robots** with **large language models**.
It wires an LLM (DeepSeek, Kimi, Claude, OpenAI, or any OpenAI-compatible endpoint) to a simulation
backend (MuJoCo, Genesis, Gymnasium, robosuite, or a dependency-free toy environment) through a single
control loop, and evaluates whether the agent finishes the task.

## Highlights

- **Pluggable LLMs** - one OpenAI-compatible client covers DeepSeek, Kimi (Moonshot), OpenAI, vLLM and
  Ollama; a dedicated client covers Claude (Anthropic); a `mock` provider runs fully offline for tests/demos.
- **Pluggable environments** - toy tabletop (zero extra deps), Gymnasium (MuJoCo manipulation via
  gymnasium-robotics), Genesis (genesis-world), and robosuite all map onto one `Env` interface.
- **Multiple control strategies** - `json` (closed-loop ReAct), `code` (Code-as-Policies), `plan` (plan-then-act).
- **Typed configuration** - YAML/JSON drives everything; swapping a provider/env/agent is a one-line change.
- **Evaluation & logging** - success rate, reward, step counts, and JSONL trajectory logging.
- **Harder tasks & tools** - procedural task generators (stack, sort, obstacle avoidance, ...)
  and a tool-use agent mode that calls named tools (grasp, move_to, list_objects, ...).
- **Modular docs** - a docs/ tree (index + per-module docs + extension guides) designed
  for context-efficient navigation as the repo grows.

## Directory layout

```text
robotics_harness/
├── harness/
│   ├── cli.py                # `python -m harness.cli <config>` entry point
│   ├── config.py             # typed config (YAML/JSON -> dataclasses)
│   ├── runner.py             # config -> env + llm + agent -> evaluation
│   ├── types.py              # Obs, Action, StepResult, Episode, spaces
│   ├── agent/                # the LLM control loop + parsing + skills + sandbox
│   ├── envs/                 # base + toy + gymnasium + genesis + robosuite adapters
│   ├── llm/                  # openai-compat, anthropic, mock clients + registry
│   ├── robot/                # robot specs (Franka, UR5e) + action helpers
│   ├── perception/           # image -> base64 for vision-capable models
│   ├── tasks/                # TaskSpec + procedural harder-task generators
│   ├── tools/                # named tools for the tool-use agent mode
│   ├── viz/                  # trace recorder + html/live viewers
│   ├── eval/                 # metrics + JSONL trajectory logger
│   └── utils/                # logging + seeds
├── configs/                  # example YAML configs (toy/deepseek/kimi/mujoco/genesis)
├── examples/                 # quickstart.py, run_task.py, harder_tasks.py
├── docs/                     # modular docs (index + modules + guides)
└── tests/                    # unittest suite (49 tests)
```

## Install

```bash
# core (numpy, httpx, pyyaml, python-dotenv, pillow)
pip install -e .

# optional backends
pip install mujoco gymnasium gymnasium-robotics   # MuJoCo manipulation
pip install genesis-world                          # Genesis
pip install robosuite torch                        # robosuite
```

## Quickstart (no API key, no simulator)

```bash
python examples/quickstart.py
# success=True  steps=5  reward=-0.666
# actions: ['ee_delta', 'ee_delta', 'noop', 'ee_delta', 'ee_delta']
```

Or run an offline scripted agent through the config path:

```bash
python -m harness.cli configs/toy_pick_place.yaml
```

## Use a real LLM (DeepSeek)

```bash
export DEEPSEEK_API_KEY=sk-...
python -m harness.cli configs/deepseek_pick_place.yaml
```

Any OpenAI-compatible provider works by changing `provider` (and optionally `base_url`/`model`):
`deepseek`, `kimi`, `openai`, `claude` (Anthropic), `ollama`, `vllm`, or `custom`.

## Configuration

Every run is described by one YAML file:

```yaml
llm:
  provider: deepseek          # deepseek | kimi | openai | claude | ollama | vllm | custom | mock
  model: deepseek-chat
  api_key_env: DEEPSEEK_API_KEY
  base_url: https://api.deepseek.com
  temperature: 0.2

env:
  name: toy_tabletop          # toy_tabletop | gymnasium:<id> | genesis:<task> | robosuite:<task>
  task: pick_and_place
  max_episode_steps: 100

agent:
  mode: json                  # json | code | plan
  max_steps: 40
  use_vision: false

eval:
  episodes: 3
  log_dir: logs
  save_trajectories: true
```

## Environments

| `env.name` | Backend | Requires | Notes |
| --- | --- | --- | --- |
| `toy_tabletop` | numpy | nothing | 2D pick-and-place; great for fast iteration |
| `gymnasium:<id>` | Gymnasium | gymnasium (+ gymnasium-robotics for MuJoCo) | e.g. `gymnasium:FetchPickAndPlace-v3` |
| `genesis:<task>` | Genesis | genesis-world | Franka reach; `params.control_mode` = `ee_delta` or `joint_position` |
| `robolab:<task>` | Isaac Lab | RoboLab (Linux + CUDA) | 120+ manipulation tasks; in-process adapter (see examples/run_robolab.py) |
| `robosuite:<task>` | robosuite | robosuite + torch | e.g. `robosuite:Lift` |

## Agent modes

- **json** (default) - each step, the LLM sees the state as text and returns one JSON action
  (`move`, `move_to`, `gripper`, `joints`, `stop`). The parser is tolerant of markdown fences and prose.
- **code** (Code-as-Policies) - the LLM writes a short Python snippet that calls a skill library
  (`move_delta`, `grasp`, `release`, `state`, `done`, ...). Executed in a restricted sandbox.
- **plan** - the LLM first writes a plan, then executes it action-by-action.

## Visualization

Watch the robot animation replay **in sync** with the LLM trace (what the model
observed, what it decided, the action it chose, and the reward):

    python examples/visualize.py                           # offline toy demo
    python examples/visualize.py configs/deepseek_pick_place.yaml

This writes a self-contained **logs/viz.html** - open it in a browser to see the
animation on the left and the step-by-step LLM trace on the right. Click any
step to jump to it; use play/pause, speed, and the scrubber to step through.

Live options (via the **viz** config section):

| viz.backend | behaviour |
| --- | --- |
| html (default) | write a replay file (viz.output) you open in a browser |
| console | print each step (state + LLM response + action) live in the terminal |
| live | open a matplotlib window showing frames as the run happens |
| none | disable |

    viz:
      enabled: true
      backend: html        # html | console | live | none
      output: logs/viz.html
      fps: 8
      capture_frames: true

For MuJoCo/Gymnasium and robosuite you can also open the native GUI: pass
params.render_mode = "human" for gymnasium, or params.show_viewer = true for
Genesis (see harness/envs/genesis.py). Genesis also exports video directly:

    python examples/genesis_demo.py --task pick_place --scripted --video demo.mp4

(Use --show-viewer for the live GUI, --viz run.html for the trace replay.)

## Harder tasks & tools

For harder robot-arm problems (Franka-style), the harness adds a declarative
task framework and a tool-use agent:

- **harness/tasks/** - TaskSpec data + procedural generators. Difficulty (0..1)
  scales distances, obstacle sizes, and object count. Built-ins: pick_place,
  pick_place_obstacle, push, stack, sort, reach_avoid.
- **harness/tools/** - named, schema-described tools (grasp, move_to, list_objects,
  ...). agent.mode: tools lets the LLM call tools instead of emitting raw actions.
- **harness/envs/tabletop.py** - a multi-object env that runs every harder task.
- **harness/envs/genesis.py** - GenesisFrankaEnv: the same 3D TaskSpecs drive a
  Franka Panda in Genesis (IK control + kinematic grasp + camera). See
  docs/modules/genesis.md and examples/genesis_demo.py. (With genesis-world 0.2.x,
  pin libigl==2.5.1 - see docs/modules/genesis.md.)
- **Long-horizon** - agent.mode: skills (a planner decomposes a task like "put
  the bread into the oven, then press the button" into skills, then runs each
  skill closed-loop with subgoal verification). See docs/guides/long-horizon.md
  and examples/long_horizon.py.

    python examples/harder_tasks.py                  # offline demo (mock LLM + tools)
    python examples/harder_tasks.py --task sort      # real DeepSeek, generated task

    env:   { name: tabletop, task: stack, params: { difficulty: 0.7 } }
    agent: { mode: tools }

## Documentation

Modular docs live in **docs/** - start at **docs/README.md** for a navigation
map, then read only the module doc or guide relevant to your task. Each doc is
self-contained with a Purpose / Location / Read-when header, so context stays
small as the repo grows.

## Extending

**New LLM provider**: if it speaks OpenAI's `/chat/completions`, just add a default to
`harness/llm/registry.py::_PROVIDER_DEFAULTS`; otherwise subclass `LLMClient`.

**New environment**: subclass `harness.envs.base.Env`, then register it in
`harness/envs/registry.py::get_env`.

**New robot**: add a `RobotSpec` to `harness/robot/specs.py`.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Security note

`code` mode executes model-generated Python. It uses a builtins allowlist, an AST import blocklist,
and a timeout, but it is **not** a hard security boundary - prefer `json` mode for untrusted models.

## Related work

- VoxPoser (LLM + VLM generated value maps), Code-as-Policies (LLM writes robot code), Eureka (LLM writes
  reward functions).
- robosuite (modular MuJoCo manipulation), Genesis (genesis-world), Gymnasium/Gymnasium-Robotics, LeRobot.

## License

MIT.
