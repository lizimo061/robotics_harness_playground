# skills + planning (harness/skills/)
> Purpose: subgoal-level skills and the plan-then-execute long-horizon loop.
> Read when: adding a skill, or understanding how NL tasks become actions.
> Key files: base.py, builtin.py, registry.py, executor.py, planning.py

## Public API

- Skill (satisfied(env) / plan_action(env) / parameters / tolerance)
- SkillResult (success, feedback, steps)
- make_skill(name, **args); skill_catalog() -> [{name, description, signature, parameters}]
- run_skill(env, skill, budget, on_step) -> SkillResult
- build_plan_prompt(task, scene, catalog) -> [ChatMessage]; parse_plan(text) -> [{"skill","args"}]

## Built-in skills

pick(object), place(object, target), put_in(object, container), open(container),
press(button). Each is a closed-loop state machine over primitive Actions
(ee_pose + gripper) using the env's query API.

## The agent mode

agent.mode = "skills": (1) plan - LLM returns an ordered skill list (falls back
to the TaskSpec.steps gold plan); (2) act - run each skill with subgoal
verification, retrying once on failure. See guides/long-horizon.md.

## Extension points

- Add a skill: subclass Skill, implement satisfied + plan_action, register in
  registry.py _SKILL_CLASSES.
- Add replanning: in llm_controller._run_skills_mode, ask the LLM to revise the
  remaining plan on a failed skill.

## Related
- guides/long-horizon.md, modules/tasks.md (steps), modules/envs.md (query API).
