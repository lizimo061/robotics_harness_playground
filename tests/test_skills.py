import unittest

from harness.agent import LLMController
from harness.config import LLMConfig
from harness.envs.kitchen import KitchenEnv
from harness.llm import get_llm
from harness.skills import make_skill, run_skill, skill_catalog
from harness.skills.planning import parse_plan
from harness.tasks import generate_task


class TestParsePlan(unittest.TestCase):
    def test_plan_list(self):
        plan = parse_plan(
            '{"plan": [{"skill": "open", "args": {"container": "oven"}}, '
            '{"skill": "press", "args": {"button": "button"}}]}'
        )
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0], {"skill": "open", "args": {"container": "oven"}})

    def test_garbage(self):
        self.assertEqual(parse_plan("not json"), [])


class TestSkillsSolve(unittest.TestCase):
    def _env(self):
        return KitchenEnv(task_spec=generate_task("cook_bread", seed=1))

    def test_catalog(self):
        names = {c["name"] for c in skill_catalog()}
        self.assertIn("put_in", names)
        self.assertIn("press", names)

    def test_open_skill(self):
        env = self._env()
        env.reset()
        res = run_skill(env, make_skill("open", container="oven"), budget=20)
        self.assertTrue(res.success)
        self.assertTrue(env.is_container_open("oven"))

    def test_put_in_skill(self):
        env = self._env()
        env.reset()
        res = run_skill(env, make_skill("put_in", object="bread", container="oven"), budget=80)
        self.assertTrue(res.success)
        self.assertTrue(env.check_subgoal("bread_in_oven"))

    def test_press_skill(self):
        env = self._env()
        env.reset()
        res = run_skill(env, make_skill("press", button="button"), budget=20)
        self.assertTrue(res.success)
        self.assertTrue(env.is_button_pressed("button"))


class TestSkillsMode(unittest.TestCase):
    def test_skills_mode_solves_cook_bread(self):
        # mock LLM returns non-JSON for the planning call -> falls back to the
        # task spec's gold plan, then skills execute deterministically.
        env = KitchenEnv(task_spec=generate_task("cook_bread", seed=1))
        llm = get_llm(LLMConfig(provider="mock", extra={"fallback": "not json"}))
        ctrl = LLMController(llm, mode="skills", max_steps=200)
        ep = ctrl.run(env)
        self.assertTrue(ep.success)
        self.assertEqual(ep.metadata["mode"], "skills")
        # the gold plan was used
        self.assertEqual([s["skill"] for s in ep.metadata["plan"]], ["open", "put_in", "press"])


if __name__ == "__main__":
    unittest.main()
