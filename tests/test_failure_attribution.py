"""P4: whose fault was it -- the world's or the planner's?

VoLoAgent's diagnostics separate world failures (wrong-object pick, stuck) from VLM
failures (planning error, missed failure detection, tool-use mismatch). The existing
taxonomy answered a different question -- "is this number trustworthy?" -- and left
both of these as `task_failed`.

The rule these tests enforce is that attribution needs *evidence*. An unattributed
failure is honest; a guessed one puts fabricated blame into a report. So `task_failed`
and `agent_timeout` stay unattributed on purpose, and each attributed mode requires
something positive in the trace.
"""
import unittest

from harness.eval.results import FailureMode, classify_failure
from harness.types import Action, Episode, Obs


def _obs(text):
    return Obs(text=text)


class TestBlameBuckets(unittest.TestCase):
    def test_world_failures(self):
        for m in (FailureMode.WRONG_OBJECT, FailureMode.STUCK):
            self.assertEqual(FailureMode.blame(m), "world", m)

    def test_planner_failures(self):
        for m in (FailureMode.PLANNING_ERROR, FailureMode.MISSED_FAILURE,
                  FailureMode.TOOL_MISMATCH, FailureMode.NO_PROGRESS):
            self.assertEqual(FailureMode.blame(m), "planner", m)

    def test_harness_failures_keep_their_own_bucket(self):
        for m in FailureMode.NOT_MODEL_FAULT:
            self.assertEqual(FailureMode.blame(m), "harness", m)

    def test_success_is_not_a_failure(self):
        self.assertEqual(FailureMode.blame(FailureMode.NONE), "none")
        self.assertEqual(FailureMode.blame(None), "none")

    def test_a_plain_task_failure_stays_unattributed(self):
        """It says the task was not solved, not whose fault that was."""
        self.assertEqual(FailureMode.blame(FailureMode.TASK_FAILED), "unattributed")
        self.assertEqual(FailureMode.blame(FailureMode.AGENT_TIMEOUT), "unattributed")

    def test_the_world_and_planner_sets_do_not_overlap(self):
        self.assertFalse(set(FailureMode.WORLD_FAILURE) & set(FailureMode.PLANNER_FAILURE))


class TestPlanningError(unittest.TestCase):
    def test_declaring_done_on_an_unsolved_task_is_a_planning_error(self):
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.2]),
                              Action(kind="stop", comment="done")],
                     metadata={"llm_calls": 3, "max_steps": 30})
        ep.success = False
        self.assertEqual(classify_failure(ep), FailureMode.PLANNING_ERROR)

    def test_declaring_done_on_a_solved_task_is_not_a_failure_at_all(self):
        ep = Episode(actions=[Action(kind="stop")], infos=[{"success": True}])
        ep.success = True
        self.assertEqual(classify_failure(ep), FailureMode.NONE)


class TestMissedFailureDetection(unittest.TestCase):
    """Only measurable because P2 created a moment where the agent chooses."""

    def test_continuing_past_observed_divergence_is_the_planner_s_fault(self):
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.1])],
                     metadata={"monitor_continues_after_divergence": 2,
                               "llm_calls": 5, "max_steps": 30})
        self.assertEqual(classify_failure(ep), FailureMode.MISSED_FAILURE)

    def test_without_a_monitor_the_mode_cannot_fire(self):
        """No monitor, no moment at which the agent could have noticed."""
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.1])],
                     metadata={"llm_calls": 5, "max_steps": 30})
        self.assertNotEqual(classify_failure(ep), FailureMode.MISSED_FAILURE)


class TestWrongObject(unittest.TestCase):
    def test_touching_only_unnamed_objects_is_a_world_failure(self):
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.1])],
                     metadata={"objects_touched": ["banana"],
                               "objects_in_instruction": ["rubiks_cube"],
                               "llm_calls": 4, "max_steps": 30})
        self.assertEqual(classify_failure(ep), FailureMode.WRONG_OBJECT)

    def test_touching_the_right_object_is_not_flagged(self):
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.1])],
                     metadata={"objects_touched": ["rubiks_cube", "table"],
                               "objects_in_instruction": ["rubiks_cube"],
                               "llm_calls": 4, "max_steps": 30})
        self.assertNotEqual(classify_failure(ep), FailureMode.WRONG_OBJECT)

    def test_missing_evidence_means_no_attribution(self):
        """Without both lists there is nothing to compare, so do not guess."""
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.1])],
                     metadata={"objects_touched": ["banana"],
                               "llm_calls": 4, "max_steps": 30})
        self.assertNotEqual(classify_failure(ep), FailureMode.WRONG_OBJECT)


class TestStuck(unittest.TestCase):
    def test_a_sustained_run_of_identical_observations_is_stuck(self):
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.1])] * 10,
                     observations=[_obs("ee at (0.1, 0.1)")] * 10,
                     metadata={"llm_calls": 10, "max_steps": 30})
        self.assertEqual(classify_failure(ep), FailureMode.STUCK)

    def test_a_brief_pause_is_not_stuck(self):
        """A gripper closing or a controller settling must not trip this."""
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.1])] * 6,
                     observations=[_obs("a"), _obs("a"), _obs("a"),
                                   _obs("b"), _obs("c"), _obs("d")],
                     metadata={"llm_calls": 6, "max_steps": 30})
        self.assertNotEqual(classify_failure(ep), FailureMode.STUCK)

    def test_a_short_episode_cannot_be_stuck(self):
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.1])] * 3,
                     observations=[_obs("same")] * 3,
                     metadata={"llm_calls": 3, "max_steps": 30})
        self.assertNotEqual(classify_failure(ep), FailureMode.STUCK)

    def test_moving_observations_are_never_stuck(self):
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.1])] * 10,
                     observations=[_obs(f"ee at {i}") for i in range(10)],
                     metadata={"llm_calls": 10, "max_steps": 30})
        self.assertNotEqual(classify_failure(ep), FailureMode.STUCK)


class TestPriorityAgainstBudgetModes(unittest.TestCase):
    """Attribution is checked before the budget verdict, on purpose.

    'Ran out of turns' says when an episode stopped, not what was wrong with it -- so a
    trace that shows a wrong-object pick should report that even if the budget also
    ran out.
    """

    def test_attribution_beats_agent_timeout(self):
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.1])],
                     metadata={"objects_touched": ["banana"],
                               "objects_in_instruction": ["cube"],
                               "llm_calls": 30, "max_steps": 30})
        self.assertEqual(classify_failure(ep), FailureMode.WRONG_OBJECT)

    def test_harness_faults_still_win_over_everything(self):
        """A provider outage is not a statement about the agent's reasoning."""
        class TransientLLMError(Exception):
            pass

        ep = Episode(actions=[Action(kind="stop")],
                     metadata={"llm_calls": 30, "max_steps": 30})
        self.assertEqual(classify_failure(ep, error=TransientLLMError()),
                         FailureMode.PROVIDER_ERROR)

    def test_an_unattributable_failure_is_still_reported_as_one(self):
        ep = Episode(actions=[Action(kind="ee_pose", value=[0.1, 0.1])],
                     observations=[_obs(f"s{i}") for i in range(4)],
                     metadata={"llm_calls": 30, "max_steps": 30})
        self.assertEqual(classify_failure(ep), FailureMode.AGENT_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
