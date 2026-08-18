"""P5: task packs, and refusing to mislabel a run.

RoboVoLo adds 126 tasks over four reasoning suites (common sense, memory, complex
references, world knowledge) on top of RoboLab's benchmark set. It installs separately,
and RoboLab's task discovery **skips a missing folder without comment**.

That silence is fine for RoboLab and wrong for an evaluation: a sweep that asked for
RoboVoLo and quietly measured the base 120 tasks would publish base-set numbers under
the pack's name. So a missing pack raises by default, and falling back has to be asked
for explicitly.

These tests run without Isaac by exercising the resolver directly -- which is the whole
of the logic worth testing here.
"""
import unittest
from unittest import mock

import harness.envs.robolab as rl


def _robolab_importable() -> bool:
    try:
        import robolab  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


class TestPackResolution(unittest.TestCase):
    def test_the_base_pack_is_the_benchmark_folder(self):
        with mock.patch.object(rl.RoboLabEnv, "_task_dir_exists", return_value=True):
            self.assertEqual(rl.RoboLabEnv._resolve_task_dirs("benchmark"), ["benchmark"])

    def test_robovolo_adds_its_folder_on_top(self):
        with mock.patch.object(rl.RoboLabEnv, "_task_dir_exists", return_value=True):
            self.assertEqual(rl.RoboLabEnv._resolve_task_dirs("robovolo"),
                             ["benchmark", "robovolo"])

    def test_it_mirrors_robolab_s_own_convention(self):
        """RoboLab's VOLO_TASK_SUBFOLDERS is [*DEFAULT, 'robovolo']."""
        self.assertEqual(rl.RoboLabEnv._TASK_PACKS["robovolo"][-1], "robovolo")
        self.assertEqual(rl.RoboLabEnv._TASK_PACKS["robovolo"][0], "benchmark")

    def test_an_unknown_pack_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            rl.RoboLabEnv._resolve_task_dirs("robovolo2")
        self.assertIn("unknown task_pack", str(ctx.exception))

    def test_a_missing_pack_raises_by_default(self):
        """The important one: silence here would mislabel every result."""
        with mock.patch.object(rl.RoboLabEnv, "_task_dir_exists",
                               side_effect=lambda d: d != "robovolo"):
            with self.assertRaises(FileNotFoundError) as ctx:
                rl.RoboLabEnv._resolve_task_dirs("robovolo")
        msg = str(ctx.exception)
        self.assertIn("robovolo", msg)
        self.assertIn("github.com/NVlabs/RoboVoLo", msg, "the error should say where to get it")
        self.assertIn("quietly fall back", msg)

    def test_a_missing_pack_can_be_waived_explicitly_and_warns(self):
        with mock.patch.object(rl.RoboLabEnv, "_task_dir_exists",
                               side_effect=lambda d: d != "robovolo"):
            with self.assertLogs("harness.envs.robolab", level="WARNING") as cm:
                dirs = rl.RoboLabEnv._resolve_task_dirs("robovolo", require=False)
        self.assertEqual(dirs, ["benchmark"])
        self.assertTrue(any("not installed" in m for m in cm.output))

    def test_a_missing_base_folder_also_raises(self):
        with mock.patch.object(rl.RoboLabEnv, "_task_dir_exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                rl.RoboLabEnv._resolve_task_dirs("benchmark")

    @unittest.skipUnless(_robolab_importable(), "RoboLab lives in its own venv")
    def test_it_documents_which_packs_this_machine_has(self):
        """Only meaningful from inside the RoboLab venv.

        The harness venv cannot import robolab at all, so both folders read as absent
        there -- which is why the resolver returning False on ImportError is the
        correct behaviour rather than a bug.
        """
        self.assertTrue(rl.RoboLabEnv._task_dir_exists("benchmark"),
                        "RoboLab's own benchmark folder should exist in its venv")
        if rl.RoboLabEnv._task_dir_exists("robovolo"):
            self.skipTest("RoboVoLo pack is installed -- rerun sweeps with "
                          "--task-pack robovolo and update the reports")


class TestNoneMeansDefault(unittest.TestCase):
    def test_an_empty_pack_falls_back_to_benchmark(self):
        with mock.patch.object(rl.RoboLabEnv, "_task_dir_exists", return_value=True):
            self.assertEqual(rl.RoboLabEnv._resolve_task_dirs(None), ["benchmark"])
            self.assertEqual(rl.RoboLabEnv._resolve_task_dirs(""), ["benchmark"])

    def test_case_is_not_significant(self):
        with mock.patch.object(rl.RoboLabEnv, "_task_dir_exists", return_value=True):
            self.assertEqual(rl.RoboLabEnv._resolve_task_dirs("RoboVoLo"),
                             ["benchmark", "robovolo"])


class TestExistenceCheckIsSafe(unittest.TestCase):
    def test_it_reports_false_when_robolab_is_absent(self):
        """The check must not explode on a machine without RoboLab installed."""
        import builtins

        real_import = builtins.__import__

        def _no_robolab(name, *a, **kw):
            if name.startswith("robolab"):
                raise ImportError("no robolab here")
            return real_import(name, *a, **kw)

        with mock.patch.object(builtins, "__import__", _no_robolab):
            self.assertFalse(rl.RoboLabEnv._task_dir_exists("benchmark"))


if __name__ == "__main__":
    unittest.main()
