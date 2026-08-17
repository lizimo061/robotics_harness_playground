import unittest

from harness.agent.code_executor import execute_code


class TestCodeExecutor(unittest.TestCase):
    def test_allowed_code(self):
        r = execute_code("x = sum([1, 2, 3]); print(x)", namespace={})
        self.assertTrue(r["ok"])
        self.assertIn("6", r["output"])

    def test_blocks_os_import(self):
        r = execute_code("import os", namespace={})
        self.assertFalse(r["ok"])
        self.assertIn("not allowed", r["error"])

    def test_blocks_open_call(self):
        r = execute_code("open('/etc/passwd')", namespace={})
        self.assertFalse(r["ok"])

    def test_namespace_functions_are_callable(self):
        calls = []

        def skill(x):
            calls.append(x)
            return x

        r = execute_code("skill(42)", namespace={"skill": skill})
        self.assertTrue(r["ok"])
        self.assertEqual(calls, [42])

    def test_syntax_error(self):
        r = execute_code("def broken(:", namespace={})
        self.assertFalse(r["ok"])


if __name__ == "__main__":
    unittest.main()
