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

    def test_blocks_subclasses_sandbox_escape(self):
        # classic object-graph-walk escape: reach a live module (e.g. os) via
        # ().__class__.__base__.__subclasses__() without ever importing it.
        code = (
            "target = None\n"
            "for c in ().__class__.__base__.__subclasses__():\n"
            "    try:\n"
            "        g = c.__init__.__globals__\n"
            "    except:\n"
            "        continue\n"
            "    if 'os' in g:\n"
            "        target = g['os']\n"
            "        break\n"
            "target.system('id')\n"
        )
        r = execute_code(code, namespace={})
        self.assertFalse(r["ok"])
        self.assertIn("dunder", r["error"])

    def test_blocks_bare_dunder_name(self):
        r = execute_code("print(__builtins__)", namespace={})
        self.assertFalse(r["ok"])


if __name__ == "__main__":
    unittest.main()
