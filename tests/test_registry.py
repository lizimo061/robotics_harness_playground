import unittest

from harness.registry import Registry


class TestRegistry(unittest.TestCase):
    def test_register_and_get(self):
        reg = Registry("test")
        reg.register("a")(lambda: 1)
        self.assertEqual(reg.get("a")(), 1)
        self.assertIn("a", reg)
        self.assertEqual(reg.available(), ["a"])

    def test_unknown_key(self):
        reg = Registry("test")
        with self.assertRaises(KeyError):
            reg.get("nope")


if __name__ == "__main__":
    unittest.main()
