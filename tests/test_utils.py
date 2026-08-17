import unittest

import numpy as np

from harness.utils import get_logger, set_seed


class TestUtils(unittest.TestCase):
    def test_set_seed_deterministic(self):
        set_seed(0)
        a = np.random.rand(5)
        set_seed(0)
        b = np.random.rand(5)
        np.testing.assert_allclose(a, b)

    def test_get_logger(self):
        log = get_logger("test_utils")
        self.assertIsNotNone(log)


if __name__ == "__main__":
    unittest.main()
