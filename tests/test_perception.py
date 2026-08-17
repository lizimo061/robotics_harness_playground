import unittest

import numpy as np

from harness.perception import encode_image, image_to_data_uri


class TestPerception(unittest.TestCase):
    def test_encode_rgb(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        b64 = encode_image(img)
        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 0)

    def test_encode_gray_to_rgb(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        b64 = encode_image(img)
        self.assertIsInstance(b64, str)

    def test_data_uri(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        uri = image_to_data_uri(img)
        self.assertTrue(uri.startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
