import glob
import os
import unittest

import numpy as np

from group_9.optimizer_2d import load_image


class Part1(unittest.TestCase):
    def test_load_image_if_file_exists_return_np_array(self):
        img = load_image(os.path.join(os.pardir, "group_9", "fn", "camel3.png"))
        self.assertEqual(img.dtype, np.dtype('float64'))

    def test_scaling_to_range_0_1(self):
        img_paths = glob.glob(os.path.join(os.pardir, "group_9", "fn", "*.png"))
        for i in img_paths:
            img = load_image(i)
            self.assertTrue(np.min(img) >= 0)
            self.assertTrue(np.max(img) <= 1)

    def test_load_image_if_file_not_exists(self):
        with self.assertRaises(FileNotFoundError):
            load_image(os.path.join("fn", "this-is-not-an-image.png"))


if __name__ == '__main__':
    unittest.main()
