# Copyright 2020 Toyota Research Institute. All rights reserved.
import unittest

from dgp.utils.colors import get_unique_colors


class TestDGPUnitTestColors(unittest.TestCase):
    def test_get_unique_colors(self):
        num_colors = 3
        in_bgr = False
        res = get_unique_colors(num_colors=num_colors, in_bgr=in_bgr)
        self.assertEquals(len(res), 3)
        return

    def test_empty_colors(self):
        num_colors = 0
        in_bgr = False
        res = get_unique_colors(num_colors=num_colors, in_bgr=in_bgr)
        self.assertEquals(len(res), 0)
        return

if __name__ == "__main__":
    unittest.main()
