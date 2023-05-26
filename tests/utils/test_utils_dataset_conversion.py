# Copyright 2021 Toyota Research Institute. All rights reserved.
import unittest

import numpy as np

from dgp.utils.dataset_conversion import read_cloud_ply, write_cloud_ply


class TestDataConversion(unittest.TestCase):
    def test_ply_write_read(self):
        N = 10
        TEST_FILENAME = "test.ply"
        test_points = np.random.random((N, 3))
        test_intensities = np.random.randint(0, 255, size=(N, ), dtype=np.uint8)
        test_timestamps = np.random.random((N, ))

        write_cloud_ply(TEST_FILENAME, test_points, test_intensities, test_timestamps)
        out_points, out_intensities, out_timestamps = read_cloud_ply(TEST_FILENAME)  # pylint: disable=unbalanced-tuple-unpacking
        self.assertTrue(np.array_equal(out_points, test_points))
        self.assertTrue(np.array_equal(out_intensities, test_intensities))
        self.assertTrue(np.array_equal(out_timestamps, test_timestamps))

        write_cloud_ply(TEST_FILENAME, test_points, test_intensities)
        out_points, out_intensities, _ = read_cloud_ply(TEST_FILENAME)  # pylint: disable=unbalanced-tuple-unpacking
        self.assertTrue(np.array_equal(out_points, test_points))
        self.assertTrue(np.array_equal(out_intensities, test_intensities))

        write_cloud_ply(TEST_FILENAME, test_points)
        out_points, _, _ = read_cloud_ply(TEST_FILENAME)  # pylint: disable=unbalanced-tuple-unpacking
        self.assertTrue(np.array_equal(out_points, test_points))

        self.assertRaises(AssertionError, write_cloud_ply, TEST_FILENAME, test_points, test_timestamps)


if __name__ == "__main__":
    unittest.main()
