import glob
import os
import unittest
from collections import OrderedDict

from dgp import DGP_CACHE_DIR, TRI_DGP_JSON_PREFIX
from dgp.datasets.cache import diskcache
from dgp.datasets.synchronized_dataset import (SynchronizedScene,
                                               SynchronizedSceneDataset)
from dgp.utils.geometry import Pose
from dgp.utils.testing import assert_raises, assert_true
from tests import TEST_DATA_DIR


class TestDataset(unittest.TestCase):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    EXPECTED_FIELDS = set([
        'index',
        'rgb',
        'timestamp',
        'datum_name',
    ])

    @staticmethod
    def _test_labeled_dataset(dataset):
        expected_camera_fields = set([
            'rgb', 'timestamp', 'datum_name', 'pose', 'intrinsics', 'extrinsics', 'bounding_box_2d', 'bounding_box_3d',
            'class_ids', 'instance_ids'
        ])
        expected_lidar_fields = set([
            'point_cloud', 'timestamp', 'datum_name', 'pose', 'extrinsics', 'bounding_box_3d', 'class_ids',
            'instance_ids', 'extra_channels'
        ])

        # Iterate through labeled dataset and check expected fields
        assert dataset.calibration_table is not None
        for _, item in enumerate(dataset):
            # Context size is 3 (forward + backward + reference)
            assert_true(len(item) == 3)

            # Check both datum and time-dimensions for expected fields
            for t_item in item:
                # Four selected datums
                assert_true(len(t_item) == 4)

            # LIDAR should have point_cloud set
            for t_item in item:
                assert_true(set(t_item[0].keys()) == expected_lidar_fields)

            # CAMERA_01 should have intrinsics/extrinsics set
            im_size = None
            for t_item in item:
                assert_true(t_item[1]['intrinsics'].shape == (3, 3))
                assert_true(t_item[1]['extrinsics'].matrix.shape == (4, 4))
                # Check image sizes for context frames
                assert_true(set(t_item[1].keys()) == expected_camera_fields)
                if im_size is None:
                    im_size = t_item[1]['rgb'].size
                assert_true(t_item[1]['rgb'].size == im_size)

    def test_labeled_synchronized_scene_dataset(self):
        """Test synchronized scene dataset"""
        expected_camera_fields = set([
            'rgb', 'timestamp', 'datum_name', 'pose', 'intrinsics', 'extrinsics', 'bounding_box_2d', 'bounding_box_3d',
            'class_ids', 'instance_ids', 'depth'
        ])
        expected_lidar_fields = set([
            'point_cloud', 'timestamp', 'datum_name', 'pose', 'extrinsics', 'bounding_box_3d', 'class_ids',
            'instance_ids', 'extra_channels'
        ])
        expected_metadata_fields = set([
            'scene_index', 'sample_index_in_scene', 'log_id', 'timestamp', 'scene_name', 'scene_description'
        ])

        # Initialize synchronized dataset with 2 datums
        scenes_dataset_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
        dataset = SynchronizedSceneDataset(
            scenes_dataset_json,
            split='train',
            forward_context=1,
            backward_context=1,
            generate_depth_from_datum='LIDAR',
            requested_annotations=("bounding_box_2d", "bounding_box_3d")
        )
        dataset.select_datums(['LIDAR', 'CAMERA_01'])
        dataset.prefetch()

        # There are only 3 samples in the train and val split.
        # With a forward and backward context of 1 each, the number of
        # items in the dataset with the desired context frames is 1.
        assert len(dataset) == 2

        # Iterate through labeled dataset and check expected fields
        assert dataset.calibration_table is not None
        for idx, item in enumerate(dataset):
            # Context size is 3 (forward + backward + reference)
            assert_true(len(item) == 3)

            # Two selected datums
            for t_item in item:
                assert_true(len(t_item) == 2)

            # LIDAR should have point_cloud set
            for t_item in item:
                assert_true(set(t_item[0].keys()) == expected_lidar_fields)
                assert_true(isinstance(t_item[0], OrderedDict))

            # CAMERA_01 should have intrinsics/extrinsics set
            im_size = None
            for t_item in item:
                assert_true(isinstance(t_item[1], OrderedDict))
                assert_true(t_item[1]['intrinsics'].shape == (3, 3))
                assert_true(isinstance(t_item[1]['extrinsics'], Pose))
                assert_true(isinstance(t_item[1]['pose'], Pose))
                # Check image sizes for context frames
                assert_true(set(t_item[1].keys()) == expected_camera_fields)
                if im_size is None:
                    im_size = t_item[1]['rgb'].size
                assert_true(t_item[1]['rgb'].size == im_size)

            # Retrieve metadata about dataset item at index=idx
            metadata = dataset.get_scene_metadata(idx)
            assert_true(metadata.keys() == expected_metadata_fields)

        # Make sure you cannot select unavailable datums
        with assert_raises(AssertionError) as _:
            dataset.select_datums(['FAKE_LIDAR_NAME'])

    def test_synchronized_scene(self):
        """Test a single synchronized scene with labels"""
        scene_json = os.path.join(
            self.DGP_TEST_DATASET_DIR, "test_scene/scene_01/scene_a8dc5ed1da0923563f85ea129f0e0a83e7fe1867.json"
        )
        dataset = SynchronizedScene(
            scene_json,
            forward_context=1,
            backward_context=1,
            requested_annotations=("bounding_box_2d", "bounding_box_3d")
        )
        dataset.select_datums(['LIDAR', 'CAMERA_01', 'CAMERA_05', 'CAMERA_06'])
        dataset.prefetch()
        TestDataset._test_labeled_dataset(dataset)

    def test_cached_synchronized_scene_dataset(self):
        """Test cached synchronized scene dataset"""

        # Initialize synchronized dataset with 2 datums
        scenes_dataset_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")

        # Intialize dataset, and check to see if we have cached any new files.
        dataset_args = (scenes_dataset_json, )
        dataset_kwargs = dict(
            split='train',
            datum_names=('LIDAR', 'CAMERA_01'),
            requested_annotations=("bounding_box_2d", "bounding_box_3d")
        )
        dataset = diskcache(protocol='pkl')(SynchronizedSceneDataset)(*dataset_args, **dataset_kwargs)
        cached_files = set(glob.glob(os.path.join(DGP_CACHE_DIR, '*.pkl')))

        # There are only 2 secnes, 6 samples in the train and val split.
        assert_true(len(dataset) == 6)

        # Reinitialize dataset, this should load the cached version.
        cached_dataset = diskcache(protocol='pkl')(SynchronizedSceneDataset)(*dataset_args, **dataset_kwargs)
        # Check to see if the number of cached files have not changed.
        assert_true(set(cached_files) == set(glob.glob(os.path.join(DGP_CACHE_DIR, '*.pkl'))))
        assert_true(len(cached_dataset) == len(dataset))
        assert_true(cached_dataset.datum_index == dataset.datum_index)
        assert_true(cached_dataset.dataset_item_index == dataset.dataset_item_index)


if __name__ == "__main__":
    unittest.main()
