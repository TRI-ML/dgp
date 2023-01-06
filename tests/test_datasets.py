import glob
import os
import unittest
from collections import OrderedDict

from dgp import DGP_CACHE_DIR
from dgp.datasets.synchronized_dataset import (
    SynchronizedScene,
    SynchronizedSceneDataset,
)
from dgp.utils.cache import diskcache
from dgp.utils.pose import Pose
from dgp.utils.testing import assert_array_equal, assert_true
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
    def _test_labeled_dataset(dataset, with_rgb=True, with_points=True):
        """Test the dataset

        Parameters
        ----------
        dataset: a dgp dataset
        with_rgb: bool, default: True
            Determines how we should test the output of datum['rgb']. If false, datum['rgb'] should be None
        with_points: bool, default: True
            Determines how we should test the output of datum['point_cloud'] and datum['extra_channels']

        Raises
        ------
        RuntimeError
            If an unexpected datum is found
        """
        expected_camera_fields = set([
            'rgb',
            'timestamp',
            'datum_name',
            'pose',
            'intrinsics',
            'distortion',
            'extrinsics',
            'bounding_box_2d',
            'bounding_box_3d',
            'datum_type',
        ])
        expected_lidar_fields = set([
            'point_cloud',
            'timestamp',
            'datum_name',
            'pose',
            'extrinsics',
            'bounding_box_2d',
            'bounding_box_3d',
            'extra_channels',
            'datum_type',
        ])

        # Iterate through labeled dataset and check expected fields
        assert dataset.calibration_table is not None
        for _, item in enumerate(dataset):
            # Context size is 3 (forward + backward + reference)
            assert_true(len(item) == 3)

            # Check both datum and time-dimensions for expected fields
            im_size = None
            for t_item in item:
                # Four selected datums
                assert_true(len(t_item) == 4)
                for datum in t_item:
                    if datum['datum_name'] == 'LIDAR':
                        # LIDAR should have point_cloud set
                        assert_true(set(datum.keys()) == expected_lidar_fields)
                        if with_points:
                            N = datum['point_cloud'].shape[0]
                            assert_true(datum['point_cloud'].shape == (N, 3))
                            assert_true(datum['extra_channels'].shape[0] == N)
                        else:
                            assert_true(datum['point_cloud'] is None)
                            assert_true(datum['extra_channels'] is None)

                    elif datum['datum_name'].startswith('CAMERA_'):
                        # CAMERA_01 should have intrinsics/extrinsics set
                        assert_true(datum['intrinsics'].shape == (3, 3))
                        assert_true(datum['extrinsics'].matrix.shape == (4, 4))
                        # Check image sizes for context frames
                        assert_true(set(datum.keys()) == expected_camera_fields)
                        if with_rgb:
                            if im_size is None:
                                im_size = datum['rgb'].size
                            assert_true(datum['rgb'].size == im_size)
                        else:
                            assert datum['rgb'] is None
                    else:
                        raise RuntimeError('Unexpected datum_name {}'.format(datum['datum_name']))

    def test_labeled_synchronized_scene_dataset(self):
        """Test synchronized scene dataset"""
        expected_camera_fields = set([
            'rgb',
            'timestamp',
            'datum_name',
            'pose',
            'intrinsics',
            'distortion',
            'extrinsics',
            'bounding_box_2d',
            'bounding_box_3d',
            'depth',
            'datum_type',
        ])
        expected_lidar_fields = set([
            'point_cloud',
            'timestamp',
            'datum_name',
            'pose',
            'extrinsics',
            'bounding_box_2d',
            'bounding_box_3d',
            'extra_channels',
            'datum_type',
        ])
        expected_metadata_fields = set([
            'scene_index', 'sample_index_in_scene', 'log_id', 'timestamp', 'scene_name', 'scene_description'
        ])

        # Initialize synchronized dataset with 2 datums
        scenes_dataset_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
        dataset = SynchronizedSceneDataset(
            scenes_dataset_json,
            split='train',
            datum_names=['LIDAR', 'CAMERA_01'],
            forward_context=1,
            backward_context=1,
            generate_depth_from_datum='LIDAR',
            requested_annotations=("bounding_box_2d", "bounding_box_3d")
        )

        # There are only 3 samples in the train and val split.
        # With a forward and backward context of 1 each, the number of
        # items in the dataset with the desired context frames is 1.
        assert len(dataset) == 2

        # Iterate through labeled dataset and check expected fields
        assert dataset.calibration_table is not None
        for idx, item in enumerate(dataset):
            # Context size is 3 (forward + backward + reference)
            assert_true(len(item) == 3)

            # Check both datum and time-dimensions for expected fields
            im_size = None
            for t_item in item:
                # Two selected datums
                assert_true(len(t_item) == 2)
                for datum in t_item:
                    if datum['datum_name'] == 'LIDAR':
                        # LIDAR should have point_cloud set
                        assert_true(set(datum.keys()) == expected_lidar_fields)
                        assert_true(isinstance(datum, OrderedDict))
                    elif datum['datum_name'].startswith('CAMERA_'):
                        # CAMERA_01 should have intrinsics/extrinsics set
                        assert_true(isinstance(datum, OrderedDict))
                        assert_true(datum['intrinsics'].shape == (3, 3))
                        assert_true(isinstance(datum['extrinsics'], Pose))
                        assert_true(isinstance(datum['pose'], Pose))
                        # Check image sizes for context frames
                        assert_true(set(datum.keys()) == expected_camera_fields)
                        if im_size is None:
                            im_size = datum['rgb'].size
                        assert_true(datum['rgb'].size == im_size)
                    else:
                        raise RuntimeError('Unexpected datum_name {}'.format(datum['datum_name']))

            # Retrieve metadata about dataset item at index=idx
            metadata = dataset.get_scene_metadata(idx)
            assert_true(metadata.keys() == expected_metadata_fields)

    def test_synchronized_scene(self):
        """Test a single synchronized scene with labels"""
        scene_json = os.path.join(
            self.DGP_TEST_DATASET_DIR, "test_scene/scene_01/scene_a8dc5ed1da0923563f85ea129f0e0a83e7fe1867.json"
        )
        dataset = SynchronizedScene(
            scene_json,
            datum_names=['LIDAR', 'CAMERA_01', 'CAMERA_05', 'CAMERA_06'],
            forward_context=1,
            backward_context=1,
            requested_annotations=("bounding_box_2d", "bounding_box_3d")
        )
        TestDataset._test_labeled_dataset(dataset)

    def test_synchronized_scene_without_rgb(self):
        """Test a single synchronized scene with labels but without rgb loading"""
        scene_json = os.path.join(
            self.DGP_TEST_DATASET_DIR, "test_scene/scene_01/scene_a8dc5ed1da0923563f85ea129f0e0a83e7fe1867.json"
        )
        dataset = SynchronizedScene(
            scene_json,
            datum_names=['LIDAR', 'CAMERA_01', 'CAMERA_05', 'CAMERA_06'],
            forward_context=1,
            backward_context=1,
            requested_annotations=("bounding_box_2d", "bounding_box_3d"),
            ignore_raw_datum=['image'],
        )
        TestDataset._test_labeled_dataset(dataset, with_rgb=False)

    def test_synchronized_scene_without_points(self):
        """Test a single synchronized scene with labels but without point loading"""
        scene_json = os.path.join(
            self.DGP_TEST_DATASET_DIR, "test_scene/scene_01/scene_a8dc5ed1da0923563f85ea129f0e0a83e7fe1867.json"
        )
        dataset = SynchronizedScene(
            scene_json,
            datum_names=['LIDAR', 'CAMERA_01', 'CAMERA_05', 'CAMERA_06'],
            forward_context=1,
            backward_context=1,
            requested_annotations=("bounding_box_2d", "bounding_box_3d"),
            ignore_raw_datum=['point_cloud'],
        )
        TestDataset._test_labeled_dataset(dataset, with_points=False)

    def test_cached_synchronized_scene_dataset(self):
        """Test cached synchronized scene dataset"""

        # Initialize synchronized dataset with 2 datums
        scenes_dataset_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")

        # Intialize dataset, and check to see if we have cached any new files.
        dataset_args = (scenes_dataset_json, )
        dataset_kwargs = dict(
            split='train',
            datum_names=['LIDAR', 'CAMERA_01'],
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
        for idx in range(len(cached_dataset.datum_index)):
            assert_array_equal(cached_dataset.datum_index[idx], dataset.datum_index[idx])
        assert_true(cached_dataset.dataset_item_index == dataset.dataset_item_index)


if __name__ == "__main__":
    unittest.main()
