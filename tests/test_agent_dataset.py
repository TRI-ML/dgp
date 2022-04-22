# Copyright 2021-2022 Toyota Research Institute. All rights reserved.
import os
import unittest

from dgp.datasets.agent_dataset import AgentDataset, AgentDatasetLite
from dgp.utils.testing import assert_true
from tests import TEST_DATA_DIR


class TestAgentDataset(unittest.TestCase):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")

    def setUp(self):
        self.test_scene_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene/scene_dataset_v1.0.json")

        self.agent_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene/agents_pcc_mini_v1.json")

    def test_prediction_agent_dataset_lite(self):
        #Test agent dataset loading
        expected_lidar_fields = set([
            'timestamp',
            'datum_name',
            'extrinsics',
            'pose',
            'point_cloud',
            'extra_channels',
            'datum_type',
        ])

        expected_camera_fields = set([
            'timestamp',
            'datum_name',
            'rgb',
            'intrinsics',
            'distortion',
            'extrinsics',
            'pose',
            'datum_type',
        ])

        dataset = AgentDatasetLite(
            self.test_scene_json,
            self.agent_json,
            split='train',
            requested_agent_type='agent_3d',
            datum_names=['lidar', 'CAMERA_01'],
            requested_main_agent_classes=('Car', 'Person'),
            requested_feature_types=("parked_car", ),
            batch_per_agent=False
        )
        # Check length of dataset
        assert len(dataset) == 6

        for item in dataset:
            for datum in item[0]['datums']:
                if datum['datum_name'] == 'LIDAR':
                    # Check LIDAR fields
                    assert_true(set(datum.keys()) == expected_lidar_fields)
                elif datum['datum_name'].startswith('CAMERA_'):
                    # CAMERA_01 should have intrinsics/extrinsics set
                    assert_true(datum['intrinsics'].shape == (3, 3))
                    assert_true(datum['extrinsics'].matrix.shape == (4, 4))
                    # Check CAMERA fields
                    assert_true(set(datum.keys()) == expected_camera_fields)
                else:
                    raise RuntimeError('Unexpected datum_name {}'.format(datum['datum_name']))

    def test_prediction_agent_dataset(self):
        expected_lidar_fields = set([
            'timestamp',
            'datum_name',
            'extrinsics',
            'pose',
            'point_cloud',
            'extra_channels',
            'datum_type',
        ])

        expected_camera_fields = set([
            'timestamp',
            'datum_name',
            'rgb',
            'intrinsics',
            'distortion',
            'extrinsics',
            'pose',
            'datum_type',
        ])

        dataset = AgentDataset(
            self.test_scene_json,
            self.agent_json,
            split='train',
            datum_names=['lidar', 'CAMERA_01'],
            requested_main_agent_classes=('Car', 'Person'),
            requested_feature_types=("parked_car", ),
            batch_per_agent=False
        )
        # Check dataset length
        assert len(dataset) == 313

        for item in dataset:
            for datum in item[0]['datums']:
                if datum['datum_name'] == 'LIDAR':
                    # Check LIDAR fields
                    assert_true(set(datum.keys()) == expected_lidar_fields)
                elif datum['datum_name'].startswith('CAMERA_'):
                    # CAMERA_01 should have intrinsics/extrinsics set
                    assert_true(datum['intrinsics'].shape == (3, 3))
                    assert_true(datum['extrinsics'].matrix.shape == (4, 4))
                    # Check CAMERA fields
                    assert_true(set(datum.keys()) == expected_camera_fields)
                else:
                    raise RuntimeError('Unexpected datum_name {}'.format(datum['datum_name']))


if __name__ == "__main__":
    unittest.main()
