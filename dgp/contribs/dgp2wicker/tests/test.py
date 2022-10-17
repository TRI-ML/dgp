# Copyright 2022 Woven Planet. All rights reserved.
import json
import os
import unittest

import cv2
import dgp2wicker.serializers as ws
import numpy as np
import wicker.plugins.spark as wsp
from click.testing import CliRunner
from dgp2wicker.cli import ingest
from dgp2wicker.dataset import DGPS3Dataset, compute_columns
from dgp2wicker.ingest import (
    FIELD_TO_WICKER_SERIALIZER,
    dgp_to_wicker_sample,
    gen_wicker_key,
    ingest_dgp_to_wicker,
    parse_wicker_key,
    wicker_types_from_sample,
)
from PIL.Image import Image

import dgp
from dgp.datasets import SynchronizedSceneDataset

DGP_DIR = os.path.dirname(os.path.relpath(dgp.__file__))
TEST_DATA_DIR = os.path.join(DGP_DIR, '..', 'tests', 'data', 'dgp')

TEST_WICKER_DATASET_JSON = os.path.join(TEST_DATA_DIR, "test_scene", "scene_dataset_v1.0.json")
TEST_WICKER_DATASET_NAME = 'test_wicker_dataset'
TEST_WICKER_DATASET_VERSION = '0.0.1'
TEST_WICKER_DATASET_KWARGS = {
    'datum_names': [
        'LIDAR',
        'CAMERA_01',
    ],
    'requested_annotations': ['bounding_box_2d', 'bounding_box_3d'],
    'only_annotated_datums': True
}


def s3_is_configured():
    """Utility to check if there is a valid s3 dataset path configured"""
    wicker_config_path = os.getenv('WICKER_CONFIG_PATH', os.path.expanduser('~/wickerconfig.json'))
    with open(wicker_config_path, 'r', encoding='utf-8') as f:
        wicker_config = json.loads(f.read())

    return wicker_config['aws_s3_config']['s3_datasets_path'].startswith('s3://')


class TestDDGP2Wicker(unittest.TestCase):
    def setUp(self):
        """Create a local dgp dataset for testing"""
        self.dataset = SynchronizedSceneDataset(
            TEST_WICKER_DATASET_JSON,
            split='train',
            datum_names=['LIDAR', 'CAMERA_01'],
            forward_context=0,
            backward_context=0,
            requested_annotations=("bounding_box_2d", "bounding_box_3d")
        )

    def test_keys(self):
        """Sanity check the key parsing"""
        datum_key, datum_field = 'CAMERA_01', 'timestamp'
        key = gen_wicker_key(datum_key, datum_field)
        datum_key2, datum_field2 = parse_wicker_key(key)
        assert datum_key == datum_key2
        assert datum_field == datum_field2

    def test_schema(self):
        """Sanity check the schema generation"""
        sample = self.dataset[0][0]
        ontology_table = self.dataset.dataset_metadata.ontology_table
        wicker_types = wicker_types_from_sample(sample, ontology_table, skip_camera_cuboids=True)
        expected_keys = [
            'CAMERA_01____timestamp', 'CAMERA_01____rgb', 'CAMERA_01____intrinsics', 'CAMERA_01____distortion',
            'CAMERA_01____extrinsics', 'CAMERA_01____pose', 'CAMERA_01____bounding_box_2d', 'CAMERA_01____datum_type',
            'LIDAR____timestamp', 'LIDAR____extrinsics', 'LIDAR____pose', 'LIDAR____point_cloud',
            'LIDAR____extra_channels', 'LIDAR____bounding_box_3d', 'LIDAR____datum_type', 'ontology____bounding_box_2d',
            'ontology____bounding_box_3d', 'scene_index', 'sample_index_in_scene', 'scene_uri'
        ]

        assert (set(expected_keys) == set(wicker_types.keys()))

    def test_conversion(self):
        """Test serializers and conversion to wicker formats"""
        sample = self.dataset[0][0]
        ontology_table = self.dataset.dataset_metadata.ontology_table
        wicker_types = wicker_types_from_sample(sample, ontology_table, skip_camera_cuboids=True)

        sample_dict = {datum['datum_name']: datum for datum in sample}
        # This tests wicker serialization
        wicker_sample = dgp_to_wicker_sample(
            sample=sample,
            wicker_keys=list(wicker_types.keys()),
            scene_index=0,
            sample_index_in_scene=0,
            ontology_table=ontology_table,
            scene_uri='scene/scene.json'
        )
        assert set(wicker_sample.keys()) == set(wicker_types.keys())

        # Test that we can correctly unserialize all the objects
        # NOTE: this only tests the datum_name/datum_filed combinations
        # for what is actually in the sample dataset. Types not available in the small
        # public test dataset should be manually tested offline.
        for key, raw in wicker_sample.items():
            # Parse the key to figure out what datum/field we have
            if key in ('scene_index', 'sample_index_in_scene', 'scene_uri'):
                continue
            datum_key, datum_field = parse_wicker_key(key)
            # Grab the correct serializer
            if datum_key == 'ontology':
                serializer = ws.OntologySerializer(datum_field)
            elif datum_field in FIELD_TO_WICKER_SERIALIZER:
                serializer = FIELD_TO_WICKER_SERIALIZER[datum_field]()
            else:
                print(f'{key} not supported')
                continue

            if hasattr(serializer, 'ontology'):
                serializer.ontology = ontology_table[datum_field]

            unserialized = serializer.unserialize(raw)

            # comparison depends on the type. Images for example are wickerized to jpeg,
            # so there may be some slight loss of quality. Numpy arrays should just be close etc.
            if datum_field == 'rgb':
                assert isinstance(unserialized, Image)
                org_im = np.array(sample_dict[datum_key]['rgb'])
                new_im = np.array(unserialized)
                psnr = cv2.PSNR(org_im, new_im)
                assert psnr > 40
            elif datum_field in ('point_cloud', 'extra_channels', 'intrinsics'):
                org = sample_dict[datum_key][datum_field]
                assert np.allclose(org, unserialized)
            elif datum_field in ('pose', 'extrinsics'):
                org = sample_dict[datum_key][datum_field]
                assert np.allclose(org.matrix, unserialized.matrix)
            elif datum_key != 'ontology':
                org = sample_dict[datum_key][datum_field]
                assert org == unserialized

    @unittest.skipUnless(s3_is_configured(), 'Requires S3')
    def test_ingest(self):
        """Test ingestion"""
        # The test dataset is really small, smaller than the expected partition size
        wsp.SPARK_PARTITION_SIZE = 12

        output = ingest_dgp_to_wicker(
            scene_dataset_json=TEST_WICKER_DATASET_JSON,
            wicker_dataset_name=TEST_WICKER_DATASET_NAME,
            wicker_dataset_version=TEST_WICKER_DATASET_VERSION,
            dataset_kwargs=TEST_WICKER_DATASET_KWARGS,
            spark_context=None,
            pipeline=None,
            max_num_scenes=None,
            max_len=1000,
            chunk_size=1000,
            skip_camera_cuboids=True,
            num_partitions=None,
            num_repartitions=None,
            is_pd=False,
            data_uri=None,
        )

        assert output['train'] == 6
        assert output['val'] == 6

    @unittest.skipUnless(s3_is_configured(), 'Requires S3')
    def test_ingest_cli(self):
        """Test ingestion via the cli"""

        # The test dataset is really small, smaller than the expected partition size
        wsp.SPARK_PARTITION_SIZE = 12

        cmd = f'--scene-dataset-json {TEST_WICKER_DATASET_JSON}\n'
        cmd += f'--wicker-dataset-name {TEST_WICKER_DATASET_NAME}\n'
        cmd += f'--wicker-dataset-version {TEST_WICKER_DATASET_VERSION}\n'
        cmd += '--datum-names CAMERA_01,LIDAR\n'
        cmd += '--requested-annotations bounding_box_2d,bounding_box_3d\n'
        cmd += '--only-annotated-datums\n'
        cmd += '--half-size-images\n'
        cmd += '--add-lidar-points'

        runner = CliRunner()
        result = runner.invoke(ingest, cmd)

        assert result.exit_code == 0

    @unittest.skipUnless(s3_is_configured(), 'Requires S3')
    def test_dataset(self):
        """Test That we can read a dataset from wicker"""
        self.test_ingest()

        columns = compute_columns(
            datum_names=[
                'CAMERA_01',
                'LIDAR',
            ],
            datum_types=[
                'image',
                'point_cloud',
            ],
            requested_annotations=['bounding_box_2d', 'bounding_box_3d'],
            cuboid_datum='LIDAR',
            with_ontology_table=True
        )

        dataset = DGPS3Dataset(
            dataset_name=TEST_WICKER_DATASET_NAME,
            dataset_version=TEST_WICKER_DATASET_VERSION,
            dataset_partition_name='train',
            columns_to_load=columns,
        )
        sample = dataset[0][0]

        expected_camera_fields = {
            'extrinsics', 'bounding_box_2d', 'pose', 'datum_name', 'datum_type', 'distortion', 'intrinsics', 'rgb',
            'timestamp'
        }
        expected_lidar_fields = {
            'pose', 'datum_name', 'datum_type', 'extra_channels', 'point_cloud', 'bounding_box_3d', 'extrinsics',
            'timestamp'
        }

        assert set(sample['CAMERA_01'].keys()) == expected_camera_fields
        assert isinstance(sample['CAMERA_01']['rgb'], Image)
        assert set(sample['LIDAR'].keys()) == expected_lidar_fields
        assert set(dataset.ontology_table.keys()) == {'bounding_box_2d', 'bounding_box_3d'}


if __name__ == '__main__':
    unittest.main()
