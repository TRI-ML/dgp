# Copyright 2021 Toyota Research Institute.  All rights reserved.
# TODO: merge this test into tests/test_datasets.py once it migrates to pytest.
import os

import pytest

from dgp.datasets.frame_dataset import FrameSceneDataset
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from tests import TEST_DATA_DIR
from tests.utilities import requires_env


@requires_env("TEST_RUNNER")
@pytest.mark.parametrize(
    "split, datum_names, requested_autolabels, only_annotated_datums, use_diskcache, requested_annotations, skip_missing_data, expected_len",
    [
        ('train', ['LIDAR', 'CAMERA_01'], None, False, True, ("bounding_box_2d", "bounding_box_3d"), False, 12),
    ]
)
@pytest.mark.timeout(6.0)  # Repeat 10 times, takes ~0.6 second to instantiate FrameSceneDataset.
def test_create_frame_scene_dataset( # pylint: disable=missing-any-param-doc
    split, datum_names, requested_autolabels, only_annotated_datums, use_diskcache, requested_annotations,
    skip_missing_data, expected_len
):
    """Uses parametrized testing to run multiple cases for FrameSceneDataset."""

    scenes_dataset_json = os.path.join(os.path.join(TEST_DATA_DIR, "dgp"), "test_scene", "scene_dataset_v1.0.json")
    for _ in range(10):
        dataset = FrameSceneDataset(
            scenes_dataset_json,
            split=split,
            datum_names=datum_names,
            requested_autolabels=requested_autolabels,
            only_annotated_datums=only_annotated_datums,
            use_diskcache=use_diskcache,
            skip_missing_data=skip_missing_data,
            requested_annotations=requested_annotations,
        )
        assert len(dataset) == expected_len


@requires_env("TEST_RUNNER")
@pytest.mark.parametrize(
    "split, datum_names, forward_context, backward_context, generate_depth_from_datum, requested_annotations, expected_len",
    [
        ('train', ['LIDAR', 'CAMERA_01'], 1, 1, 'LIDAR', ("bounding_box_2d", "bounding_box_3d"), 2),
    ]
)
@pytest.mark.timeout(6.0)  # Repeat 10 times, takes ~0.6 second to instantiate SynchronizedSceneDataset.
def test_create_sync_scene_dataset( # pylint: disable=missing-any-param-doc
    split, datum_names, forward_context, backward_context, generate_depth_from_datum, requested_annotations,
    expected_len
):
    """Uses parametrized testing to run multiple cases for SynchronizedSceneDataset."""

    scenes_dataset_json = os.path.join(os.path.join(TEST_DATA_DIR, "dgp"), "test_scene", "scene_dataset_v1.0.json")
    for _ in range(10):
        dataset = SynchronizedSceneDataset(
            scenes_dataset_json,
            split=split,
            datum_names=datum_names,
            forward_context=forward_context,
            backward_context=backward_context,
            generate_depth_from_datum=generate_depth_from_datum,
            requested_annotations=requested_annotations
        )
        assert len(dataset) == expected_len
