import os
import pytest
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.datasets.frame_dataset import FrameSceneDataset

from tests import TEST_DATA_DIR


@pytest.mark.parametrize(
    "split, datum_names, requested_autolabels, only_annotated_datums, use_diskcache, requested_annotations,skip_missing_data,expected_len",
    [
        ('train', ['LIDAR', 'CAMERA_01'], None, False, True, ("bounding_box_2d", "bounding_box_3d"), False, 2),
    ]
)
@pytest.mark.timeout(6)
def test_create_sync_scene_dataset(
    split, datum_names, requested_autolabels, only_annotated_datums, use_diskcache, requested_annotations,
    skip_missing_data, expected_len
):
    '''
    Uses parametrized testing to run multiple cases for FrameSceneDataset
    '''

    scenes_dataset_json = os.path.join(os.path.join(TEST_DATA_DIR, "dgp"), "test_scene", "scene_dataset_v1.0.json")
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


@pytest.mark.parametrize(
    "split,datum_names,forward_context,backward_context,generate_depth_from_datum,requested_annotations,expected_len", [
        ('train', ['LIDAR', 'CAMERA_01'], 1, 1, 'LIDAR', ("bounding_box_2d", "bounding_box_3d"), 2),
    ]
)
@pytest.mark.timeout(60)
def test_create_sync_scene_dataset(
    split, datum_names, forward_context, backward_context, generate_depth_from_datum, requested_annotations,
    expected_len
):
    '''
    Uses parametrized testing to run multiple cases for SynchronizedSceneDataset
    '''

    scenes_dataset_json = os.path.join(os.path.join(TEST_DATA_DIR, "dgp"), "test_scene", "scene_dataset_v1.0.json")
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
