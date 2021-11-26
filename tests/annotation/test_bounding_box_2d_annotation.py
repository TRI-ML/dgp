
import os
import numpy as np
import pytest
from dgp.annotations.bounding_box_2d_annotation import BoundingBox2DAnnotationList

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.utils.structures.bounding_box_2d import BoundingBox2D
from tests import TEST_DATA_DIR


@pytest.fixture
def bb_ontology():
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
    dataset = SynchronizedSceneDataset(
        scenes_dataset_json,
        split='train',
        datum_names=['camera_01', 'lidar'],
        backward_context=0,
        requested_annotations=("bounding_box_2d", "bounding_box_3d")
    )
    return dataset.dataset_metadata.ontology_table.get('bounding_box_3d', None)


def test_bb2d_annotation(bb_ontology):
    bounding_boxes = [BoundingBox2D(box=np.float32([i, i+5, i, i+5])) for i in range(5)]
    annotation_list = BoundingBox2DAnnotationList(bb_ontology,bounding_boxes)
    assert len(annotation_list.ltrb) == 5