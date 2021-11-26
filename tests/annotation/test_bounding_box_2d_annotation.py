import os
import numpy as np
import pytest
from dgp.annotations.bounding_box_2d_annotation import BoundingBox2DAnnotationList

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.utils.structures.bounding_box_2d import BoundingBox2D
from tests import TEST_DATA_DIR
from tests.annotation.test_ontology import get_ontology


@pytest.fixture
def bb_ontology():
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
    return get_ontology(scene_dataset_json=scenes_dataset_json, annotation_type="bounding_box_3d")


def test_bb2d_annotation(bb_ontology):
    bounding_boxes = [BoundingBox2D(box=np.float32([i, i + 5, i, i + 5])) for i in range(5)]
    annotation_list = BoundingBox2DAnnotationList(bb_ontology, bounding_boxes)
    assert len(annotation_list.ltrb) == 5


def test_bb2d_load(bb_ontology):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR, "test_scene/scene_01/bounding_box_2d/CAMERA_01/15569195938203752.json"
    )
    bb2d_list = BoundingBox2DAnnotationList.load(scenes_dataset_json, bb_ontology)


def test_bb2d_proto(bb_ontology):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR, "test_scene/scene_01/bounding_box_2d/CAMERA_01/15569195938203752.json"
    )
    bb2d_list = BoundingBox2DAnnotationList.load(scenes_dataset_json, bb_ontology)
    ouput_proto = bb2d_list.to_proto()
    assert ouput_proto.__sizeof__() == 80


def test_bb2d_save(bb_ontology):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR, "test_scene/scene_01/bounding_box_2d/CAMERA_01/15569195938203752.json"
    )
    bb2d_list = BoundingBox2DAnnotationList.load(scenes_dataset_json, bb_ontology)
    bb2d_list.save(".")
    filepath = "./219f294449ae6c01f62b6fa1d68949ee2b51ebd8.json"
    assert os.path.exists(filepath)
    os.remove(filepath)
