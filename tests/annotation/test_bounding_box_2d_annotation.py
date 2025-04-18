import os

import numpy as np
import pytest

from dgp.annotations.bounding_box_2d_annotation import (
    BoundingBox2DAnnotationList,
)
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
    expected_output = [
        [0.0, 581.0, 185.0, 652.0],
        [149.0, 564.0, 299.0, 641.0],
        [313.0, 573.0, 425.0, 627.0],
        [1334.0, 531.0, 1655.0, 745.0],
        [1559.0, 577.0, 1850.0, 726.0],
        [717.0, 367.0, 1494.0, 1021.0],
        [1232.0, 516.0, 1460.0, 681.0],
        [1218.0, 553.0, 1362.0, 656.0],
        [1199.0, 562.0, 1324.0, 643.0],
        [1185.0, 555.0, 1292.0, 645.0],
        [1175.0, 564.0, 1264.0, 633.0],
        [1158.0, 549.0, 1232.0, 607.0],
        [317.0, 556.0, 490.0, 652.0],
        [196.0, 538.0, 395.0, 642.0],
        [551.0, 525.0, 722.0, 610.0],
        [831.0, 550.0, 917.0, 611.0],
        [1213.0, 553.0, 1268.0, 595.0],
        [493.0, 563.0, 654.0, 640.0],
        [1470.0, 569.0, 1684.0, 700.0],
        [396.0, 569.0, 502.0, 604.0],
        [323.0, 569.0, 429.0, 602.0],
        [289.0, 568.0, 393.0, 601.0],
        [264.0, 568.0, 367.0, 601.0],
        [533.0, 574.0, 547.0, 608.0],
    ]
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR, "test_scene/scene_01/bounding_box_2d/CAMERA_01/15569195938203752.json"
    )
    bb2d_list = BoundingBox2DAnnotationList.load(scenes_dataset_json, bb_ontology)
    assert (bb2d_list.ltrb == expected_output).all()


def test_bb2d_proto(bb_ontology):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR, "test_scene/scene_01/bounding_box_2d/CAMERA_01/15569195938203752.json"
    )
    bb2d_list = BoundingBox2DAnnotationList.load(scenes_dataset_json, bb_ontology)
    ouput_proto = bb2d_list.to_proto()
    assert ouput_proto.__sizeof__() in {64, 80}


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
