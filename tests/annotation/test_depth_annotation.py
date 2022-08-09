import os

import numpy as np

from dgp.annotations.depth_annotation import DenseDepthAnnotation
from tests import TEST_DATA_DIR


def test_create_depth_annotation():
    depth_array = np.ones(shape=(10, 10))
    annotation = DenseDepthAnnotation(depth_array)
    annotation.render()
    assert annotation.hexdigest[:8] == "fe0e420a"


def test_depth_load():
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR, "test_scene/scene_01/point_cloud/LIDAR/15569195938203752.npz"
    )
    depth_annotations = DenseDepthAnnotation.load(scenes_dataset_json, None)
    assert depth_annotations.hexdigest[:16] == "1cd82e2fe409cb85"


def test_depth_save():
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR, "test_scene/scene_01/point_cloud/LIDAR/15569195938203752.npz"
    )
    depth_annotations = DenseDepthAnnotation.load(scenes_dataset_json, None)
    depth_annotations.save(".")
    filepath = "./1cd82e2fe409cb85b8d4e1e6ed5cc8f2a873aba2.npz"
    assert os.path.exists(filepath)
    os.remove(filepath)
