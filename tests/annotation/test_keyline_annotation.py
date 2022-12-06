import os

import numpy as np
import pytest

from dgp.annotations.key_line_2d_annotation import KeyLine2DAnnotationList
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.utils.structures.key_line_2d import KeyLine2D
from tests import TEST_DATA_DIR


def get_ontology_kl(scene_dataset_json, annotation_type):
    dataset = SynchronizedSceneDataset(
        scene_dataset_json,
        split='train',
        datum_names=['locator'],
        backward_context=0,
        requested_annotations=("key_line_2d", )
    )
    return dataset.dataset_metadata.ontology_table.get(annotation_type, None)


@pytest.fixture
def kl_ontology():
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(DGP_TEST_DATASET_DIR, "key_line_2d", "scene_dataset.json")
    return get_ontology_kl(scene_dataset_json=scenes_dataset_json, annotation_type="key_line_2d")


def test_kl2d_annotation(kl_ontology):
    keylines = [KeyLine2D(np.array([[i + j, i + 5] for i in range(5)], dtype=np.float32)) for j in range(5)]
    annotation_list = KeyLine2DAnnotationList(kl_ontology, keylines)
    assert len(annotation_list.xy) == 5


def test_kl2d_load(kl_ontology):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    expected_output = "b67e1"
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR,
        "key_line_2d/scene_000000/key_line_2d/FCM_front/000000000000000005_23caffa10d786a53782f9530a6ad796db0eaea21.json"
    )
    kl2d_list = KeyLine2DAnnotationList.load(scenes_dataset_json, kl_ontology)
    assert kl2d_list.hexdigest[0:5] == expected_output


def test_kl2d_proto(kl_ontology):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR,
        "key_line_2d/scene_000000/key_line_2d/FCM_front/000000000000000005_23caffa10d786a53782f9530a6ad796db0eaea21.json"
    )
    kl2d_list = KeyLine2DAnnotationList.load(scenes_dataset_json, kl_ontology)
    ouput_proto = kl2d_list.to_proto()
    assert ouput_proto.__sizeof__() == 80


def test_kl2d_save(kl_ontology):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR,
        "key_line_2d/scene_000000/key_line_2d/FCM_front/000000000000000005_23caffa10d786a53782f9530a6ad796db0eaea21.json"
    )
    kl2d_list = KeyLine2DAnnotationList.load(scenes_dataset_json, kl_ontology)
    kl2d_list.save(".")
    filepath = "./b67e1088cbf3761cc1511cfb1a4c2c2f0d353316.json"
    assert os.path.exists(filepath)
    os.remove(filepath)
