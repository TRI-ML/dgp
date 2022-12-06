import os

import numpy as np
import pytest

from dgp.annotations.key_line_3d_annotation import KeyLine3DAnnotationList
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.utils.structures.key_line_3d import KeyLine3D
from tests import TEST_DATA_DIR


def get_ontology_kl(scene_dataset_json, annotation_type):
    dataset = SynchronizedSceneDataset(
        scene_dataset_json,
        split='train',
        datum_names=['lcm_25tm'],
        backward_context=0,
        requested_annotations=("key_line_3d", )
    )
    return dataset.dataset_metadata.ontology_table.get(annotation_type, None)


@pytest.fixture
def kl_ontology():
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(DGP_TEST_DATASET_DIR, "key_line_3d", "scene_dataset.json")
    return get_ontology_kl(scene_dataset_json=scenes_dataset_json, annotation_type="key_line_3d")


def test_kl3d_annotation(kl_ontology):
    keylines = [KeyLine3D(np.array([[i + j, i + 1, i + 2] for i in range(5)], dtype=np.float32)) for j in range(5)]
    annotation_list = KeyLine3DAnnotationList(kl_ontology, keylines)
    assert len(annotation_list.xyz) == 5


def test_kl3d_load(kl_ontology):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    expected_output = "ac354"
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR,
        "key_line_3d/scene_000000/key_line_3d/lcm_25tm/000000000000000005_21e2436af96fb6388eb0c64cc029cfdc928a3e95.json"
    )
    kl3d_list = KeyLine3DAnnotationList.load(scenes_dataset_json, kl_ontology)
    assert kl3d_list.hexdigest[0:5] == expected_output


def test_kl3d_proto(kl_ontology):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR,
        "key_line_3d/scene_000000/key_line_3d/lcm_25tm/000000000000000005_21e2436af96fb6388eb0c64cc029cfdc928a3e95.json"
    )
    kl3d_list = KeyLine3DAnnotationList.load(scenes_dataset_json, kl_ontology)
    output_proto = kl3d_list.to_proto()
    assert output_proto.__sizeof__() == 80


def test_kl3d_save(kl_ontology):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR,
        "key_line_3d/scene_000000/key_line_3d/lcm_25tm/000000000000000005_21e2436af96fb6388eb0c64cc029cfdc928a3e95.json"
    )
    kl3d_list = KeyLine3DAnnotationList.load(scenes_dataset_json, kl_ontology)
    kl3d_list.save(".")
    filepath = "./ac35449091ebdd374aaa743be74794db561ec86a.json"
    assert os.path.exists(filepath)
    os.remove(filepath)
