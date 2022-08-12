import os

import pytest

from dgp.annotations.ontology import BoundingBoxOntology, Ontology
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from tests import TEST_DATA_DIR


def get_ontology(scene_dataset_json, annotation_type):
    dataset = SynchronizedSceneDataset(
        scene_dataset_json,
        split='train',
        datum_names=['camera_01', 'lidar'],
        backward_context=0,
        requested_annotations=("bounding_box_2d", "bounding_box_3d")
    )
    return dataset.dataset_metadata.ontology_table.get(annotation_type, None)


@pytest.fixture
def ontology():
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
    return get_ontology(scenes_dataset_json, "bounding_box_3d")


def test_ontology_name_to_id(ontology):
    assert len(ontology.name_to_id) == 10


def test_ontology_num_classes(ontology):
    assert ontology.num_classes == 10


def test_ontology_class_ids(ontology):
    assert len(ontology.class_ids) == 10


def test_ontology_id_to_name(ontology):
    assert len(ontology.id_to_name) == 10


def test_ontology_colormap(ontology):
    assert len(ontology.colormap) == 10


def test_ontology_isthing(ontology):
    assert len(ontology.isthing) == 10


def test_ontology_hexdigest(ontology):
    assert len(ontology.hexdigest) == 40


def test_ontology_load():
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR, "test_scene/scene_01/ontology/16322f7584a52ca979ed1c7049f17a7e420e86b1.json"
    )
    ontology = Ontology.load(scenes_dataset_json)
    assert ontology.hexdigest[:10] == "e7d7b0ffbc"


def test_bb_ontology_load():
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")
    scenes_dataset_json = os.path.join(
        DGP_TEST_DATASET_DIR, "test_scene/scene_01/ontology/16322f7584a52ca979ed1c7049f17a7e420e86b1.json"
    )
    ontology = BoundingBoxOntology.load(scenes_dataset_json)
    ontology.save(".")
    filepath = "./e7d7b0ffbc14565d33aaedc7d202eb270c9df254.json"
    assert os.path.exists(filepath)
    os.remove(filepath)
