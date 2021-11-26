import os
import pytest
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from tests import TEST_DATA_DIR

@pytest.fixture
def ontology():
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

def test_ontology_name_to_id(ontology):
    assert len(ontology.name_to_id)==10

def test_ontology_num_classes(ontology):
    assert ontology.num_classes==10
    
def test_ontology_class_ids(ontology):
    assert len(ontology.class_ids)==10
    
def test_ontology_name_to_id(ontology):
    assert len(ontology.name_to_id)
    
def test_ontology_id_to_name(ontology):
    assert len(ontology.id_to_name)
    
def test_ontology_colormap(ontology):
    assert len(ontology.colormap)
    
def test_ontology_isthing(ontology):
    assert len(ontology.isthing)
    
def test_ontology_hexdigest(ontology):
    assert len(ontology.hexdigest)