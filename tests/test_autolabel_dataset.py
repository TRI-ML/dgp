# Copyright 2021-2022 Toyota Research Institute. All rights reserved.
import os
import unittest
from shutil import copytree, rmtree

from dgp import (
    AUTOLABEL_FOLDER,
    AUTOLABEL_SCENE_JSON_NAME,
    BOUNDING_BOX_2D_FOLDER,
    BOUNDING_BOX_3D_FOLDER,
    DEPTH_FOLDER,
    INSTANCE_SEGMENTATION_2D_FOLDER,
    INSTANCE_SEGMENTATION_3D_FOLDER,
    ONTOLOGY_FOLDER,
    SEMANTIC_SEGMENTATION_2D_FOLDER,
    SEMANTIC_SEGMENTATION_3D_FOLDER,
)
from dgp.constants import ANNOTATION_KEY_TO_TYPE_ID
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.proto.scene_pb2 import Scene
from dgp.utils.protobuf import open_pbobject, save_pbobject_as_json
from tests import TEST_DATA_DIR

ANNOTATION_TYPE_ID_TO_FOLDER = {
    'bounding_box_2d': BOUNDING_BOX_2D_FOLDER,
    'bounding_box_3d': BOUNDING_BOX_3D_FOLDER,
    'depth': DEPTH_FOLDER,
    'semantic_segmentation_2d': SEMANTIC_SEGMENTATION_2D_FOLDER,
    'semantic_segmentation_3d': SEMANTIC_SEGMENTATION_3D_FOLDER,
    'instance_segmentation_2d': INSTANCE_SEGMENTATION_2D_FOLDER,
    'instance_segmentation_3d': INSTANCE_SEGMENTATION_3D_FOLDER,
}


def clone_scene_as_autolabel(dataset_root, autolabel_root, autolabel_model, autolabel_type):
    """Helper function to copy a scene directory for use as autolabels

    Parameters
    ----------
    dataset_root: str
        Path to dataset root folder containing scene folders

    autolabel_root: str
        Path to where autolabels should be stored

    autolabel_model: str
        Name of autolabel model

    autolabel_type: str
        Annotation type i.e., 'bounding_box_3d', 'depth' etc
    """
    # For each scene dir, copy the requested annotation into the new autolabel folder
    autolabel_dirs = []
    for scene_dir in os.listdir(dataset_root):
        if not os.path.isdir(os.path.join(dataset_root, scene_dir)):
            continue

        # Clear any existing folders
        autolabel_scene_dir = os.path.join(autolabel_root, scene_dir, AUTOLABEL_FOLDER, autolabel_model)
        if os.path.exists(autolabel_scene_dir):
            rmtree(autolabel_scene_dir)

        os.makedirs(autolabel_scene_dir, exist_ok=True)

        full_scene_dir = os.path.join(dataset_root, scene_dir)
        autolabel_dirs.append(autolabel_scene_dir)

        for scene_json in os.listdir(full_scene_dir):
            if 'scene' in scene_json and scene_json.endswith('json'):
                base_scene = open_pbobject(os.path.join(full_scene_dir, scene_json), Scene)
                for i in range(len(base_scene.data)):
                    name = base_scene.data[i].id.name
                    datum = base_scene.data[i].datum
                    datum_type = datum.WhichOneof('datum_oneof')
                    datum_value = getattr(datum, datum_type)  # This is datum.image or datum.point_cloud etc
                    annotation_type_id = ANNOTATION_KEY_TO_TYPE_ID[autolabel_type]
                    current_annotation = datum_value.annotations[annotation_type_id]
                    # NOTE: this should not actually change the path but is included for clarity
                    datum_value.annotations[annotation_type_id] = os.path.join(
                        ANNOTATION_TYPE_ID_TO_FOLDER[autolabel_type], name, os.path.basename(current_annotation)
                    )

                save_pbobject_as_json(base_scene, os.path.join(autolabel_scene_dir, AUTOLABEL_SCENE_JSON_NAME))
                # Only modify one scene.json, test scene should not contain multiple scene.jsons
                break

        ontology_dir = os.path.join(autolabel_scene_dir, ONTOLOGY_FOLDER)
        if os.path.exists(ontology_dir):
            rmtree(ontology_dir)
        copytree(os.path.join(full_scene_dir, ONTOLOGY_FOLDER), ontology_dir)

        annotation_dir = os.path.join(autolabel_scene_dir, ANNOTATION_TYPE_ID_TO_FOLDER[autolabel_type])
        if os.path.exists(annotation_dir):
            rmtree(annotation_dir)
        copytree(os.path.join(full_scene_dir, ANNOTATION_TYPE_ID_TO_FOLDER[autolabel_type]), annotation_dir)

    return autolabel_dirs


class TestAutolabelDataset(unittest.TestCase):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")

    def test_autolabels_default_root(self):
        """Test that we can load autolabels stored in scene/autolabels/model/autolabel_type folders"""

        scenes_dataset_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
        autolabel_model = 'test-model'
        autolabel_annotation = 'bounding_box_3d'
        requested_autolabels = (f'{autolabel_model}/{autolabel_annotation}', )
        dataset_root = os.path.dirname(scenes_dataset_json)
        autolabel_root = dataset_root

        clone_scene_as_autolabel(dataset_root, autolabel_root, autolabel_model, autolabel_annotation)

        dataset = SynchronizedSceneDataset(
            scenes_dataset_json,
            split='train',
            datum_names=['LIDAR'],
            forward_context=1,
            backward_context=1,
            requested_annotations=('bounding_box_3d', ),
            requested_autolabels=requested_autolabels,
            autolabel_root=autolabel_root,
            use_diskcache=False,
        )

        assert len(dataset) == 2

        for context in dataset:
            for sample in context:
                lidar = sample[0]
                assert lidar['bounding_box_3d'] == lidar[requested_autolabels[0]]

    def test_autolabels_custom_root(self):
        """Test that we can load autolabels using autolabel_root"""

        scenes_dataset_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
        autolabel_model = 'test-model'
        autolabel_annotation = 'bounding_box_3d'
        requested_autolabels = (f'{autolabel_model}/{autolabel_annotation}', )
        dataset_root = os.path.dirname(scenes_dataset_json)
        autolabel_root = os.path.join(self.DGP_TEST_DATASET_DIR, 'autolabel_root')

        clone_scene_as_autolabel(dataset_root, autolabel_root, autolabel_model, autolabel_annotation)

        dataset = SynchronizedSceneDataset(
            scenes_dataset_json,
            split='train',
            datum_names=['LIDAR'],
            forward_context=1,
            backward_context=1,
            requested_annotations=('bounding_box_3d', ),
            requested_autolabels=requested_autolabels,
            autolabel_root=autolabel_root,
            use_diskcache=False,
        )

        assert len(dataset) == 2

        for context in dataset:
            for sample in context:
                lidar = sample[0]
                assert lidar['bounding_box_3d'] == lidar[requested_autolabels[0]]

    def test_autolabels_missing_files(self):
        """Test that skip missing data can be used to skip missing autolabel scene dirs"""

        scenes_dataset_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
        autolabel_model = 'test-model'
        autolabel_annotation = 'bounding_box_3d'
        requested_autolabels = (f'{autolabel_model}/{autolabel_annotation}', )
        dataset_root = os.path.dirname(scenes_dataset_json)
        autolabel_root = os.path.join(self.DGP_TEST_DATASET_DIR, 'autolabel_root')

        autolabel_dirs = clone_scene_as_autolabel(dataset_root, autolabel_root, autolabel_model, autolabel_annotation)

        # remove a scene dir and check we can still load the data
        rmtree(autolabel_dirs[0])
        # Test skip missing data allows us to load the dataset
        dataset = SynchronizedSceneDataset(
            scenes_dataset_json,
            split='train',
            datum_names=['LIDAR'],
            forward_context=1,
            backward_context=1,
            requested_annotations=('bounding_box_3d', ),
            requested_autolabels=requested_autolabels,
            autolabel_root=autolabel_root,
            skip_missing_data=True,
            use_diskcache=False,
        )

        assert len(dataset) == 2

        for context in dataset:
            for sample in context:
                lidar = sample[0]
                autolab = lidar[requested_autolabels[0]]
                assert autolab is None or lidar['bounding_box_3d'] == autolab

    def test_only_annotated_datums(self):
        """Test that only annotated datums applies to autolabels also"""

        scenes_dataset_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
        autolabel_model = 'test-model'
        autolabel_annotation = 'bounding_box_3d'
        requested_autolabels = (f'{autolabel_model}/{autolabel_annotation}', )
        dataset_root = os.path.dirname(scenes_dataset_json)
        autolabel_root = os.path.join(self.DGP_TEST_DATASET_DIR, 'autolabel_root')

        autolabel_dirs = clone_scene_as_autolabel(dataset_root, autolabel_root, autolabel_model, autolabel_annotation)

        # remove a scene dir and check we can still load the data
        rmtree(autolabel_dirs[0])

        # Test that only annotated datums works
        dataset = SynchronizedSceneDataset(
            scenes_dataset_json,
            split='train',
            datum_names=['LIDAR'],
            forward_context=1,
            backward_context=1,
            requested_autolabels=requested_autolabels,
            autolabel_root=autolabel_root,
            only_annotated_datums=True,
            skip_missing_data=True,
            use_diskcache=False,
        )

        assert len(dataset) == 1
        for context in dataset:
            for sample in context:
                lidar = sample[0]
                assert lidar[requested_autolabels[0]] is not None


if __name__ == "__main__":
    unittest.main()
