# Copyright 2023 Woven by Toyota.  All rights reserved.
"""Unit test to merge dgp datasets."""
import logging
import os
import tempfile
import unittest
import unittest.mock as mock

import dgp.utils.render_3d_to_2d as render_engine
from tests import TEST_DATA_DIR

SCENE_JSON = 'scene_6245881cb04e9f71ae6de99064e771dfa370329d.json'
# The list of cameras' name
FRONT_CAMERA = ["CAMERA_21"]
# The list of Lidars' name
LIDAR = ["LIDAR"]
# Define the class names from bounding_box_2d to bounding_box_3d if the class names are different.
ONTOLOGY_NAME_MAPPER_2D_to_3D = {
    'Pedestrian': 'Person',
    'Car': 'Car',
}


class TestAssociateDGP3dto2d(unittest.TestCase):
    logging.getLogger().setLevel(logging.INFO)

    def test_associate_scene(self):
        """Verifies the target bounding box can be associated successfully."""
        # answer = gt_engine.associate_3d_and_2d_annotations_scene(scene_json=os.path.join(TEST_DATA_DIR,'dgp/test_scene/scene_03', SCENE_JSON))
        answer = render_engine.associate_3d_and_2d_annotations_scene(
            scene_json=os.path.join(TEST_DATA_DIR, 'dgp/associate_2d_to_3d_scene/scene_01', SCENE_JSON),
            ontology_name_mapper=ONTOLOGY_NAME_MAPPER_2D_to_3D,
            camera_datum_names=FRONT_CAMERA,
            lidar_datum_names=LIDAR,
            max_num_items=1
        )
        assert FRONT_CAMERA[0] in answer
        for class_name in ONTOLOGY_NAME_MAPPER_2D_to_3D.keys():
            if class_name == "Pedestrian":
                assert len(answer[FRONT_CAMERA[0]][class_name]) == 4
            elif class_name == "Car":
                assert len(answer[FRONT_CAMERA[0]][class_name]) == 0
            else:
                raise RuntimeError('Unexpected class_name {}'.format(class_name))


if __name__ == "__main__":
    unittest.main()
