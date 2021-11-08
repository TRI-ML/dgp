# Copyright 2021 Toyota Research Institute. All rights reserved.
import os
import unittest

from dgp.proto import annotations_pb2
from dgp.proto.ontology_pb2 import Ontology, OntologyItem
from dgp.proto.scene_pb2 import Scene
from dgp.utils.protobuf import open_pbobject
from dgp.utils.statistics import get_scene_class_statistics


class TestStats(unittest.TestCase):
    def test_class_count_cuboid(self):
        # test scene_01
        scene_json = 'tests/data/dgp/test_scene/scene_01/scene_a8dc5ed1da0923563f85ea129f0e0a83e7fe1867.json'
        scene_dir = os.path.dirname(scene_json)
        scene = open_pbobject(scene_json, Scene)
        annotation_enum = annotations_pb2.BOUNDING_BOX_3D
        ontology = None

        class_stats = get_scene_class_statistics(scene, scene_dir, annotation_enum, ontology)

        self.assertTrue(class_stats['Bicycle'] == 0)
        self.assertTrue(class_stats['Bus/RV/Caravan'] == 6)
        self.assertTrue(class_stats['Car'] == 278)
        self.assertTrue(class_stats['Motorcycle'] == 0)
        self.assertTrue(class_stats['Person'] == 2)
        self.assertTrue(class_stats['Towed Object'] == 0)
        self.assertTrue(class_stats['Trailer'] == 0)
        self.assertTrue(class_stats['Train'] == 0)
        self.assertTrue(class_stats['Truck'] == 0)
        self.assertTrue(class_stats['Wheeled Slow'] == 0)

        # test scene_02
        scene_json = 'tests/data/dgp/test_scene/scene_02/scene_fe9f29d3bde25d182dcf88caf1011acd8cc13624.json'
        scene_dir = os.path.dirname(scene_json)
        scene = open_pbobject(scene_json, Scene)
        annotation_enum = annotations_pb2.BOUNDING_BOX_3D
        ontology = None

        class_stats = get_scene_class_statistics(scene, scene_dir, annotation_enum, ontology)
        self.assertTrue(class_stats['Bicycle'] == 0)
        self.assertTrue(class_stats['Bus/RV/Caravan'] == 0)
        self.assertTrue(class_stats['Car'] == 33)
        self.assertTrue(class_stats['Motorcycle'] == 0)
        self.assertTrue(class_stats['Person'] == 0)
        self.assertTrue(class_stats['Towed Object'] == 0)
        self.assertTrue(class_stats['Trailer'] == 0)
        self.assertTrue(class_stats['Train'] == 0)
        self.assertTrue(class_stats['Truck'] == 6)
        self.assertTrue(class_stats['Wheeled Slow'] == 0)

        # test assertion
        wrong_ontology = Ontology(
            items=[
                OntologyItem(name='dummy', id=0, color=OntologyItem.Color(r=220, b=60, g=20), isthing=True),
            ]
        )
        self.assertRaises(AssertionError, get_scene_class_statistics, scene, scene_dir, annotation_enum, wrong_ontology)

    def test_class_count_panoptic(self):
        # test scene_01
        scene_json = 'tests/data/dgp/test_scene/scene_01/scene_a8dc5ed1da0923563f85ea129f0e0a83e7fe1867.json'
        scene_dir = os.path.dirname(scene_json)
        scene = open_pbobject(scene_json, Scene)
        annotation_enum = annotations_pb2.BOUNDING_BOX_2D
        ontology = None

        class_stats = get_scene_class_statistics(scene, scene_dir, annotation_enum, ontology)

        self.assertTrue(class_stats['Bicycle'] == 0)
        self.assertTrue(class_stats['Bus/RV/Caravan'] == 3)
        self.assertTrue(class_stats['Car'] == 152)
        self.assertTrue(class_stats['Motorcycle'] == 0)
        self.assertTrue(class_stats['Person'] == 4)
        self.assertTrue(class_stats['Towed Object'] == 0)
        self.assertTrue(class_stats['Trailer'] == 0)
        self.assertTrue(class_stats['Train'] == 0)
        self.assertTrue(class_stats['Truck'] == 0)
        self.assertTrue(class_stats['Wheeled Slow'] == 0)

        # test scene_02
        scene_json = 'tests/data/dgp/test_scene/scene_02/scene_fe9f29d3bde25d182dcf88caf1011acd8cc13624.json'
        scene_dir = os.path.dirname(scene_json)
        scene = open_pbobject(scene_json, Scene)
        annotation_enum = annotations_pb2.BOUNDING_BOX_2D
        ontology = None

        class_stats = get_scene_class_statistics(scene, scene_dir, annotation_enum, ontology)

        self.assertTrue(class_stats['Bicycle'] == 0)
        self.assertTrue(class_stats['Bus/RV/Caravan'] == 0)
        self.assertTrue(class_stats['Car'] == 33)
        self.assertTrue(class_stats['Motorcycle'] == 0)
        self.assertTrue(class_stats['Person'] == 0)
        self.assertTrue(class_stats['Towed Object'] == 0)
        self.assertTrue(class_stats['Trailer'] == 0)
        self.assertTrue(class_stats['Train'] == 0)
        self.assertTrue(class_stats['Truck'] == 6)
        self.assertTrue(class_stats['Wheeled Slow'] == 0)

        # test assertion
        wrong_ontology = Ontology(
            items=[
                OntologyItem(name='dummy', id=0, color=OntologyItem.Color(r=220, b=60, g=20), isthing=True),
            ]
        )
        self.assertRaises(AssertionError, get_scene_class_statistics, scene, scene_dir, annotation_enum, wrong_ontology)


if __name__ == "__main__":
    unittest.main()
