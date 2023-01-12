# Copyright 2020-2021 Toyota Research Institute. All rights reserved.

from collections import OrderedDict

from dgp.annotations.ontology import (
    AgentBehaviorOntology,
    BoundingBoxOntology,
    InstanceSegmentationOntology,
    KeyLineOntology,
    KeyPointOntology,
    Ontology,
    SemanticSegmentationOntology,
)

# Ontologies are used in Annotations, hence import ordering
from dgp.annotations.base_annotation import Annotation  # isort:skip
from dgp.annotations.bounding_box_2d_annotation import BoundingBox2DAnnotationList  # isort:skip
from dgp.annotations.bounding_box_3d_annotation import BoundingBox3DAnnotationList  # isort:skip
from dgp.annotations.panoptic_segmentation_2d_annotation import PanopticSegmentation2DAnnotation  # isort:skip
from dgp.annotations.semantic_segmentation_2d_annotation import SemanticSegmentation2DAnnotation  # isort:skip
from dgp.annotations.key_line_2d_annotation import KeyLine2DAnnotationList  # isort:skip
from dgp.annotations.key_line_3d_annotation import KeyLine3DAnnotationList  # isort:skip
from dgp.annotations.key_point_2d_annotation import KeyPoint2DAnnotationList  # isort:skip
from dgp.annotations.key_point_3d_annotation import KeyPoint3DAnnotationList  # isort:skip
from dgp.annotations.depth_annotation import DenseDepthAnnotation  # isort:skip

# Ontology handlers for each annotation type
ONTOLOGY_REGISTRY = {
    "bounding_box_2d": BoundingBoxOntology,
    "bounding_box_3d": BoundingBoxOntology,
    "semantic_segmentation_2d": SemanticSegmentationOntology,
    "semantic_segmentation_3d": SemanticSegmentationOntology,
    "instance_segmentation_2d": InstanceSegmentationOntology,
    "instance_segmentation_3d": InstanceSegmentationOntology,
    "key_point_2d": KeyPointOntology,
    "key_point_3d": KeyPointOntology,
    "key_line_2d": KeyLineOntology,
    "key_line_3d": KeyLineOntology,
    "agent_behavior": AgentBehaviorOntology,
    "depth": None,
    "surface_normals_2d": None,
    "surface_normals_3d": None,
    "motion_vectors_2d": None,
    "motion_vectors_3d": None
}

# Annotation objects for each annotation type
ANNOTATION_REGISTRY = {
    "bounding_box_2d": BoundingBox2DAnnotationList,
    "bounding_box_3d": BoundingBox3DAnnotationList,
    "semantic_segmentation_2d": SemanticSegmentation2DAnnotation,
    "instance_segmentation_2d": PanopticSegmentation2DAnnotation,
    "key_point_2d": KeyPoint2DAnnotationList,
    "key_point_3d": KeyPoint3DAnnotationList,
    "key_line_2d": KeyLine2DAnnotationList,
    "key_line_3d": KeyLine3DAnnotationList,
    "depth": DenseDepthAnnotation
}

# Annotation groups for each annotation type: 2d/3d
ANNOTATION_TYPE_TO_ANNOTATION_GROUP = {
    "bounding_box_2d": "2d",
    "bounding_box_3d": "3d",
    "semantic_segmentation_2d": "2d",
    "semantic_segmentation_3d": "3d",
    "instance_segmentation_2d": "2d",
    "instance_segmentation_3d": "3d",
    "surface_normals_2d": "2d",
    "surface_normals_3d": "3d",
    "motion_vectors_2d": "2d",
    "motion_vectors_3d": "3d",
    "key_point_2d": "2d",
    "key_line_2d": "2d",
    "key_point_3d": "3d",
    "key_line_3d": "3d",
    "depth": "2d"
}
