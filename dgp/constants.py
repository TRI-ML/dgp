# Copyright 2021-2022 Toyota Research Institute. All rights reserved.
"""
Common DGP constants. Constants here only depened on dgp.proto to avoid circular imports. 
"""
from collections import OrderedDict

from dgp.proto import annotations_pb2, dataset_pb2, features_pb2

# String identifiers for dataset splits
DATASET_SPLIT_NAME_TO_KEY = OrderedDict({k.lower(): v for k, v in dataset_pb2.DatasetSplit.items()})
DATASET_SPLIT_KEY_TO_NAME = OrderedDict({v: k for k, v in DATASET_SPLIT_NAME_TO_KEY.items()})

# Provide mapping from annotation types to proto IDs, (i.e. 'bounding_box_2d': annotations_pb2.BOUNDING_BOX_2D, 'depth': annotations_pb2.DEPTH).
ANNOTATION_KEY_TO_TYPE_ID = OrderedDict({k.lower(): v for k, v in annotations_pb2.AnnotationType.items()})
ANNOTATION_TYPE_ID_TO_KEY = OrderedDict({v: k for k, v in ANNOTATION_KEY_TO_TYPE_ID.items()})
# String identifiers for annotation types
ALL_ANNOTATION_TYPES = tuple(ANNOTATION_KEY_TO_TYPE_ID.keys())

# Provide supported annotations for each type of datum
DATUM_TYPE_TO_SUPPORTED_ANNOTATION_TYPE = OrderedDict({
    'image': [
        'bounding_box_2d', 'bounding_box_3d', 'semantic_segmentation_2d', 'instance_segmentation_2d', 'depth',
        'surface_normals_2d', 'motion_vectors_2d', 'key_point_2d', 'key_line_2d', 'agent_behavior'
    ],
    'point_cloud': [
        'bounding_box_3d', 'semantic_segmentation_3d', 'instance_segmentation_3d', 'surface_normals_3d',
        'motion_vectors_3d', 'agent_behavior'
    ]
})

# Provide mapping from feature types to proto IDs, (i.e. 'agent_3d': features_pb2.AGENT_3D,
# 'ego_intention': features_pb2.EGO_INTENTION).
FEATURE_KEY_TO_TYPE_ID = OrderedDict({k.lower(): v for k, v in features_pb2.FeatureType.items()})
FEATURE_TYPE_ID_TO_KEY = OrderedDict({v: k for k, v in FEATURE_KEY_TO_TYPE_ID.items()})
# String identifiers for feature types
ALL_FEATURE_TYPES = tuple(FEATURE_KEY_TO_TYPE_ID.keys())
