# Copyright 2021 Toyota Research Institute. All rights reserved.
"""
Common DGP constants. Constants here only depened on dgp.proto to avoid circular imports. 
"""
from collections import OrderedDict

from dgp.proto import annotations_pb2, dataset_pb2

# String identifiers for dataset splits
DATASET_SPLIT_NAME_TO_KEY = OrderedDict({k.lower(): v for k, v in dataset_pb2.DatasetSplit.items()})
DATASET_SPLIT_KEY_TO_NAME = OrderedDict({v: k for k, v in DATASET_SPLIT_NAME_TO_KEY.items()})

# Provide mapping from annotation types to proto IDs, (i.e. 'bounding_box_2d': annotations_pb2.BOUNDING_BOX_2D, 'depth': annotations_pb2.DEPTH).
ANNOTATION_KEY_TO_TYPE_ID = OrderedDict({k.lower(): v for k, v in annotations_pb2.AnnotationType.items()})
ANNOTATION_TYPE_ID_TO_KEY = OrderedDict({v: k for k, v in ANNOTATION_KEY_TO_TYPE_ID.items()})
# String identifiers for annotation types
ALL_ANNOTATION_TYPES = tuple(ANNOTATION_KEY_TO_TYPE_ID.keys())
