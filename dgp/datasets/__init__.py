# Copyright 2019 Toyota Research Institute. All rights reserved.

from collections import OrderedDict  # isort:skip

from dgp.proto import annotations_pb2  # isort:skip

# Provide mapping from annotation types to proto IDs, (i.e. 'bounding_box_2d': annotations_pb2.BOUNDING_BOX_2D, 'depth': annotations_pb2.DEPTH).
ANNOTATION_KEY_TO_TYPE_ID = OrderedDict({k.lower(): v for k, v in annotations_pb2.AnnotationType.items()})
ANNOTATION_TYPE_ID_TO_KEY = OrderedDict({v: k for k, v in ANNOTATION_KEY_TO_TYPE_ID.items()})

# String identifiers for annotation types
ALL_ANNOTATION_TYPES = tuple(ANNOTATION_KEY_TO_TYPE_ID.keys())

from dgp.datasets.base_dataset import BaseSceneDataset  # isort:skip
from dgp.datasets.synchronized_dataset import (  # isort:skip
    SynchronizedScene, SynchronizedSceneDataset
)
