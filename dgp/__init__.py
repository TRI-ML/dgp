# Copyright 2019-2021 Toyota Research Institute. All rights reserved.
import os
from collections import OrderedDict

from dgp.proto import annotations_pb2, dataset_pb2

__version__ = '1.0'

DGP_PATH = os.getenv('DGP_PATH', default=os.getenv('HOME'))
DGP_DATA_DIR = os.path.join(DGP_PATH, '.dgp')
DGP_CACHE_DIR = os.path.join(DGP_DATA_DIR, 'cache')
DGP_DATASETS_CACHE_DIR = os.path.join(DGP_DATA_DIR, 'datasets')

TRI_DGP_FOLDER_PREFIX = "dgp/"
TRI_RAW_FOLDER_PREFIX = "raw/"
TRI_DGP_JSON_PREFIX = "dataset_v"

# DGP Directory structure constants
RGB_FOLDER = 'rgb'
POINT_CLOUD_FOLDER = 'point_cloud'
RADAR_POINT_CLOUD_FOLDER = "radar_point_cloud"
BOUNDING_BOX_2D_FOLDER = 'bounding_box_2d'
BOUNDING_BOX_3D_FOLDER = 'bounding_box_3d'
SEMANTIC_SEGMENTATION_2D_FOLDER = 'semantic_segmentation_2d'
SEMANTIC_SEGMENTATION_3D_FOLDER = 'semantic_segmentation_3d'
INSTANCE_SEGMENTATION_2D_FOLDER = 'instance_segmentation_2d'
INSTANCE_SEGMENTATION_3D_FOLDER = 'instance_segmentation_3d'
DEPTH_FOLDER = 'depth'
EXTRA_DATA_FOLDER = "extra_data"

# Scene Directory structure constants
AUTOLABEL_FOLDER = 'autolabels'
CALIBRATION_FOLDER = 'calibration'
ONTOLOGY_FOLDER = 'ontology'
SCENE_JSON_FILENAME = 'scene.json'

# DGP file naming conventions
TRI_DGP_SCENE_DATASET_JSON_NAME = "scene_dataset_v{version}.json"
TRI_DGP_SCENE_JSON_NAME = "scene_{scene_hash}.json"
ANNOTATION_FILE_NAME = '{image_content_hash}_{annotation_content_hash}.json'

# String identifiers for dataset splits
DATASET_SPLIT_NAME_TO_KEY = OrderedDict({k.lower(): v for k, v in dataset_pb2.DatasetSplit.items()})
DATASET_SPLIT_KEY_TO_NAME = OrderedDict({v: k for k, v in DATASET_SPLIT_NAME_TO_KEY.items()})

# Provide mapping from annotation types to proto IDs, (i.e. 'bounding_box_2d': annotations_pb2.BOUNDING_BOX_2D, 'depth': annotations_pb2.DEPTH).
ANNOTATION_KEY_TO_TYPE_ID = OrderedDict({k.lower(): v for k, v in annotations_pb2.AnnotationType.items()})
ANNOTATION_TYPE_ID_TO_KEY = OrderedDict({v: k for k, v in ANNOTATION_KEY_TO_TYPE_ID.items()})
# String identifiers for annotation types
ALL_ANNOTATION_TYPES = tuple(ANNOTATION_KEY_TO_TYPE_ID.keys())
