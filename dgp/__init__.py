# Copyright 2019 Toyota Research Institute. All rights reserved.
import os

__version__ = '0.1'

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
BOUNDING_BOX_2D_FOLDER = 'bounding_box_2d'
BOUNDING_BOX_3D_FOLDER = 'bounding_box_3d'
SEMANTIC_SEGMENTATION_2D_FOLDER = 'semantic_segmentation_2d'
SEMANTIC_SEGMENTATION_3D_FOLDER = 'semantic_segmentation_3d'
INSTANCE_SEGMENTATION_2D_FOLDER = 'instance_segmentation_2d'
INSTANCE_SEGMENTATION_3D_FOLDER = 'instance_segmentation_3d'
DEPTH_FOLDER = 'depth'

# Scene Directory structure constants
AUTOLABEL_FOLDER = 'autolabels'
CALIBRATION_FOLDER = 'calibration'
ONTOLOGY_FOLDER = 'ontology'
SCENE_JSON_FILENAME = 'scene.json'

# DGP file naming conventions
TRI_DGP_SCENE_DATASET_JSON_NAME = "scene_dataset_v{version}.json"
TRI_DGP_SCENE_JSON_NAME = "scene_{scene_hash}.json"
ANNOTATION_FILE_NAME = '{image_content_hash}_{annotation_content_hash}.json'
