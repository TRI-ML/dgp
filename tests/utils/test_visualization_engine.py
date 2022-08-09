import json
import os

import cv2
import numpy as np

from dgp.datasets.synchronized_dataset import SynchronizedScene
from dgp.utils.visualization_engine import (
    visualize_dataset_2d,
    visualize_dataset_3d,
    visualize_dataset_sample_2d,
    visualize_dataset_sample_3d,
)
from tests import TEST_DATA_DIR


def dummy_caption(dataset, idx):
    return "SAMPLE"


def test_visualize_dataset_3d():
    '''
    Uses parametrized testing to run multiple cases for SynchronizedSceneDataset
    '''
    scene_json = os.path.join(
        TEST_DATA_DIR, "dgp", "test_scene/scene_01/scene_a8dc5ed1da0923563f85ea129f0e0a83e7fe1867.json"
    )
    filepath = "./test_3d_vis.avi"
    dataset = SynchronizedScene(
        scene_json,
        datum_names=['LIDAR', 'CAMERA_01', 'CAMERA_05', 'CAMERA_06'],
        forward_context=1,
        backward_context=1,
        requested_annotations=("bounding_box_2d", "bounding_box_3d")
    )
    visualize_dataset_3d(
        dataset=dataset,
        camera_datum_names=['CAMERA_01'],
        lidar_datum_names=['LIDAR'],
        radar_datum_names=[],
        output_video_file=filepath
    )
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_visualize_dataset_2d():
    '''
    Uses parametrized testing to run multiple cases for SynchronizedSceneDataset
    '''
    scene_json = os.path.join(
        TEST_DATA_DIR, "dgp", "test_scene/scene_01/scene_a8dc5ed1da0923563f85ea129f0e0a83e7fe1867.json"
    )
    filepath = "./test_2d_vis.avi"
    dataset = SynchronizedScene(
        scene_json,
        datum_names=['LIDAR', 'CAMERA_01', 'CAMERA_05', 'CAMERA_06'],
        forward_context=1,
        backward_context=1,
        requested_annotations=("bounding_box_2d", "bounding_box_3d")
    )

    visualize_dataset_2d(
        dataset=dataset, camera_datum_names=['CAMERA_01'], caption_fn=dummy_caption, output_video_file=filepath
    )
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_visualize_dataset_sample_3d():
    '''
    Uses parametrized testing to run multiple cases for SynchronizedSceneDataset
    '''
    scene_json = os.path.join(
        TEST_DATA_DIR, "dgp", "test_scene/scene_01/scene_a8dc5ed1da0923563f85ea129f0e0a83e7fe1867.json"
    )
    dataset = SynchronizedScene(
        scene_json,
        datum_names=['LIDAR', 'CAMERA_01', 'CAMERA_05', 'CAMERA_06'],
        forward_context=1,
        backward_context=1,
        requested_annotations=("bounding_box_2d", "bounding_box_3d")
    )

    result = visualize_dataset_sample_3d(dataset=dataset, scene_idx=0, sample_idx=0, camera_datum_names=['camera_05'])
    data = cv2.cvtColor(cv2.imread('tests/data/dgp/vis_output.png', cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert np.allclose(result['camera_05'], data, rtol=1e-3)
