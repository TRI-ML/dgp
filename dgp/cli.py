#!/usr/bin/env python
# Copyright 2019-2021 Toyota Research Institute.  All rights reserved.
"""DGP command line interface
"""
import logging
import os
from functools import partial

import click

from dgp.annotations import ANNOTATION_TYPE_TO_ANNOTATION_GROUP
from dgp.constants import ALL_ANNOTATION_TYPES, DATASET_SPLIT_NAME_TO_KEY
from dgp.datasets.pd_dataset import ParallelDomainScene
from dgp.datasets.synchronized_dataset import SynchronizedScene
from dgp.proto.dataset_pb2 import SceneDataset
from dgp.utils.cli_utils import add_options
from dgp.utils.protobuf import open_pbobject
from dgp.utils.visualization_engine import (
    visualize_dataset_2d,
    visualize_dataset_3d,
)
from dgp.utils.visualization_utils import make_caption

VISUALIZE_OPTIONS = [
    click.option(
        "--annotations",
        "-a",
        type=click.Choice(ALL_ANNOTATION_TYPES),
        help="If specified, visualize subset corresponding to annotations. \
        Specify each desired annotation type separately, i.e. `-a bounding_box_3d -a semantic_segmentation_2d`",
        multiple=True
    ),
    click.option(
        "--camera-datum-names",
        "-c",
        required=True,
        help=
        "Camera datum names (case-sensitive). If specified, visualize corresponding camera datum. If not, visualize all. \
        Specify each camera datum names separately, i.e. `-c camera_05 -c camera_01`",
        multiple=True
    ),
    click.option(
        "--dataset-class",
        required=False,
        default='SynchronizedScene',
        type=click.Choice(['ParallelDomainScene', 'SynchronizedScene']),
        help="If specified, use corresponding dataset class. If not, use SynchronizedScene(Dataset)"
    ),
    click.option("--show-instance-id", is_flag=True, required=False, help="Show instance ID"),
    click.option("--max-num-items", required=False, help="Max num of samples to visualize"),
    click.option("--video-fps", default=10, help="Frame rate of generated videos"),
    click.option(
        "--dst-dir",
        required=False,
        help="Destination directory. \
        If given, then generate one video per scene and save in this directory. \
        If not, then show on X window"
    ),
    click.option("--verbose", "-v", is_flag=True, help="Verbose logging"),
    # 3D visualization options
    click.option(
        "--lidar_datum_names",
        "-l",
        required=False,
        default=['LIDAR'],
        help=
        "Lidar datum names (case-sensitive). If specified, visualize corresponding lidar datum. If not, visualize LIDAR. \
        Specify each lidar datum names separately, i.e. `-l lidar`",
        multiple=True
    ),
    click.option("--render-pointcloud", is_flag=True, required=False, help="Render projected pointcloud on images."),
    click.option(
        "--radar_datum_names",
        "-r",
        required=False,
        default=[],
        help=
        "Radar datum names (case-sensitive). If specified, visualize corresponding radar datum. If not, visualize RADAR. \
        Specify each radar datum names separately, i.e. `-r radar`",
        multiple=True
    ),
    click.option(
        "--render-radar-pointcloud", is_flag=True, required=False, help="Render projected radar pointcloud on images."
    ),
    click.option(
        "--render-raw", is_flag=True, required=False, help="Wether or not to render raw data without annotations."
    ),
]


@click.group()
@click.version_option()
def cli():
    logging.getLogger().setLevel(level=logging.INFO)


@cli.command(name='visualize-scene')
@add_options(options=VISUALIZE_OPTIONS)
@click.option("--scene-json", required=True, help="Path to Scene JSON")
def visualize_scene( # pylint: disable=missing-any-param-doc
    scene_json, annotations, camera_datum_names, dataset_class, show_instance_id, max_num_items, video_fps, dst_dir,
    verbose, lidar_datum_names, render_pointcloud, radar_datum_names, render_radar_pointcloud, render_raw
):
    """Parallelized visualizing of a scene.

    Example
    -------
    $ cli.py visualize-scene \
        --scene-json tests/data/dgp/test_scene/scene_01/scene_a8dc5ed1da0923563f85ea129f0e0a83e7fe1867.json \
        --dst-dir /mnt/fsx -a bounding_box_3d \
        -c camera_01
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Scene lands in directory based on scene directory name
    base_path = os.path.dirname(scene_json)
    if dst_dir is not None:
        video_path = os.path.basename(base_path) + '.avi'
        logging.info('Visualizing scene {} into {}'.format(os.path.basename(base_path), dst_dir))
    else:
        video_file = None

    scene_dataset_class = ParallelDomainScene if dataset_class == 'ParallelDomainScene' else SynchronizedScene

    # 2D visualization
    annotations_2d = tuple([a for a in annotations if ANNOTATION_TYPE_TO_ANNOTATION_GROUP[a] == '2d'])
    if annotations_2d:
        dataset = scene_dataset_class(
            scene_json,
            datum_names=camera_datum_names,
            requested_annotations=annotations_2d,
            only_annotated_datums=True
        )
        if len(dataset):
            if dst_dir is not None:
                os.makedirs(os.path.join(dst_dir, '2d'), exist_ok=True)
                video_file = os.path.join(dst_dir, '2d', video_path)
            visualize_dataset_2d(
                dataset,
                camera_datum_names=camera_datum_names,
                caption_fn=partial(make_caption, prefix=base_path),
                output_video_file=video_file,
                output_video_fps=video_fps,
                max_num_items=max_num_items,
                show_instance_id=show_instance_id
            )
            logging.info('Visualizing 2D annotation visualizations to {}'.format(video_file))
        else:
            logging.info(
                'Scene {} does not contain any of the requested datums {} annotated with {}. Skip 2d visualization.'.
                format(scene_json, camera_datum_names, annotations_2d)
            )
    # 3D visualization
    annotations_3d = tuple([a for a in annotations if ANNOTATION_TYPE_TO_ANNOTATION_GROUP[a] == '3d'])
    if annotations_3d or render_pointcloud or render_radar_pointcloud:
        datum_names = list(camera_datum_names) + list(lidar_datum_names) + list(radar_datum_names)
        dataset = SynchronizedScene(
            scene_json, datum_names=datum_names, requested_annotations=annotations_3d, only_annotated_datums=True
        )
        if len(dataset):
            if dst_dir is not None:
                os.makedirs(os.path.join(dst_dir, '3d'), exist_ok=True)
                video_file = os.path.join(dst_dir, '3d', video_path)
            visualize_dataset_3d(
                dataset,
                camera_datum_names=camera_datum_names,
                lidar_datum_names=lidar_datum_names,
                caption_fn=partial(make_caption, prefix=base_path),
                output_video_file=video_file,
                output_video_fps=video_fps,
                render_pointcloud_on_images=render_pointcloud,
                max_num_items=max_num_items,
                show_instance_id_on_bev=show_instance_id,
                radar_datum_names=radar_datum_names,
                render_radar_pointcloud_on_images=render_radar_pointcloud
            )
            logging.info('Visualizing 3D annotation visualizations to {}'.format(video_file))
        else:
            logging.info(
                'Scene {} does not contain any of the requested samples {} annotated with {}. Skip 3d visualization.'.
                format(scene_json, datum_names, annotations_3d)
            )
    if render_raw:
        datum_names = list(camera_datum_names) + list(lidar_datum_names) + list(radar_datum_names)
        dataset = SynchronizedScene(scene_json, datum_names=datum_names, only_annotated_datums=False)
        if len(dataset):
            if dst_dir is not None:
                os.makedirs(os.path.join(dst_dir, 'raw'), exist_ok=True)
                video_file = os.path.join(dst_dir, 'raw', video_path)
            visualize_dataset_3d(
                dataset,
                camera_datum_names=camera_datum_names,
                lidar_datum_names=lidar_datum_names,
                caption_fn=partial(make_caption, prefix=base_path),
                output_video_file=video_file,
                output_video_fps=video_fps,
                render_pointcloud_on_images=render_pointcloud,
                max_num_items=max_num_items,
                show_instance_id_on_bev=False,
                radar_datum_names=radar_datum_names,
                render_radar_pointcloud_on_images=render_radar_pointcloud
            )
            logging.info('Visualizing raw sensory data visualizations to {}'.format(video_file))
        else:
            logging.info(
                'Scene {} does not contain any of the requested samples {}. Skip visualization.'.format(
                    scene_json, datum_names
                )
            )


@cli.command(name='visualize-scenes')
@click.option("--scene-dataset-json", required=True, help="Path to SceneDataset JSON")
@click.option(
    "--split",
    type=click.Choice(['train', 'val', 'test', 'train_overfit']),
    required=True,
    help="Dataset split to be fetched."
)
@add_options(options=VISUALIZE_OPTIONS)
def visualize_scenes( # pylint: disable=missing-any-param-doc
    scene_dataset_json, split, annotations, camera_datum_names, dataset_class, show_instance_id, max_num_items,
    video_fps, dst_dir, verbose, lidar_datum_names, render_pointcloud, radar_datum_names, render_radar_pointcloud,
    render_raw
):
    """Parallelized visualizing of scene dataset.

    Example
    -------
    $ cli.py visualize-scenes \
        --scene-dataset-json tests/data/dgp/test_scene/scene_dataset_v1.0.json \
        --dst-dir /mnt/fsx -a bounding_box_3d \
        --split train \
        -c camera_01
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Load dataset
    dataset = open_pbobject(scene_dataset_json, pb_class=SceneDataset)

    # Dataset lands in directory based on scene dataset name
    if dst_dir is not None:
        dataset_directory = os.path.join(dst_dir, os.path.basename(scene_dataset_json).split('.')[0])
        os.makedirs(dataset_directory, exist_ok=True)
        logging.info('Visualizing dataset into {}'.format(dataset_directory))
    else:
        dataset_directory = None
    scene_jsons = dataset.scene_splits[DATASET_SPLIT_NAME_TO_KEY[split]].filenames
    for scene_json in scene_jsons:
        scene_json = os.path.join(os.path.dirname(scene_dataset_json), scene_json)
        visualize_scene.callback(
            scene_json,
            annotations=annotations,
            camera_datum_names=camera_datum_names,
            dataset_class=dataset_class,
            show_instance_id=show_instance_id,
            max_num_items=max_num_items,
            video_fps=video_fps,
            dst_dir=dataset_directory,
            verbose=verbose,
            lidar_datum_names=lidar_datum_names,
            render_pointcloud=render_pointcloud,
            radar_datum_names=radar_datum_names,
            render_radar_pointcloud=render_radar_pointcloud,
            render_raw=render_raw
        )


if __name__ == '__main__':
    cli()
