#!/usr/bin/env python
# Copyright 2022 Woven Planet NA. All rights reserved.
"""dgp2wicker command line interface
"""
import logging
import os
import sys
from functools import partial
from typing import Any, Dict, List

import click
from dgp2wicker.ingest import ingest_dgp_to_wicker

from dgp.annotations.camera_transforms import ScaleAffineTransform
from dgp.annotations.transforms import AddLidarCuboidPoints


class AddLidarCuboidPointsContext(AddLidarCuboidPoints):
    """Add Lidar Points but applied to samples not datums"""
    def __call__(self, sample: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        new_sample = []
        for datum in sample:
            if datum['datum_type'] == 'point_cloud' and 'bounding_box_3d' in datum:
                if datum['bounding_box_3d'] is not None:
                    datum = super().__call__(datum)
            new_sample.append(datum)
        return new_sample


class ScaleImages(ScaleAffineTransform):
    """Scale Transform but applied to samples not datums"""
    def __call__(self, sample: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        new_sample = []
        for datum in sample:
            if datum['datum_type'] == 'image' and 'rgb' in datum:
                datum = super().__call__(datum)
            new_sample.append(datum)
        return new_sample


@click.group()
@click.version_option()
def cli():
    logging.getLogger('dgp2widker').setLevel(level=logging.INFO)
    logging.getLogger('py4j').setLevel(level=logging.CRITICAL)
    logging.getLogger('botocore').setLevel(logging.CRITICAL)
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    logging.getLogger('PIL').setLevel(logging.CRITICAL)


@cli.command(name='ingest')
@click.option("--scene-dataset-json", required=True, help="Path to DGP Dataset JSON")
@click.option("--wicker-dataset-name", required=True, default=None, help="Name of dataset in Wicker")
@click.option("--wicker-dataset-version", required=True, help="Version of dataset in Wicker")
@click.option("--datum-names", required=True, help="List of datum names")
@click.option("--requested-annotations", help="List of annotation types")
@click.option("--only-annotated-datums", is_flag=True, help="Apply only annotated datums")
@click.option("--max-num-scenes", required=False, default=None, help="The maximum number of scenes to process")
@click.option("--max-len", required=False, default=1000, help="The maximum number of samples per scene")
@click.option("--chunk-size", required=False, default=1000, help="The number of samples per chunk")
@click.option("--skip-camera-cuboids", is_flag=True, help="If True, skip cuboids for non lidar datums")
@click.option("--num-partitions", required=False, default=None, help="Number of scene partitions")
@click.option("--num-repartitions", required=False, default=None, help="Number of sample partitions")
@click.option("--is-pd", is_flag=True, help="If true, process the dataset with ParallelDomainScene")
@click.option("--data-uri", required=False, default=None, help="Alternate location for scene data")
@click.option("--add-lidar-points", is_flag=True, help="Add lidar point count to lidar cuboids")
@click.option("--half-size-images", is_flag=True, help="Resize image datums to half size")
@click.option("--alternate-scene-uri", required=False, default=None, help="Alternate scene locaiton to sync")
def ingest(
    scene_dataset_json,
    wicker_dataset_name,
    wicker_dataset_version,
    datum_names,
    requested_annotations,
    only_annotated_datums,
    max_num_scenes,
    max_len,
    chunk_size,
    skip_camera_cuboids,
    num_partitions,
    num_repartitions,
    is_pd,
    data_uri,
    add_lidar_points,
    half_size_images,
    alternate_scene_uri,
):
    datum_names = [x.strip() for x in datum_names.split(',')]
    requested_annotations = [x.strip() for x in requested_annotations.split(',')] if requested_annotations else None
    dataset_kwargs = {
        'datum_names': datum_names,
        'requested_annotations': requested_annotations,
        'only_annotated_datums': only_annotated_datums,
    }

    pipeline = []
    if add_lidar_points:
        pipeline.append(AddLidarCuboidPointsContext())
    if half_size_images:
        pipeline.append(ScaleImages(s=.5))

    results = ingest_dgp_to_wicker(
        scene_dataset_json=scene_dataset_json,
        wicker_dataset_name=wicker_dataset_name,
        wicker_dataset_version=wicker_dataset_version,
        dataset_kwargs=dataset_kwargs,
        spark_context=None,
        pipeline=pipeline,
        max_num_scenes=max_num_scenes,
        max_len=max_len,
        chunk_size=chunk_size,
        skip_camera_cuboids=skip_camera_cuboids,
        num_partitions=num_partitions,
        num_repartitions=num_repartitions,
        is_pd=is_pd,
        data_uri=data_uri,
        alternate_scene_uri=alternate_scene_uri,
    )

    print('Finished ingest!')
    print(results)


if __name__ == '__main__':
    cli()
