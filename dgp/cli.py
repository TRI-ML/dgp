#!/usr/bin/env python
# Copyright 2019-2020 Toyota Research Institute.  All rights reserved.
"""DGP command line interface
"""
import glob
import itertools
import os
import sys
from multiprocessing import Pool, cpu_count

import click

from dgp.proto.dataset_pb2 import SceneDataset
from dgp.utils.aws import (convert_uri_to_bucket_path,
                           parallel_upload_s3_objects)
from dgp.utils.dataset_conversion import MergeSceneDatasetGen
from dgp.utils.protobuf import open_pbobject


@click.group()
@click.version_option()
def cli():
    pass


@cli.command(name="upload-scenes")
@click.option(
    "--scene-dataset-json",
    required=True,
    help="Path to a local scene dataset .json file i.e. /mnt/scene_dataset_v1.2.json"
)
@click.option(
    "--s3-dst-dir", required=True, help="Prefix for uploaded scenes"
)
def upload_scenes(scene_dataset_json, s3_dst_dir):
    """Parallelized upload for scenes from a scene dataset JSON.
    NOTE: This tool only verifies the presence of a scene, not the validity any of its contents.
    """
    bucket_name, s3_base_path = convert_uri_to_bucket_path(s3_dst_dir)
    dataset = open_pbobject(scene_dataset_json, SceneDataset)
    local_dataset_root = os.path.dirname(os.path.abspath(scene_dataset_json))
    if not dataset:
        print('Failed to parse dataset artifacts {}'.format(scene_dataset_json))
        sys.exit(0)

    scene_dirs = []
    for split in dataset.scene_splits.keys():
        scene_dirs.extend([
            os.path.join(local_dataset_root, os.path.dirname(filename))
            for filename in dataset.scene_splits[split].filenames
        ])

    # Make sure the scenes exist
    with Pool(cpu_count()) as proc:
        file_list = list(itertools.chain.from_iterable(proc.map(_get_scene_files, scene_dirs)))

    # Upload the scene JSON, too.
    file_list += [scene_dataset_json]
    print("Creating file manifest for S3 for {} files".format(len(file_list)))
    s3_file_list = [os.path.join(s3_base_path, os.path.relpath(_f, local_dataset_root)) for _f in file_list]
    print("Done. Uploading to S3.")

    parallel_upload_s3_objects(file_list, s3_file_list, bucket_name)


def _get_scene_files(scene):
    assert os.path.exists(scene), "Scene {} doesn't exist".format(scene)
    scene_files = glob.glob(os.path.join(scene, "**"), recursive=True)
    return [_f for _f in scene_files if os.path.isfile(_f)]


if __name__ == '__main__':
    cli()
