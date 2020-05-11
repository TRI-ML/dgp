#!/usr/bin/env python
# Copyright 2019 Toyota Research Institute. All rights reserved.
"""Utilites to create DGP-compliant artifacts. The resulting dataset artifacts
should be tracked via source version control.
"""
import hashlib
import logging
import os
import subprocess

from dgp import CALIBRATION_FOLDER, ONTOLOGY_FOLDER
from dgp.utils.dataset import list_datum_files


def compute_scene_hash(scene, scene_root_path):
    """Utility to compute a deterministic hash of a scene pb object by hashing all datum,
    annotation, calibration, ontology files listed in scene pb object and return a hexadecimal hash.

    Parameters
    ----------
    scene: dgp.proto.scene_pb2.Scene
        Scene protobuf object

    scene_root_path: str
        Local root path to the scene

    Returns
    -------
    scene_hash: str
        hexadecimal hash of the scene.
    """
    # Identify image, point cloud and annotation files
    full_path = lambda x: os.path.join(scene_root_path, x)

    files = []
    for datum in scene.data:
        files.extend([full_path(_f) for _f in list_datum_files(datum)])

    calibration_file = os.path.join(CALIBRATION_FOLDER, '{}.json').format(scene.samples[0].calibration_key)
    files.append(full_path(calibration_file))

    for _, ontology_key in scene.ontologies.items():
        ontology_file = os.path.join(ONTOLOGY_FOLDER, '{}.json').format(ontology_key)
        files.append(full_path(ontology_file))

    # sort the files to get deterministic hash.
    files.sort()
    h = hashlib.md5()
    for f in files:
        h.update(open(f, "rb").read())
    scene_hash = h.hexdigest()

    logging.info('Scene hash: {}'.format(scene_hash))
    return scene_hash


def compress_and_hash_directory(directory, working_directory):
    """Utility to compress a directory and generate a SHA1 hash of it

    Parameters
    ----------
    directory: str
        Directory to hash

    working_directory: str
        The working directory (where the compressed directory is saved)

    Returns
    -------
    sha1: str
        sha1sum of compressed directory
    """
    # Time-agnostic archiving and sha generation
    print('Compressing and generating hash ...')
    ps = subprocess.Popen(
        'gzip -cnr --fast {} | sha1sum'.format(directory),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        cwd=working_directory
    )
    cmd_output, _ = ps.communicate()
    sha1, _ = cmd_output.split()
    return sha1.decode()
