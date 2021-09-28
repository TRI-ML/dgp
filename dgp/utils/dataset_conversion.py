# Copyright 2019-2021 Toyota Research Institute.  All rights reserved.
"""Useful utilities for pre-processing datasets"""
import datetime
import hashlib
import logging
import os
from functools import lru_cache
from multiprocessing import Pool, cpu_count

import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp
from PIL import ImageStat

from dgp.utils.protobuf import save_pbobject_as_json


@lru_cache(maxsize=None)
def _mkdir(dirname):
    """Smarter mkdir by caching on dirname to avoid redundant disk I/O."""
    os.makedirs(dirname, exist_ok=True)


def _write_point_cloud(filename, X):
    """Utility function for writing point clouds."""
    _mkdir(os.path.dirname(filename))
    np.savez_compressed(filename, data=X)


def _write_annotation(filename, annotation):
    """Utility function for writing 3d annotations."""
    _mkdir(os.path.dirname(filename))
    save_pbobject_as_json(annotation, filename)


def compute_image_statistics(image_list, image_open_fn):
    """Given a list of images (paths to files), return the per channel mean and stdev.
    Also returns a dictionary mapping filename to image size

    Parameters
    ----------
    image_list: list
        List of str of filepaths to images that can be opened by PIL

    image_open_fn: function
        Function to open image files in image_list, i.e. PIL.Image.open

    Returns
    -------
    global_mean: np.ndarray
        Channel wise mean of images over images in given list

    global_stdevr: np.ndarray
        Channel wise standard deviation of images in given list

    all_image_sizes: dict
        Dict mapping from filenames to image sizes (C, H, W)
    """

    valid_extensions = (".jpg", ".png", ".bmp", ".pgm", ".tiff")
    image_list = list(filter(lambda x: x.endswith(valid_extensions), image_list))

    num_processes = cpu_count()
    if len(image_list) < num_processes:
        num_processes = len(image_list)

    chunk_size = int(len(image_list) / num_processes)
    p = Pool(num_processes)

    image_stats_per_process = p.starmap(
        _get_image_stats,
        [(image_list[i:i + chunk_size], i, num_processes, image_open_fn) for i in range(0, len(image_list), chunk_size)]
    )

    global_mean, global_var, all_image_sizes = np.array([0., 0., 0.]), np.array([0., 0., 0.]), {}
    for means, variances, image_sizes in image_stats_per_process:
        global_mean += means
        global_var += variances
        all_image_sizes.update(image_sizes)

    global_mean /= len(image_list)
    global_stdev = np.sqrt(global_var / len(image_list))

    return list(global_mean), list(global_stdev), all_image_sizes


def rgb2id(color):
    """Function converting color to instance id.
    This function the the conversion function used by COCO dataset.
    We adapted this function from the panoptic api:
    ```https://github.com/cocodataset/panopticapi```
    Parameters
    ----------
    color: list
        A list of 3 channel color intensity in rgb.
    Returns
    -------
    id: int
        A single id encoded by the 3 channel color intensity.
    """
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.uint32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def bgr2id(color):
    """Function converting color to instance id.
    This function the the conversion function used by COCO dataset.
    We adapted this function from the panoptic api:
    ```https://github.com/cocodataset/panopticapi```
    Parameters
    ----------
    color: list
        A list of 3 channel color intensity in rgb.
    Returns
    -------
    id: int
        A single id encoded by the 3 channel color intensity.
    """
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.uint32)
        return color[:, :, 2] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 0]
    return int(color[2] + 256 * color[1] + 256 * 256 * color[0])


def _get_image_stats(image_sub_list, process_index, num_processes, image_open_fn):
    """Given a list of images, computes the mean, stddev, and image size"""
    mean, stddev, image_sizes = np.array([0., 0., 0.]), np.array([0., 0., 0.]), {}

    for i, image_file in enumerate(image_sub_list):
        image = image_open_fn(image_file)
        image_stats = ImageStat.Stat(image)
        channels = 3 if image.mode == 'RGB' else 1

        if channels == 3:
            mean += np.array(image_stats.mean)
            stddev += np.array(image_stats.stddev)

        width, height = image.size
        image_sizes.update({image_file: {"channels": channels, "height": height, "width": width}})

        if i % 100 == 0 and process_index == 0:
            logging.info("Completed {:d} images per process, with {:d} processes running".format(i, num_processes))

    return mean, stddev, image_sizes


def generate_uid_from_image(image):
    """Given a unique identifier, return the SHA1 hash hexdigest.
    Used for creating unique frame IDs

    Parameters
    ----------
    image: PIL.Image
        Image to be hashed

    Returns
    -------
    Hexdigest of image content
    """
    image = np.uint8(image.convert("RGB"))
    return hashlib.sha1(image).hexdigest()


def generate_uid_from_semantic_segmentation_2d_annotation(annotation):
    """Given a unique identifier, return the SHA1 hash hexdigest.

    Parameters
    ----------
    image: np.array
        semantic_segmentation_2d annotation to be hashed

    Returns
    -------
    Hexdigest of annotation content
    """
    if annotation.dtype != np.uint8:
        raise TypeError('`annotation` should be of type np.uint8')
    if len(annotation.shape) != 2:
        raise ValueError('`annotation` should be two-dimensional (one class ID per pixel)')
    return hashlib.sha1(annotation).hexdigest()


def generate_uid_from_instance_segmentation_2d_annotation(annotation):
    """Given a unique identifier, return the SHA1 hash hexdigest.

    Parameters
    ----------
    image: np.array
        instance_segmentation_2d annotation to be hashed

    Returns
    -------
    Hexdigest of annotation content
    """
    if annotation.dtype not in (np.uint16, np.uint32):
        raise TypeError('`annotation` should be of type np.uint16 or np.uint32')
    if len(annotation.shape) != 2:
        raise ValueError('`annotation` should be two-dimensional (one instance ID per pixel)')
    return hashlib.sha1(annotation).hexdigest()


def generate_uid_from_point_cloud(point_cloud):
    """Given a unique identifier, return the SHA1 hash hexdigest.
    Used for creating unique datum IDs

    Parameters
    ----------
    point_cloud: np.ndarray
        Point cloud to be hashed

    Returns
    -------
    Hexdigest of point cloud
    """
    return hashlib.sha1(point_cloud).hexdigest()


def parse_all_files_in_directory(directory):
    """Walk through subdirectories and pull all filenames

    Parameters
    ----------
    directory: str
        Path to directory to recurse

    Returns
    -------
    file_list: list
        Full paths to the files in all subdirectories
    """
    file_list = []
    for (dirpath, _, filenames) in os.walk(directory):
        file_list.extend([os.path.join(dirpath, file) for file in filenames])
    return file_list


def make_dir(dir_path, exist_ok=False):
    """Create a directory and catch the error if it already exists.

    Parameters
    ----------
    dir_path: str
        Directory to create
    exist_ok: bool
        Raise error if directory exists when this is set to False

    Returns
    -------
    dir_path: str
        Created directory name
    """
    os.makedirs(dir_path, exist_ok=exist_ok)
    return dir_path


def get_date():
    """Get today's date. In format yyyy-mm-dd"""
    return datetime.date.today().strftime("%Y-%m-%d")


def get_datetime_proto():
    """Returns current date time proto in UTC.

    Returns
    -------
    datetime_proto: google.protobuf.timestamp_pb2.Timestamp
        Current date time proto object in UTC.
    """
    timestamp = Timestamp()
    timestamp.GetCurrentTime()
    return timestamp
