# Copyright 2019-2021 Toyota Research Institute.  All rights reserved.
"""Useful utilities for pre-processing datasets"""
import datetime
import hashlib
import logging
import os
from collections import OrderedDict
from functools import lru_cache

import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp
from PIL import Image, ImageStat
from torch.multiprocessing import Pool, cpu_count

from dgp.proto import dataset_pb2
from dgp.proto.dataset_pb2 import SceneDataset
from dgp.utils.cloud.s3 import s3_copy
from dgp.utils.protobuf import (
    open_pbobject,
    open_remote_pb_object,
    save_pbobject_as_json,
)


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


def compute_image_statistics(image_list, image_open_fn, single_process=False, num_processes=None):
    """Given a list of images (paths to files), return the per channel mean and stdev.
    Also returns a dictionary mapping filename to image size

    Parameters
    ----------
    image_list: list
        List of str of filepaths to images that can be opened by PIL

    image_open_fn: function
        Function to open image files in image_list, i.e. PIL.Image.open

    single_process: bool
        If it's True, it gets image stats in single process for debugging. Defaults to False.

    num_processes: int
        A number of process in multiprocessing.Pool. If it's None, it uses cpu count.
        Defaults to None.

    Returns
    -------
    global_mean: np.ndarray
        Channel wise mean of images over images in given list

    global_stdevr: np.ndarray
        Channel wise standard deviation of images in given list

    all_image_sizes: dict
        Dict mapping from filenames to image sizes (C, H, W)

    Raises
    ------
    ValueError
        Raised if all the images are invalid extensions.
    """

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".pgm",
        ".tiff",
    )
    image_list = list(filter(lambda x: x.lower().endswith(valid_extensions), image_list))
    if not image_list:
        raise ValueError("There are no valid images for image statistics calculation.")

    if single_process:
        image_stats_per_process = []
        for idx, image in enumerate(image_list):
            image_stats_per_process.append(_get_image_stats([image], idx, len(image_list), image_open_fn))
    else:
        num_processes = num_processes or cpu_count()
        if len(image_list) < num_processes:
            num_processes = len(image_list)

        chunk_size = int(len(image_list) / num_processes)

        with Pool(num_processes) as p:
            image_stats_per_process = p.starmap(
                _get_image_stats, [(image_list[i:i + chunk_size], i, num_processes, image_open_fn)
                                   for i in range(0, len(image_list), chunk_size)]
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
    annotation: np.array
        semantic_segmentation_2d annotation to be hashed

    Returns
    -------
    Hexdigest of annotation content

    Raises
    ------
    TypeError
        Raised if annotation is not of type np.uint8.
    ValueError
        Raised if annotation is not 2-dimensional.
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
    annotation: np.array
        instance_segmentation_2d annotation to be hashed

    Returns
    -------
    Hexdigest of annotation content

    Raises
    ------
    TypeError
        Raised if annotation is not of type np.uint16 or np.uint23.
    ValueError
        Raised if annotation is not 2-dimensional.
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


class DatasetGen:
    DATASET_NAME = ""
    DESCRIPTION = ""
    PUBLIC = True
    EMAIL = "ml@tri.global"
    NAME_TO_ID = OrderedDict()
    ID_TO_NAME = OrderedDict()
    IS_THING = OrderedDict()
    COLORMAP = OrderedDict()

    def __init__(self, version, raw_path, local_output_path):
        """Parse raw data into the DGP format given directories.
        Parameters
        ----------
        version: int
            Version of the dataset
        raw_path: str
            Path to original files for dataset (should match public format)
        local_output_path: str, default: None
            Local path to save merged scene dataset JSON
        """
        self.version = version
        self.raw_path = raw_path
        self.scene_dataset_pb2 = SceneDataset()
        self.local_output_path = local_output_path
        self.ontologies = {}
        # Populate metadata and ontology
        self.populate_metadata()
        self.populate_ontologies()

    def convert(self):
        """Parse raw data into DGP format and return the dataset json.

        Returns
        -------
        dataset_json_path: str
            Dataset json path

        Raises
        ------
        NotImplementedError
            Unconditionally.
        """
        raise NotImplementedError

    def populate_metadata(self):
        """Populate boilerplate fields for dataset conversion. Statistics, size, etc...
        are recomputed during conversion

        Raises
        ------
        NotImplementedError
            Unconditionally.
        """
        raise NotImplementedError

    def populate_ontologies(self):
        """Populate ontologies' fields for dataset conversion.

        Raises
        ------
        NotImplementedError
            Unconditionally.
        """
        raise NotImplementedError

    def populate_statistics(self):
        """Compute dataset (image/point_cloud) statistics.

        Raises
        ------
        NotImplementedError
            Unconditionally.
        """
        raise NotImplementedError

    @staticmethod
    def open_image(filename):
        """Returns an image given a filename.

        Parameters
        ----------
        filename: str
            A pathname of an image to open.

        Returns
        -------
        PIL.Image
            Image that was opened.
        """
        return Image.open(filename)

    def write_dataset(self, upload=False):
        """Write the final scene dataset JSON.

        Parameters
        ----------
        upload: bool, optional
            If True, upload the dataset to the scene pool in s3. Default: False.

        Returns
        -------
        scene_dataset_json_path: str
            Path of the scene dataset JSON file created.
        """
        scene_dataset_json_path = os.path.join(
            self.local_output_path,
            '{}_v{}.json'.format(self.scene_dataset_pb2.metadata.name, self.scene_dataset_pb2.metadata.version)
        )
        save_pbobject_as_json(self.scene_dataset_pb2, scene_dataset_json_path)

        # Printing SceneDataset scene counts per split (post-merging)
        logging.info('-' * 80)
        logging.info(
            'Output SceneDataset {} has: {} train, {} val, {} test'.format(
                scene_dataset_json_path, len(self.scene_dataset_pb2.scene_splits[dataset_pb2.TRAIN].filenames),
                len(self.scene_dataset_pb2.scene_splits[dataset_pb2.VAL].filenames),
                len(self.scene_dataset_pb2.scene_splits[dataset_pb2.TEST].filenames)
            )
        )

        s3_path = os.path.join(
            self.scene_dataset_pb2.metadata.bucket_path.value, os.path.basename(scene_dataset_json_path)
        )
        if upload:
            s3_copy(scene_dataset_json_path, s3_path)

        else:
            logging.info(
                'Upload the DGP-compliant scene dataset JSON to s3 via `aws s3 cp --acl bucket-owner-full-control {} {}`'
                .format(scene_dataset_json_path, s3_path)
            )
        return scene_dataset_json_path


class MergeSceneDatasetGen:
    """Class to merge multiple scene dataset JSONs together.

    Parameters
    ----------
    scene_dataset_json_paths: list
        List of scene dataset JSON paths.

    description: str
        Description of the dataset.

    version: str
        Version of the dataset.

    local_output_path: str, default: None
        Local path to save merged scene dataset JSON

    bucket_path: str
        Where merged dataset should be saved to on s3 (defaults to that of the first dataset)

    name: str, default: None
        Name of the dataset

    email: str, default: None
        Email of the dataset creator

    """
    def __init__(
        self,
        scene_dataset_json_paths,
        description,
        version,
        local_output_path=None,
        bucket_path=None,
        name=None,
        email=None,
    ):
        assert len(scene_dataset_json_paths) >= 2, \
            'At least 2 scene datasets required in order to merge'
        self.scene_dataset_json_paths = scene_dataset_json_paths
        assert all([f.endswith('.json') for f in scene_dataset_json_paths]), 'Invalid scene dataset JSON.'
        self.scene_datasets = [
            open_remote_pb_object(f, SceneDataset) if f.startswith('s3://') else open_pbobject(f, SceneDataset)
            for f in scene_dataset_json_paths
        ]
        assert all([item is not None for item in self.scene_datasets]), \
            'Some of the scene dataset failed to load'

        self.local_output_path = local_output_path
        self.scene_dataset_pb2 = SceneDataset()

        # Copy Metadata.
        self.scene_dataset_pb2.metadata.description = description
        self.scene_dataset_pb2.metadata.version = version
        self.scene_dataset_pb2.metadata.origin = self.scene_datasets[0].metadata.origin
        self.scene_dataset_pb2.metadata.creation_date = get_date()

        # raw_path for merged dataset is combination of *bucket_paths* for input datasets
        self.scene_dataset_pb2.metadata.raw_path.value = ",".join(
            set([
                scene_dataset.metadata.bucket_path.value  # NOTE: *bucket_path*
                for scene_dataset in self.scene_datasets
                if scene_dataset.metadata.bucket_path.value
            ])
        )

        # bucket_path is specified by the user (and defaults to the bucket_path of the first dataset)
        assert bucket_path is None or bucket_path.startswith('s3://')
        self.scene_dataset_pb2.metadata.bucket_path.value = bucket_path or self.scene_datasets[
            0].metadata.bucket_path.value

        # Write in available annotation types
        ann_types = []
        for scene_dataset in self.scene_datasets:
            ann_types.extend(scene_dataset.metadata.available_annotation_types)
        self.scene_dataset_pb2.metadata.available_annotation_types.extend(set(ann_types))

        # Dataset name and creator email
        self.scene_dataset_pb2.metadata.name = name or self.scene_datasets[0].metadata.name
        self.scene_dataset_pb2.metadata.creator = email or self.scene_datasets[0].metadata.creator

    def merge_scene_split_files(self):
        """Aggregate scene JSON paths together.
        """
        # Look-up from scene directory to which dataset contains the latest scene_<sha1>.json for that scene
        scene_dir_to_dataset_index = {}

        # Printing individual SceneDataset scene counts per split
        for scene_dataset, scene_dataset_json_path in zip(self.scene_datasets, self.scene_dataset_json_paths):
            logging.info('-' * 80)
            logging.info(
                'SceneDataset {} has: {} train, {} val, {} test'.format(
                    scene_dataset_json_path, len(scene_dataset.scene_splits[dataset_pb2.TRAIN].filenames),
                    len(scene_dataset.scene_splits[dataset_pb2.VAL].filenames),
                    len(scene_dataset.scene_splits[dataset_pb2.TEST].filenames)
                )
            )

        # Make one pass throught SceneDataset's to find which dataset contains the latest scene_<sha1>.json
        # for each scene_dir. (For a given Scene, we assume that 'scene_<sha1>.json' files increase monotonically
        # in time from the first SceneDataset to the last one).
        # TODO: should we verify this with timestamps?
        for dataset_idx, scene_dataset in enumerate(self.scene_datasets):
            for split_id, scene_files in scene_dataset.scene_splits.items():

                # Iterate over scene_files and update scene_dir
                for scene_file in scene_files.filenames:
                    scene_dir = os.path.dirname(scene_file)
                    scene_dir_to_dataset_index[scene_dir] = dataset_idx

        # Make another pass to actually insert 'scene_dir/scene_<sha1>.json' paths into output SceneDataset
        for dataset_idx, scene_dataset in enumerate(self.scene_datasets):
            for split_id, scene_files in scene_dataset.scene_splits.items():
                for scene_file in scene_files.filenames:
                    if dataset_idx == scene_dir_to_dataset_index[os.path.dirname(scene_file)]:
                        self.scene_dataset_pb2.scene_splits[split_id].filenames.extend([scene_file])

    def write_dataset(self, upload=False):
        """Write the final scene dataset JSON.

        Parameters
        ----------
        upload: bool, optional
            If true, upload the dataset to the scene pool in s3. Default: False.

        Returns
        -------
        scene_dataset_json_path: str
            Path of the scene dataset JSON file created.
        """
        return DatasetGen.write_dataset(self, upload)
