# Copyright 2023 Toyota Motor Corporation.  All rights reserved.
import logging
import os
import time
from collections import defaultdict
from typing import Any, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from dgp.datasets.synchronized_dataset import (
    SynchronizedScene,
    SynchronizedSceneDataset,
)
from dgp.utils.camera import Camera
from dgp.utils.structures.bounding_box_3d import BoundingBox3D

ANNOTATIONS_3D = "bounding_box_3d"
ANNOTATIONS_2D = "bounding_box_2d"


def render_bounding_box_3d_to_2d(bbox_3d: BoundingBox3D, camera: Camera) -> np.ndarray:
    """Render the bounding box from 3d to 2d to get the centroid.
    Parameters
    ----------
    bbox_3d: BoundingBox3D
        3D bounding box (cuboid) that is centered at `pose` with extent `sizes.
    camera: dgp.utils.camera.Camera
        Camera used to render the bounding box.
    Returns
    ----------
    centroid: np.ndarray
        Centroid in image plane.
    Raises
    ------
    TypeError
        Raised if camera is not an instance of Camera.
    """
    if not isinstance(camera, Camera):
        raise TypeError("`camera` should be of type Camera")
    if (bbox_3d.corners[:, 2] <= 0).any():
        return None
    # Get the centroid in image plane.
    return camera.project(np.vstack([bbox_3d.pose.tvec, bbox_3d.pose.tvec, bbox_3d.pose.tvec])).astype(np.int32)


def render_bounding_boxes_3d_of_lidars(
    dataset: SynchronizedSceneDataset,
    camera_datum_names: Optional[list[str]] = None,
    lidar_datum_names: Optional[list[str]] = None,
    max_num_items: Optional[int] = None,
) -> defaultdict[defaultdict[list]]:
    """Load and project 3D bounding boxes to 2d image with given dataset, camera_datum_name and lidar_datum_names.
    Parameters
    ----------
    dataset: SynchronizedSceneDataset
        A DGP dataset.
    camera_datum_names: Optional[list[str]]
        List of camera names.
        If None, use all the cameras available in the DGP dataset.
    lidar_datum_names: Optional[list[str]]
        List of lidar names.
        If None, use all the lidars available in the DGP dataset.
    max_num_items: Optional[int]
        If not None, then show only up to this number of items. This is useful for debugging a large dataset.
        Default: None.
    Returns
    -------
    bbox_2d_from_3d: defaultdict[defaultdict[list]]
        a dictionary with key is the camera name, value is a dictionary whose key is class_name of bounding_box_3d,
        value is list of (bbox_3d, centroid_2d)
    """
    ontology = dataset.dataset_metadata.ontology_table.get(ANNOTATIONS_3D, None)
    id_to_name = ontology.contiguous_id_to_name if ontology else dict()

    if camera_datum_names is None:
        camera_datum_names = sorted(dataset.list_datum_names_available_in_all_scenes(datum_type="image"))
    if lidar_datum_names is None:
        lidar_datum_names = sorted(dataset.list_datum_names_available_in_all_scenes(datum_type="point_cloud"))

    bbox_2d_from_3d = defaultdict(lambda: defaultdict(list))

    st = time.time()
    logging_span = 200
    for idx, context in enumerate(dataset):
        # no temporal context
        context = context[0]
        if idx == max_num_items:
            break
        if idx % logging_span == 0:
            logging.info(f"2D:Frame {idx + 1} of {len(dataset)} in {time.time() - st:.2f}s.")

        context = {datum["datum_name"]: datum for datum in context}
        camera_datums = [(camera_datum_name, context[camera_datum_name]) for camera_datum_name in camera_datum_names]

        for camera_name, camera_datum in camera_datums:
            # Render 3D bboxes
            if ANNOTATIONS_3D in camera_datum:
                for bbox_3d in camera_datum[ANNOTATIONS_3D]:
                    class_name = id_to_name[bbox_3d.class_id]
                    center_2d = render_bounding_box_3d_to_2d(bbox_3d, Camera(K=camera_datum["intrinsics"]))
                    bbox_2d_from_3d[camera_name][class_name].append((bbox_3d, center_2d))
    return bbox_2d_from_3d


def render_bounding_boxes_2d_of_cameras(
    dataset: SynchronizedSceneDataset,
    camera_datum_names: Optional[list[str]] = None,
    max_num_items: Optional[int] = None,
) -> defaultdict[defaultdict[list]]:
    """Load 2d bounding boxes with given dataset, camera_datum_name.
    Parameters
    ----------
    dataset: SynchronizedSceneDataset
        A DGP dataset.
    camera_datum_names: Optional[list[str]]
        List of camera names.
        If None, use all the cameras available in the DGP dataset.
    max_num_items: Optional[int]
        If not None, then show only up to this number of items. This is useful for debugging a large dataset.
        Default: None.
    Returns
    -------
    bboxes_2d: defaultdict[defaultdict[list]]
        a dictionary with key is the camera name, value is a dictionary whose key is class_name of bounding_box_2d,
        value is list of boxes as (N, 4) np.ndarray in format ([left, top, width, height])
    """
    bboxes_2d = defaultdict(lambda: defaultdict(list))
    if len(dataset):
        if max_num_items is not None:
            if max_num_items > len(dataset):
                logging.info(
                    "`max_num_items` is reduced to the dataset size, from {:d} to {:d}".format(
                        max_num_items, len(dataset)
                    )
                )
                max_num_items = len(dataset)

        ontology_table = dataset.dataset_metadata.ontology_table

        if ANNOTATIONS_2D in ontology_table:
            ontology = ontology_table[ANNOTATIONS_2D]

        if camera_datum_names is None:
            camera_datum_names = sorted(dataset.list_datum_names_available_in_all_scenes(datum_type="image"))

        st = time.time()
        logging_span = 200
        for idx, datums in enumerate(dataset):
            # no temporal context
            datums = datums[0]
            if idx == max_num_items:
                break
            if idx % logging_span == 0:
                logging.info(f"2D:Frame {idx + 1} of {len(dataset)} in {time.time() - st:.2f}s.")
            datums = {datum["datum_name"]: datum for datum in datums}
            camera_datums = [(camera_datum_name, datums[camera_datum_name]) for camera_datum_name in camera_datum_names]
            # Visualize bounding box 2d
            if ANNOTATIONS_2D in dataset.requested_annotations:
                for camera_datum_name, camera_datum in camera_datums:
                    for bbox in camera_datum[ANNOTATIONS_2D]:
                        bboxes_2d[camera_datum_name][ontology.contiguous_id_to_name[bbox.class_id]].append(bbox.ltwh)
    return bboxes_2d


def associate_lidar_and_camera_2d_bboxes(
    bboxes_from_camera: list[np.ndarray], bboxes_from_lidar: list[Tuple[np.ndarray, np.ndarray]]
) -> list[Tuple[Any, Any, Any]]:
    """Associate 3d bounding boxes and 2d bounding boxes to the same object by checking
       whether the projected centroid of 3d bounding box is inside the 2d bounding box or not.
       Limitation:
       1. Several 3d objects could project to the same place in an image.
       2. The 2d convex hull of a 3d projection will typically not be contained inside the tight axis aligned 2d box.
       3. One single large 2d box will lead to everything being associated. If ego is near a truck or a bus which fills
          the view, this won't work anymore.
       Future work:
       To be more robust, ideas such as measure of 2d-3d similarity and doing a bipartite matching is suggested.
    Parameters
    ----------
    bboxes_from_camera: list[np.ndarray]
        A list of 2d bounding boxes.
    bboxes_from_lidar: list[Tuple[np.ndarray, np.ndarray]]
        A list of Tuple (centroid in 2d image, 3d bounding box).
    Returns
    -------
    associated: list[Tuple[bounding_box_2d, bounding_box_3d, centroid_2d]]
        a dictionary with key is the camera name, value is a dictionary whose key is class_name of bounding_box_2d,
        value is list of boxes as (N, 4) np.ndarray in format ([left, top, width, height])
    """
    associated = []
    if bboxes_from_camera and bboxes_from_lidar:
        for bbox_camera in bboxes_from_camera:
            l, t, w, h = bbox_camera
            for bbox_lidar, bbox_centroid_2d in bboxes_from_lidar:
                if bbox_centroid_2d is None:
                    continue
                bbox_centroid_x, bbox_centroid_y = bbox_centroid_2d[0]
                if (
                    bbox_centroid_x >= l and bbox_centroid_x < l + w and bbox_centroid_y >= t
                    and bbox_centroid_y < t + h
                ):
                    associated.append((bbox_camera, bbox_lidar, bbox_centroid_2d[0]))
                    break
    return associated


def associate_3d_and_2d_annotations(
    dataset: SynchronizedSceneDataset,
    ontology_name_mapper: dict,
    camera_datum_names: Optional[list[str]],
    lidar_datum_names: Optional[list[str]],
    max_num_items: Optional[int],
) -> defaultdict[defaultdict[list[Tuple]]]:
    """Associate 3d bounding boxes and 2d bounding boxes to the same object with given dataset.
    Parameters
    ----------
    dataset: SynchronizedSceneDataset
        A DGP dataset.
    ontology_name_mapper: dict
        Map the class names from bounding_box_2d to bounding_box_3d if the class names are different.
        eg: {'Pedestrian': 'Person','Car': 'Car'}.
    camera_datum_names: Optional[list[str]]
        List of camera names.
        If None, use all the cameras available in the DGP dataset.
    lidar_datum_names: Optional[list[str]]
        List of lidar names.
        If None, use all the lidars available in the DGP dataset.
    max_num_items: Optional[int]
        If not None, then show only up to this number of items. This is useful for debugging a large dataset.
        Default: None.
    Returns
    -------
    associated_bboxes: defaultdict[defaultdict[list[Tuple]]]
        a dictionary with key is the camera name, value is a dictionary whose key is class_name of bounding_box_2d,
        value is list of Tuple [bounding_box_2d, bounding_box_3d, centroid_2d]
    """
    bboxes_2d_from_lidars = render_bounding_boxes_3d_of_lidars(
        dataset, camera_datum_names, lidar_datum_names, max_num_items
    )
    bboxes_2d_from_cameras = render_bounding_boxes_2d_of_cameras(dataset, camera_datum_names, max_num_items)
    associated_bboxes = defaultdict(lambda: defaultdict(list))
    for camera_name in camera_datum_names:
        bboxes_2d_from_lidar = bboxes_2d_from_lidars[camera_name]
        bboxes_2d_from_camera = bboxes_2d_from_cameras[camera_name]
        for name_2d, name_3d in ontology_name_mapper.items():
            logging.info("{}: Associate {} to {}".format(camera_name, name_2d, name_3d))
            bboxes_from_lidar = None
            if name_3d in bboxes_2d_from_lidar:
                bboxes_from_lidar = bboxes_2d_from_lidar[name_3d]
            bboxes_from_camera = None
            if name_2d in bboxes_2d_from_camera:
                bboxes_from_camera = bboxes_2d_from_camera[name_2d]
            associated_bboxes[camera_name][name_2d] = associate_lidar_and_camera_2d_bboxes(
                bboxes_from_camera, bboxes_from_lidar
            )
    return associated_bboxes


def associate_3d_and_2d_annotations_scene(
    scene_json: str,
    ontology_name_mapper: dict,
    camera_datum_names: Optional[list[str]],
    lidar_datum_names: Optional[list[str]],
    max_num_items: Optional[int],
) -> defaultdict[defaultdict[list[Tuple]]]:
    """Associate 3d bounding boxes and 2d bounding boxes to the same object with given scene.
    Parameters
    ----------
    scene_json: str
        Full path to the scene json.
    ontology_name_mapper: dict
        Map the class names from bounding_box_2d to bounding_box_3d if the class names are different.
        eg: {'Pedestrian': 'Person','Car': 'Car'}.
    camera_datum_names: Optional[list[str]]
        List of camera names.
        If None, use all the cameras available in the DGP dataset.
    lidar_datum_names: Optional[list[str]]
        List of lidar names.
        If None, use all the lidars available in the DGP dataset.
    max_num_items: Optional[int]
        If not None, then show only up to this number of items. This is useful for debugging a large dataset.
        Default: None.
    Returns
    -------
    associated_bboxes: defaultdict[defaultdict[list[Tuple]]]
        a dictionary with key is the camera name, value is a dictionary whose key is class_name of bounding_box_2d,
        value is list of Tuple [bounding_box_2d, bounding_box_3d, centroid_2d]
    """
    datum_names = camera_datum_names + lidar_datum_names
    dataset = SynchronizedScene(
        scene_json,
        datum_names=datum_names,
        requested_annotations=[ANNOTATIONS_2D, ANNOTATIONS_3D],
        only_annotated_datums=True,
    )
    return associate_3d_and_2d_annotations(
        dataset, ontology_name_mapper, camera_datum_names, lidar_datum_names, max_num_items
    )


def associate_3d_and_2d_annotations_dataset(
    scenes_dataset_json: str,
    ontology_name_mapper: dict,
    camera_datum_names: Optional[list[str]],
    lidar_datum_names: Optional[list[str]],
    max_num_items: Optional[int],
) -> defaultdict[defaultdict[list[Tuple]]]:
    """Associate 3d bounding boxes and 2d bounding boxes to the same object with given DGP dataset.
    Parameters
    ----------
    scenes_dataset_json: str
        Full path to the dataset scene json.
    ontology_name_mapper: dict
        Map the class names from bounding_box_2d to bounding_box_3d if the class names are different.
        eg: {'Pedestrian': 'Person','Car': 'Car'}.
    camera_datum_names: Optional[list[str]]
        List of camera names.
        If None, use all the cameras available in the DGP dataset.
    lidar_datum_names: Optional[list[str]]
        List of lidar names.
        If None, use all the lidars available in the DGP dataset.
    max_num_items: Optional[int]
        If not None, then show only up to this number of items. This is useful for debugging a large dataset.
        Default: None.
    Returns
    -------
    associated_bboxes: defaultdict[defaultdict[list[Tuple]]]
        a dictionary with key is the camera name, value is a dictionary whose key is class_name of bounding_box_2d,
        value is list of Tuple [bounding_box_2d, bounding_box_3d, centroid_2d]
    """
    # Merge Lidar and Camera datum names.
    if camera_datum_names and lidar_datum_names:
        datum_names = camera_datum_names + lidar_datum_names
    else:
        datum_names = None
    dataset = SynchronizedSceneDataset(
        scenes_dataset_json,
        datum_names=datum_names,
        requested_annotations=[ANNOTATIONS_2D, ANNOTATIONS_3D],
        only_annotated_datums=True,
    )
    associated_bboxes = associate_3d_and_2d_annotations(
        dataset, ontology_name_mapper, camera_datum_names, lidar_datum_names, max_num_items
    )
    return associated_bboxes


def draw_bounding_box_2d_distance_distribution(
    scenes_dataset_json: str,
    ontology_name_mapper: dict,
    output_dir: str,
    camera_datum_names: Optional[list[str]],
    lidar_datum_names: Optional[list[str]],
    max_num_items: Optional[int],
):
    """Draw the distance's distributution histogram of 2d bounding boxes by associating bounding_box_3d of the same object.
    Parameters
    ----------
    scenes_dataset_json: str
        Full path to the dataset scene json.
    ontology_name_mapper: dict
        Map the class names from bounding_box_2d to bounding_box_3d if the class names are different.
        eg: {'Pedestrian': 'Person','Car': 'Car'}.
    output_dir: str
        Path to save the histogram picture.
    camera_datum_names: Optional[list[str]]
        List of camera names.
        If None, use all the cameras available in the DGP dataset.
    lidar_datum_names: Optional[list[str]]
        List of lidar names.
        If None, use all the lidars available in the DGP dataset.
    max_num_items: Optional[int]
        If not None, then show only up to this number of items. This is useful for debugging a large dataset.
        Default: None.
    """
    associated_bboxes = associate_3d_and_2d_annotations_dataset(
        scenes_dataset_json, ontology_name_mapper, camera_datum_names, lidar_datum_names, max_num_items
    )
    os.makedirs(output_dir, exist_ok=True)
    # Summarize statistics per camera per class over all scenes.
    for camera_name in camera_datum_names:
        for name_2d, _ in ontology_name_mapper.items():
            logging.info("Summarizing class {}".format(name_2d))
            summarize_3d_statistics_per_class(associated_bboxes[camera_name][name_2d], output_dir, camera_name, name_2d)


def summarize_3d_statistics_per_class(
    associated_bboxes: list[Tuple], output_dir: str, camera_name: str, class_name: str
):
    """Accumulate distances of the associated bounding boxes and draw the histogram.
    Parameters
    ----------
    associated_bboxes: list[Tuple]
        A list of Tuple [bounding_box_2d, bounding_box_3d, centroid_2d].
    output_dir: str
        Path to save the histogram picture.
    camera_name: str
        camera name.
    class_name: str
        Class name.
    """
    dist = []
    for _, bbox_lidar, __ in associated_bboxes:
        dist.append(int(np.linalg.norm(bbox_lidar.pose.tvec[:2])))
    draw_hist(dist, output_dir, xlable="Dist", title=f"dist_{camera_name}_{class_name}")


def draw_hist(data: list, output_dir: str, xlable: str, title: str):
    """Draw the histogram of given data.
    Parameters
    ----------
    data: list
        A list of int.
    output_dir: str
        Path to save the histogram picture.
    xlable: str
        The label name of x.
    title: str
        The tile of the picture.
    """
    data = np.array(data)
    min_dist = -20
    max_dist = 100
    dist_bin = 10
    bins = np.arange(min_dist, max_dist, dist_bin)  # fixed bin size
    plt.hist(data, bins=bins)
    plt.title(f"Distribution {title}(fixed bin size)")
    plt.xlabel(f"variable {xlable} (bin size = {dist_bin})")
    plt.ylabel("count")
    plt.savefig(os.path.join(output_dir, f"histogram_{title}.png"))
