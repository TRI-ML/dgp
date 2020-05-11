#!/usr/bin/env python
"""Script that visualizes the DGP dataset via dataset JSON or directory of scene.json(s).

Usage:
# Visualize dataset JSON
$ python python dgp/scripts/visualize_dataset.py \
      --dataset-json /mnt/scene_dataset_v0.json --split 'train' \
      --point-cloud --video

# Visualize directory of scene JSON(s)
$ python python dgp/scripts/visualize_dataset.py \
      --dataset-directory /mnt/dataset/ \
      --point-cloud --video
"""
import argparse
import logging

import cv2
import numpy as np

from dgp.datasets.synchronized_dataset import (SynchronizedDataset,
                                               SynchronizedScene,
                                               SynchronizedSceneDataset)
from dgp.utils import tqdm
from dgp.utils.aws import fetch_remote_scene
from dgp.utils.camera import Camera
from dgp.utils.visualization import (BEVImage, mosaic, print_status,
                                     render_bbox2d_on_image,
                                     render_pointcloud_on_image)


def render_bev(lidar_datums, id_to_name=None):
    """Render point clouds and bounding boxes onto a BEV image from a list of LiDAR datums.

    Parameters
    ----------
    lidar_datums: list of OrderedDict

        "point_cloud": list of np.array (dtype np.float64)
            List of point clouds. List is in order of `selected_datums` that
            are point clouds.

        "pose": Pose
            Ego pose of datum.

        "bounding_box_3d": list of BoundingBox3D
            3D Bounding boxes for this sample specified in this point cloud
            sensor's reference frame. (i.e. this provides the bounding box
            (B) in the sensor's (S) reference frame `box_SB`).

    id_to_name: OrderedDict, default: None
        Object class id (int) to name (str).

    Returns
    -------
    pts_world_coord: numpy.ndarray, shape=(n, 3)
        Accumulated point clouds from all lidar sensors in a same single coordinate frame.

    bev_image: BEVImage
        BEV Image.

    """
    # return None if the list is empty
    if not lidar_datums:
        return None, None

    all_pts_world_coord, all_pts_sensor_coord = [], []
    all_bboxes3d_sensor_coord = []
    all_classes = []
    # Merge point clouds in the same single coordinate frame
    for d in lidar_datums:
        # Get the point cloud in the lidar sensor's (S) reference frame
        p_WS = d['pose']
        pts_sensor_coord = d['point_cloud']
        # Move the points into the World (W) reference frame
        pts_world_coord = p_WS * pts_sensor_coord
        all_pts_sensor_coord.append(pts_sensor_coord)
        all_pts_world_coord.append(pts_world_coord)
        if 'bounding_box_3d' in d:
            bboxes3d_sensor_coord = d['bounding_box_3d']
            all_bboxes3d_sensor_coord.extend(bboxes3d_sensor_coord)
            if id_to_name:
                bboxes_classes = [id_to_name[cid] for cid in d['class_ids']]
                all_classes.extend(bboxes_classes)
    pts_world_coord = np.vstack(all_pts_world_coord)
    pts_sensor_coord = np.vstack(all_pts_sensor_coord)

    # Create BEV image
    bev_image = BEVImage(xaxis=0, yaxis=1)
    bev_image.render_point_cloud(point_cloud=pts_sensor_coord)
    if len(all_bboxes3d_sensor_coord):
        bev_image.render_bounding_box_3d(bboxes3d=all_bboxes3d_sensor_coord, texts=all_classes if id_to_name else None)

    return pts_world_coord, bev_image


def render_pointcloud_and_box_onto_rgb(camera_datums, pts_world_coord=None, id_to_name=None):
    """Project LiDAR point cloud to 2D rgb and render projected points overlapping on top of rgb.

    Parameters
    ----------
    camera_datums: list of OrderedDict

        "datum_name": str
            Sensor name from which the data was collected.

        "rgb": list of PIL.Image (mode=RGB)
            List of image in RGB format. List is in order of `selected_datums` that
            are images.

        "intrinsics": np.ndarray
            Camera intrinsics.

        "pose": Pose
            Ego pose of datum.

        "bounding_box_2d": np.ndarray dtype=np.float32
            Tensor containing bounding boxes for this sample
            (x, y, w, h) in absolute pixel coordinates.

        "bounding_box_3d": list of BoundingBox3D
            3D Bounding boxes for this sample specified in this point cloud
            sensor's reference frame. (i.e. this provides the bounding box
            (B) in the sensor's (S) reference frame `box_SB`).

    pts_world_coord: numpy.ndarray, default: None
        Accumulated point clouds from all lidar sensors in a same single coordinate frame.

    id_to_name: OrderedDict, default: None
        Object class id (int) to name (str).

    Returns
    -------
    images_2d: dict
        Image Datum name to RGB image with bounding boxes.

    images_3d: dict
        Image Datum name to point clouds and bounding boxes overlapping on top of the RGB image.

    """
    images_2d, images_3d = {}, {}
    for d_cam in camera_datums:
        # Ensure the datum names are consistent with the query
        # Render the image
        img = np.array(d_cam['rgb']).copy()
        vis_2d = print_status(img, d_cam['datum_name'])
        vis_3d = vis_2d.copy()

        # Render 3d annotations in the camera
        cam_identity = Camera(K=d_cam['intrinsics'])

        # Only render bounding boxes if available
        if 'bounding_box_3d' in d_cam:
            for bbox3d in d_cam['bounding_box_3d']:
                vis_2d = bbox3d.render_on_image(cam_identity, vis_2d)

        # Render 2d annotations
        if 'bounding_box_2d' in d_cam:
            classes = [id_to_name[cid] for cid in d_cam['class_ids']] if id_to_name else None
            vis_3d = render_bbox2d_on_image(vis_3d, d_cam['bounding_box_2d'], texts=classes)

        # Render 3d points into image
        if pts_world_coord is not None:
            # Move points in sensor (S) frame to the camera (C) frame.
            p_WC = d_cam['pose']
            pts_camera_coord = p_WC.inverse() * pts_world_coord
            vis_3d = render_pointcloud_on_image(vis_3d, cam_identity, pts_camera_coord)
        images_2d[d_cam['datum_name']] = vis_2d
        images_3d[d_cam['datum_name']] = vis_3d

    return images_2d, images_3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=True
    )

    # Dataset JSON parsing OR Scene dataset directory parsing
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset-json', help='Dataset JSON file')
    group.add_argument('--scene-dataset-json', help='Scene Dataset JSON file')
    group.add_argument('--scene-json', help='Scene JSON file')
    parser.add_argument(
        '--split', help='Dataset split', required=False, choices=['train', 'val', 'test'], default='train'
    )
    parser.add_argument('--camera-prefix', help='Camera prefix for e.g. CAM', required=False, default='CAM')
    parser.add_argument('--lidar-prefix', help='Lidar prefix for e.g. LIDAR', required=False, default='LIDAR')
    parser.add_argument('--point-cloud', help='Render point clouds onto each image', action='store_true')
    parser.add_argument('--point-cloud-only', help='Render only point clouds', action='store_true')
    parser.add_argument('--image-scale', help='Camera image scale', required=False, type=float, default=0.5)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args, other_args = parser.parse_known_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Synchronized dataset with all available datums within a sample
    dataset_args = dict(
        backward_context=0, forward_context=0, requested_annotations=("bounding_box_3d", "bounding_box_2d")
    )
    if args.dataset_json:
        logging.info('dataset-json mode: Using split {}'.format(args.split))
        dataset = SynchronizedDataset(args.dataset_json, split=args.split, **dataset_args)
    elif args.scene_dataset_json:
        logging.info('scene-dataset-json mode: Using split {}'.format(args.split))
        dataset = SynchronizedSceneDataset(args.scene_dataset_json, split=args.split, **dataset_args)
    elif args.scene_json:
        logging.info('scene-json mode: Split value ignored')
        # Fetch scene from S3 to cache if remote scene JSON provided
        if args.scene_json.startswith('s3://'):
            args.scene_json = fetch_remote_scene(args.scene_json)
        dataset = SynchronizedScene(args.scene_json, **dataset_args)
    else:
        raise ValueError('Provide either --dataset-json or --scene-json')

    if args.point_cloud_only:
        dataset.select_datums(datum_names=['LIDAR'])
    logging.info('Dataset: {}'.format(len(dataset)))

    # 2D visualization
    for _, data in enumerate(tqdm(dataset)):
        d_cams = [d for d in data if d['datum_name'].startswith(args.camera_prefix)]

        X_W = _bev_image = None
        if args.point_cloud:
            d_lidar = [d for d in data if d['datum_name'].startswith(args.lidar_prefix)]
            X_W, _bev_image = render_bev(d_lidar, getattr(dataset, 'id_to_name', None))
        if not args.point_cloud_only:
            assert len(d_cams) > 0, 'No cameras found with prefix {}'.format(args.camera_prefix)

        _images_2d, _images_3d = render_pointcloud_and_box_onto_rgb(d_cams, X_W, getattr(dataset, 'id_to_name', None))

        # Visualize 2D and 3D bounding boxes on two-sets of mosaics
        if len(_images_2d):
            mosaic_im = \
            mosaic([
                mosaic(list(_images_2d.values()), scale=
                       args.image_scale, grid_width=3),
                mosaic(list(_images_3d.values()),
                       scale=args.image_scale, grid_width=3)
                ],
                   grid_width=1)[..., ::-1]
            cv2.imshow('images', mosaic_im)

        # Visualize point cloud in BeV
        if args.point_cloud:
            cv2.imshow('bev', _bev_image.data[..., ::-1])
        if cv2.waitKey(0 if not args.video else 10) == ord('q'):
            exit()
