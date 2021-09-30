# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
import os

import cv2
import numpy as np
from matplotlib.cm import get_cmap

from dgp.utils.colors import (WHITE, adjust_lightness, color_borders, get_unique_colors)
from dgp.utils.visualization_utils import (
    mosaic, render_bbox2d_on_image, visualize_bev, visualize_cameras, visualize_semantic_segmentation_2d
)

MPL_JET_CMAP = get_cmap('jet')

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def visualize_dataset_3d(
    dataset,
    lidar_datum_names=None,
    camera_datum_names=None,
    render_pointcloud_on_images=True,
    show_instance_id_on_bev=True,
    max_num_items=None,
    caption_fn=None,
    # output video configs.
    output_video_file=None,
    output_video_fps=30,
    # global rendering-related configs
    class_colormap=None,
    adjust_lightness_factor=1.0,
    rgb_resize_factor=0.5,
    rgb_border_thickness=4,
    # `BEVImage` kwargs
    bev_metric_width=100.0,
    bev_metric_height=100.0,
    bev_pixels_per_meter=10.0,
    bev_polar_step_size_meters=10,
    bev_forward=(1, 0, 0),
    bev_left=(0, 1, 0),
    bev_background_clr=(0, 0, 0),
    bev_font_scale=0.5,
    bev_line_thickness=4,
    # `BoundingBox3D` kwargs
    bbox3d_font_scale=1.0,
    bbox3d_line_thickness=4,
    # `render_pointcloud_on_image` kwargs
    pc_rgb_cmap=MPL_JET_CMAP,
    pc_rgb_norm_depth=10,
    pc_rgb_dilation=3,
    # Radar related
    radar_datum_names=None,
    render_radar_pointcloud_on_images=True,
):
    """Visualize 3D annotations and pointclouds of a synchronized DGP dataset (e.g. SynchronizedSceneDataset,
    ParallelDomainScene). The output is a video, either rendered on X window or a video file, that visualize 3D bounding
    boxes and pointcloud 1) from a BEV view and 2) projected on cameras.
    Parameters
    ----------
    dataset: _SynchronizedDataset
        A multimodel dataset of which `__getitem__` returns a list of `OrderedDict`, one item for each datum.
    lidar_datum_names: None or List[str], default: None
        Names of lidar datums. If None, then use all datums whose type is `point_cloud` and are available in all scenes.
    camera_datum_names: None or List[str], default: None
        Names of camera_datums. If None, then use all datums whose type is `image` and are available in all scenes.
        In the output video, the image visualizations are tiled in row-major order according to the order of this list.
    render_pointcloud_on_images: bool, default: True
        Whether or not to render projected pointclouds on images.
    show_instance_id_on_bev: bool, default: True
        If True, then show `instance_id` on a corner of 3D bounding boxes in BEV view.
        If False, then show `class_name` instead.
    max_num_items: None or int, default: None
        If not None, then show only up to this number of items. This is useful for debugging a large dataset.
    caption_fn: Callable or None
        A function that take as input a `_SynchronizedDataset` and index, and return a text (str).
        The text is put as caption at the top left corner of video.
    output_video_file: None or str, default: None
        If not None, then write the visualization on a video of this name. It must ends with `.avi`.
    output_video_fps: int, default: 30
        Frame rate of the video.
    overwrite_output_file: bool, default: False
        If True, then FFMPEG overwrites existing output file without prompting confirmation.
        This is useful when the `output_video_file` is created by `tempfile` with context manager.
    class_colormap: dict or None, default: None
        Dict of class name to RGB color tuple. If None, then use class color defined in `dataset`.
    adjust_lightness_factor: float, default: 1.0
        Enhance the brightness of colormap by this factor.
    rgb_resize_factor: float, default: 0.5
        Resize images by this factor before tiling them into a single panel.
    rgb_border_thickness: int, default: 10
        Put a colored boundary on camera visualization of this thickness before tiling them into single panel.
    bev-*:
        See `BEVImage` for these keyword arguments.
    bbox3d-*:
        See `geometry.BoundingBox3D` for these keyword arguments.
    pc_rgb-*:
        See `render_pointcloud_on_image()` for these keyword arguments.
    radar_datum_names: None or List[str], default: None
        Names of the radar datums
    render_radar_pointcloud_on_images: bool, default: True
        Whether or not to render projected radar pointclouds on images.
    """
    if output_video_file:
        assert output_video_file.endswith('.avi'), "'output_video' must ends with `.avi`."
        assert output_video_fps > 0
        os.makedirs(os.path.dirname(output_video_file), exist_ok=True)

    if max_num_items is not None:
        if max_num_items > len(dataset):
            LOG.info(
                "`max_num_items` is reduced to the dataset size, from {:d} to {:d}".format(max_num_items, len(dataset))
            )
            max_num_items = len(dataset)

    ontology = dataset.dataset_metadata.ontology_table.get('bounding_box_3d', None)
    if ontology is None:
        class_colormap = dict()
        id_to_name = dict()
    elif ontology is not None and class_colormap is None:
        class_colormap = ontology._contiguous_id_colormap
        id_to_name = ontology.contiguous_id_to_name
    else:
        class_colormap = {ontology.name_to_contiguous_id[class_name]: clr for class_name, clr in class_colormap.items()}
        id_to_name = ontology.contiguous_id_to_name

    if adjust_lightness_factor != 1.0:
        class_colormap = {
            class_id: adjust_lightness(clr, factor=adjust_lightness_factor)
            for class_id, clr in class_colormap.items()
        }

    if camera_datum_names is None:
        camera_datum_names = sorted(dataset.list_datum_names_available_in_all_scenes(datum_type='image'))
    if lidar_datum_names is None:
        lidar_datum_names = sorted(dataset.list_datum_names_available_in_all_scenes(datum_type='point_cloud'))
    if radar_datum_names is None:
        radar_datum_names = sorted(dataset.list_datum_names_available_in_all_scenes(datum_type='radar_point_cloud'))

    # Unique colors for cameras used in frustrum viz in BEV and image borders.
    camera_colors = get_unique_colors(len(camera_datum_names))
    for idx, datums in enumerate(dataset):
        # no temporal context
        datums = datums[0]
        if idx == max_num_items:
            break
        LOG.info("frame #%d of %d" % (idx + 1, len(dataset)))

        datums = {datum['datum_name']: datum for datum in datums}

        lidar_datums = [datums[lidar_datum_name] for lidar_datum_name in lidar_datum_names]
        camera_datums = [datums[cam_datum_name] for cam_datum_name in camera_datum_names]
        radar_datums = [datums[radar_datum_name] for radar_datum_name in radar_datum_names]

        # BEV visualization
        bev_viz = visualize_bev(
            lidar_datums,
            class_colormap,
            show_instance_id_on_bev,
            id_to_name,
            camera_datums,
            camera_colors,
            bev_metric_width,
            bev_metric_height,
            bev_pixels_per_meter,
            bev_polar_step_size_meters,
            bev_forward,
            bev_left,
            bev_background_clr,
            bev_line_thickness,
            bev_font_scale,
            radar_datums=radar_datums
        )

        # Camera visualization
        rgb_viz = visualize_cameras(
            camera_datums,
            id_to_name,
            lidar_datums if render_pointcloud_on_images else None,
            rgb_resize_factor,
            bbox3d_font_scale,
            bbox3d_line_thickness,
            pc_rgb_cmap,
            pc_rgb_norm_depth,
            pc_rgb_dilation,
            radar_datums=radar_datums if render_radar_pointcloud_on_images else None
        )
        rgb_viz = [color_borders(viz, clr, rgb_border_thickness) for viz, clr in zip(rgb_viz, camera_colors)]
        rgb_mosaic = mosaic(rgb_viz)

        vis_height = rgb_mosaic.shape[0]
        bev_viz = cv2.resize(bev_viz, (vis_height, vis_height))
        viz_frame = np.hstack([bev_viz, rgb_mosaic])
        if caption_fn is not None:
            caption = caption_fn(dataset, idx)
            cv2.putText(viz_frame, caption, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2, cv2.LINE_AA)

        if output_video_file is not None:
            if idx == 0:
                # CAVEAT: The videos are MJPG-encoded AVI files.
                # It is recommended to convert them using more standard video codec.
                # E.g. "ffmpeg -y -i <AVI_VIDEO> -c:v libx264 -crf 25 -pix_fmt yuv420p <MP4_VIDEO>"
                video_writer = cv2.VideoWriter(
                    output_video_file, cv2.VideoWriter_fourcc(*"MJPG"), output_video_fps,
                    (viz_frame.shape[1], viz_frame.shape[0])
                )
            video_writer.write(viz_frame[:, :, ::-1])
        else:
            cv2.imshow('visualize_3d', viz_frame[:, :, ::-1])
            cv2.waitKey(30)


def visualize_dataset_2d(
    dataset,
    camera_datum_names=None,
    max_num_items=None,
    caption_fn=None,
    show_instance_id=False,
    # output video configs.
    output_video_file=None,
    output_video_fps=30,
    # global rendering-related configs
    class_colormap=None,
    adjust_lightness_factor=1.0,
    font_scale=1,
    rgb_resize_factor=0.5,
):
    """Visualize 2D annotations of a synchronized DGP dataset (e.g. SynchronizedSceneDataset,
    ParallelDomainScene). The output is a video, either rendered on X window or a video file, that visualize
    2D bounding boxes and optionally instance segmentation masks.

    Parameters
    ----------
    dataset: _SynchronizedDataset
        A multimodel dataset of which `__getitem__` returns a list of `OrderedDict`, one item for each datum.

    camera_datum_names: None or List[str], default: None
        Names of camera_datums. If None, then use all datums whose type is `image` and are available in all scenes.
        In the output video, the image visualizations are tiled in row-major order according to the order of this list.

    max_num_items: None or int, default: None
        If not None, then show only up to this number of items. This is useful for debugging a large dataset.

    caption_fn: Callable or None
        A function that take as input a `_SynchronizedDataset` and index, and return a text (str).
        The text is put as caption at the top left corner of video.

    show_instance_id: bool, default: False
        Option to show instance id instead of instance class name on annotated images.

    output_video_file: None or str, default: None
        If not None, then write the visualization on a video of this name. It must ends with `.avi`.

    output_video_fps: int, default: 30
        Frame rate of the video.

    class_colormap: dict or None, default: None
        Dict of class name to RGB color tuple. If None, then use class color defined in `dataset`.

    adjust_lightness_factor: float, default: 1.0
        Enhance the brightness of colormap by this factor.

    font_scale: float, default: 1
        Font scale used for all text.

    rgb_resize_factor: float, default: 0.5
        Resize images by this factor before tiling them into a single panel.
    """
    if output_video_file:
        assert output_video_file.endswith('.avi'), "'output_video' must ends with `.avi`."
        assert output_video_fps > 0
        os.makedirs(os.path.dirname(output_video_file), exist_ok=True)

    if max_num_items is not None:
        if max_num_items > len(dataset):
            LOG.info(
                "`max_num_items` is reduced to the dataset size, from {:d} to {:d}".format(max_num_items, len(dataset))
            )
            max_num_items = len(dataset)

    ontology_table = dataset.dataset_metadata.ontology_table
    if 'bounding_box_2d' in ontology_table:
        ontology = ontology_table['bounding_box_2d']
        if class_colormap is None:
            class_colormap = ontology.colormap
        else:
            class_colormap = {ontology.name_to_contiguous_id[class_name]: \
                clr for class_name, clr in class_colormap.items()}

        if adjust_lightness_factor != 1.0:
            class_colormap = {
                class_id: adjust_lightness(clr, factor=adjust_lightness_factor)
                for class_id, clr in class_colormap.items()
            }

    if camera_datum_names is None:
        camera_datum_names = sorted(dataset.list_datum_names_available_in_all_scenes(datum_type='image'))

    for idx, datums in enumerate(dataset):
        # no temporal context
        datums = datums[0]
        if idx == max_num_items:
            break
        LOG.info("Frame #%d of %d" % (idx + 1, len(dataset)))

        datums = {datum['datum_name']: datum for datum in datums}

        camera_datums = [datums[cam_datum_name] for cam_datum_name in camera_datum_names]

        ##########
        # Cam viz.
        ##########
        rgb_mosaic = []

        # Visualize bounding box 2d
        if 'bounding_box_2d' in dataset.requested_annotations:
            for cam_datum in camera_datums:
                rgb = np.array(cam_datum['rgb']).copy()
                class_ids = [bbox2d.class_id for bbox2d in cam_datum['bounding_box_2d']]
                class_colors = [class_colormap[class_id] for class_id in class_ids]
                instance_masks = None
                if 'instance_segmentation_2d' in cam_datum.keys():
                    instance_masks = cam_datum['instance_segmentation_2d'].instances
                if show_instance_id:
                    class_names = [str(instance_id) for instance_id in cam_datum['instance_ids']]
                else:
                    class_names = [ontology.contiguous_id_to_name[class_id] for class_id in class_ids]

                bboxes = np.vstack([bbox.ltwh for bbox in cam_datum['bounding_box_2d']])

                instance_mask_patches = []

                for i, bbox in enumerate(bboxes):
                    x1, y1, w, h = bbox.astype(int)
                    x2 = x1 + w
                    y2 = y1 + h
                    if instance_masks is not None:
                        mask = instance_masks[i]
                        instance_mask_patches.append(mask[y1:y2, x1:x2])

                rgb = render_bbox2d_on_image(
                    rgb,
                    bboxes,
                    colors=class_colors,
                    texts=class_names,
                    instance_masks=instance_mask_patches if instance_masks is not None else None
                )
                rgb_mosaic.append(cv2.resize(rgb, None, fx=rgb_resize_factor, fy=rgb_resize_factor))

        # Visualize semantic segmentation 2d
        if 'semantic_segmentation_2d' in dataset.requested_annotations:
            for cam_datum in camera_datums:
                rgb = np.array(cam_datum['rgb']).copy()
                rgb = visualize_semantic_segmentation_2d(
                    cam_datum['semantic_segmentation_2d'].label, ontology_table['semantic_segmentation_2d'], image=rgb
                )
                rgb_mosaic.append(cv2.resize(rgb, None, fx=rgb_resize_factor, fy=rgb_resize_factor))
        rgb_mosaic = mosaic(rgb_mosaic)

        if caption_fn is not None:
            caption = caption_fn(dataset, idx)
            vis_width = rgb_mosaic.shape[1]
            caption_patch = np.zeros((50, vis_width, 3), np.uint8)
            cv2.putText(caption_patch, caption, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE, 2, cv2.LINE_AA)
            viz_frame = np.vstack([caption_patch, rgb_mosaic])

        if output_video_file is not None:
            if idx == 0:
                # CAVEAT: The videos are MJPG-encoded AVI files.
                # It is recommended to convert them using more standard video codec.
                # E.g. "ffmpeg -y -i <AVI_VIDEO> -c:v libx264 -crf 25 -pix_fmt yuv420p <MP4_VIDEO>"
                video_writer = cv2.VideoWriter(
                    output_video_file, cv2.VideoWriter_fourcc(*"MJPG"), output_video_fps,
                    (viz_frame.shape[1], viz_frame.shape[0])
                )
            video_writer.write(viz_frame[:, :, ::-1])
        else:
            cv2.imshow('visualize_2d', viz_frame[:, :, ::-1])
            cv2.waitKey(30)


def visualize_dataset_sample_3d(
    dataset,
    scene_idx,
    sample_idx,
    lidar_datum_names=None,
    camera_datum_names=None,
    render_pointcloud_on_images=True,
    show_instance_id_on_bev=True,
    # global rendering-related configs
    class_colormap=None,
    adjust_lightness_factor=1.0,
    rgb_resize_factor=0.5,
    rgb_border_thickness=4,
    # `BEVImage` kwargs
    bev_metric_width=100.0,
    bev_metric_height=100.0,
    bev_pixels_per_meter=10.0,
    bev_polar_step_size_meters=10,
    bev_forward=(1, 0, 0),
    bev_left=(0, 1, 0),
    bev_background_clr=(0, 0, 0),
    bev_font_scale=0.5,
    bev_line_thickness=4,
    # `BoundingBox3D` kwargs
    bbox3d_font_scale=1.0,
    bbox3d_line_thickness=4,
    # `render_pointcloud_on_image` kwargs
    pc_rgb_cmap=MPL_JET_CMAP,
    pc_rgb_norm_depth=10,
    pc_rgb_dilation=3,
    # Radar related
    radar_datum_names=None,
):
    """Visualize 3D annotations and pointclouds of a single sample of a DGP dataset (e.g. SynchronizedSceneDataset,
    ParallelDomainScene). The output is a dictionary of images keyed by camera datum name and 'bev' for the BEV image,

    Parameters
    ----------
    dataset: _SynchronizedDataset
        A multimodel dataset of which `__getitem__` returns a list of `OrderedDict`, one item for each datum.
    scene_idx: int
        scene index into dataset
    sample_idx: int
        index of sample in scene
    lidar_datum_names: None or List[str], default: None
        Names of lidar datum
    camera_datum_names: None or List[str], default: None
        Names of camera_datums
    render_pointcloud_on_images: bool, default: True
        Whether or not to render projected pointclouds on images.
    show_instance_id_on_bev: bool, default: True
        If True, then show `instance_id` on a corner of 3D bounding boxes in BEV view.
        If False, then show `class_name` instead.
    class_colormap: dict or None, default: None
        Dict of class name to RGB color tuple. If None, then use class color defined in `dataset`.
    adjust_lightness_factor: float, default: 1.0
        Enhance the brightness of colormap by this factor.
    rgb_resize_factor: float, default: 0.5
        Resize images by this factor before tiling them into a single panel.
    rgb_border_thickness: int, default: 10
        Put a colored boundary on camera visualization of this thickness before tiling them into single panel.
    bev-*:
        See `BEVImage` for these keyword arguments.
    bbox3d-*:
        See `geometry.BoundingBox3D` for these keyword arguments.
    pc_rgb-*:
        See `render_pointcloud_on_image()` for these keyword arguments.
    radar_datum_names: None or List[str], default: None
        Names of the radar datums

    Returns
    -------
    results : dict of str : np.array
        Dictionary of images keyed by their camera datum name or 'bev' for the bird's eye view.
    """

    ontology = dataset.dataset_metadata.ontology_table.get('bounding_box_3d', None)
    if ontology is None:
        class_colormap = dict()
        id_to_name = dict()
    elif ontology is not None and class_colormap is None:
        class_colormap = ontology._contiguous_id_colormap
        id_to_name = ontology.contiguous_id_to_name
    else:
        class_colormap = {ontology.name_to_contiguous_id[class_name]: clr for class_name, clr in class_colormap.items()}
        id_to_name = ontology.contiguous_id_to_name

    if adjust_lightness_factor != 1.0:
        class_colormap = {
            class_id: adjust_lightness(clr, factor=adjust_lightness_factor)
            for class_id, clr in class_colormap.items()
        }

    results = {}

    # Unique colors for cameras used in frustrum viz in BEV and image borders.
    camera_datums = []
    camera_colors = []
    if camera_datum_names:
        camera_colors = get_unique_colors(len(camera_datum_names))
        camera_datums = [
            dataset.get_datum_data(scene_idx, sample_idx, cam_datum_name) for cam_datum_name in camera_datum_names
        ]

    lidar_datums = []
    if lidar_datum_names:
        lidar_datums = [
            dataset.get_datum_data(scene_idx, sample_idx, lidar_datum_name) for lidar_datum_name in lidar_datum_names
        ]

    radar_datums = []
    if radar_datum_names:
        radar_datums = [
            dataset.get_datum_data(scene_idx, sample_idx, radar_datum_name) for radar_datum_name in radar_datum_names
        ]

    # BEV visualization
    if lidar_datums:
        bev_viz = visualize_bev(
            lidar_datums,
            class_colormap,
            show_instance_id_on_bev,
            id_to_name,
            camera_datums,
            camera_colors,
            bev_metric_width,
            bev_metric_height,
            bev_pixels_per_meter,
            bev_polar_step_size_meters,
            bev_forward,
            bev_left,
            bev_background_clr,
            bev_line_thickness,
            bev_font_scale,
            radar_datums=radar_datums
        )
        results['bev'] = bev_viz

    # Camera visualization
    rgb_viz = visualize_cameras(
        camera_datums,
        id_to_name,
        lidar_datums if render_pointcloud_on_images else None,
        rgb_resize_factor,
        bbox3d_font_scale,
        bbox3d_line_thickness,
        pc_rgb_cmap,
        pc_rgb_norm_depth,
        pc_rgb_dilation,
        radar_datums=radar_datums
    )

    if rgb_border_thickness > 0:
        rgb_viz = [color_borders(viz, clr, rgb_border_thickness) for viz, clr in zip(rgb_viz, camera_colors)]

    if camera_datum_names:
        for k, cam_name in enumerate(camera_datum_names):
            results[cam_name] = rgb_viz[k]
    return results


def visualize_dataset_sample_2d(
    dataset,
    scene_idx,
    sample_idx,
    camera_datum_names=None,
    show_instance_id=False,
    # global rendering-related configs
    class_colormap=None,
    adjust_lightness_factor=1.0,
    rgb_resize_factor=0.5,
):
    """Visualize 2D annotations of a single sample from a synchronized DGP dataset (e.g. SynchronizedSceneDataset,
    ParallelDomainScene). The output is a tuple dictionaries of images keyed by camera datum name (bounding box images, semantic segmentation images).

    Parameters
    ----------
    dataset: _SynchronizedDataset
        A multimodel dataset of which `__getitem__` returns a list of `OrderedDict`, one item for each datum.

    camera_datum_names: None or List[str], default: None
        Names of camera_datums. If None, then use all datums whose type is `image` and are available in all scenes.
        In the output video, the image visualizations are tiled in row-major order according to the order of this list.

    max_num_items: None or int, default: None
        If not None, then show only up to this number of items. This is useful for debugging a large dataset.

    show_instance_id: bool, default: False
        Option to show instance id instead of instance class name on annotated images.

    class_colormap: dict or None, default: None
        Dict of class name to RGB color tuple. If None, then use class color defined in `dataset`.

    adjust_lightness_factor: float, default: 1.0
        Enhance the brightness of colormap by this factor.

    rgb_resize_factor: float, default: 0.5
        Resize images by this factor before tiling them into a single panel.

    Returns
    -------
    bbox_rgb : dict of str : np.array
        Dictionary of bounding box annotated images keyed by the camera datum name.

    semantic_rgb : dict of str : np.array
        Dictionary of semantic segmentation images keyed by camera datum name.
    """
    ontology_table = dataset.dataset_metadata.ontology_table
    if 'bounding_box_2d' in ontology_table:
        ontology = ontology_table['bounding_box_2d']
        if class_colormap is None:
            class_colormap = ontology.colormap
        else:
            class_colormap = {ontology.name_to_contiguous_id[class_name]: \
                clr for class_name, clr in class_colormap.items()}

        if adjust_lightness_factor != 1.0:
            class_colormap = {
                class_id: adjust_lightness(clr, factor=adjust_lightness_factor)
                for class_id, clr in class_colormap.items()
            }

    if camera_datum_names is None:
        camera_datum_names = sorted(dataset.list_datum_names_available_in_all_scenes(datum_type='image'))

    camera_datums = [
        dataset.get_datum_data(scene_idx, sample_idx, cam_datum_name) for cam_datum_name in camera_datum_names
    ]

    bbox_rgb = {}
    # Visualize bounding box 2d
    if 'bounding_box_2d' in dataset.requested_annotations:
        for cam_datum, cam_datum_name in zip(camera_datums, camera_datum_names):
            rgb = np.array(cam_datum['rgb']).copy()
            class_ids = [bbox2d.class_id for bbox2d in cam_datum['bounding_box_2d']]
            class_colors = [class_colormap[class_id] for class_id in class_ids]
            instance_masks = None
            if 'instance_segmentation_2d' in cam_datum.keys():
                instance_masks = cam_datum['instance_segmentation_2d'].instances
            if show_instance_id:
                class_names = [str(instance_id) for instance_id in cam_datum['instance_ids']]
            else:
                class_names = [ontology.contiguous_id_to_name[class_id] for class_id in class_ids]

            bboxes = np.vstack([bbox.ltwh for bbox in cam_datum['bounding_box_2d']])

            instance_mask_patches = []

            for i, bbox in enumerate(bboxes):
                x1, y1, w, h = bbox.astype(int)
                x2 = x1 + w
                y2 = y1 + h
                if instance_masks is not None:
                    mask = instance_masks[i]
                    instance_mask_patches.append(mask[y1:y2, x1:x2])

            rgb = render_bbox2d_on_image(
                rgb,
                bboxes,
                colors=class_colors,
                texts=class_names,
                instance_masks=instance_mask_patches if instance_masks is not None else None
            )
            bbox_rgb[cam_datum_name] = cv2.resize(rgb, None, fx=rgb_resize_factor, fy=rgb_resize_factor)

    semantic_rgb = {}
    # Visualize semantic segmentation 2d
    if 'semantic_segmentation_2d' in dataset.requested_annotations:
        for cam_datum, cam_datum_name in zip(camera_datums, camera_datum_names):
            rgb = np.array(cam_datum['rgb']).copy()
            rgb = visualize_semantic_segmentation_2d(
                cam_datum['semantic_segmentation_2d'].label, ontology_table['semantic_segmentation_2d'], image=rgb
            )
            semantic_rgb[cam_datum_name] = cv2.resize(rgb, None, fx=rgb_resize_factor, fy=rgb_resize_factor)
    return bbox_rgb, semantic_rgb
