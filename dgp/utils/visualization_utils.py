# Copyright 2021 Toyota Research Institute.  All rights reserved.
"""Visualization tools for a variety of tasks"""
import logging

import cv2
import numpy as np
from matplotlib.cm import get_cmap

from dgp.utils.camera import Camera
from dgp.utils.colors import (DARKGRAY, GRAY, GREEN, RED, WHITE, YELLOW, get_unique_colors)
from dgp.utils.pose import Pose

# Time to wait before key press in debug visualizations
DEBUG_WAIT_TIME = 10000
MPL_JET_CMAP = get_cmap('jet')

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def make_caption(dataset, idx, prefix=''):
    """Make caption that tells scene directory and sample index.

    Paramters
    ---------
    dataset: BaseDataset
        BaseDataset. e.g. ParallelDomainScene(Dataset), SynchronizedScene(Dataset)
    idx: int
        Image index
    prefix: str
        Caption prefix

    Returns
    -------
    caption: str
        Caption of the image
    """
    scene_idx, frame_idx, _ = dataset.dataset_item_index[idx]
    scene_name = dataset.scenes[scene_idx].scene.name
    return "{:s}{:s} #{:d}".format(prefix, scene_name, frame_idx)


def print_status(image, text):
    """Adds a status bar at the bottom of image, with provided text.

    Parameters
    ----------
    image: np.array of shape (H, W, 3)
        Image to print status on.

    text: str
        Text to be printed.

    Returns
    -------
    image: np.array of shape (H, W, 3)
        Image with status printed
    """
    H, W = image.shape[:2]
    status_xmax = int(W)
    status_ymin = H - 40
    text_offset = int(5 * 1)
    cv2.rectangle(image, (0, status_ymin), (status_xmax, H), DARKGRAY, thickness=-1)
    cv2.putText(image, '%s' % text, (text_offset, H - text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, thickness=1)
    return image


def mosaic(items, scale=1.0, pad=3, grid_width=None):
    """Creates a mosaic from list of images.

    Parameters
    ----------
    items: list of np.ndarray
        List of images to mosaic.

    scale: float, default=1.0
        Scale factor applied to images. scale > 1.0 enlarges images.

    pad: int, default=3
        Padding size of the images before mosaic

    grid_width: int, default=None
        Mosaic width or grid width of the mosaic

    Returns
    -------
    image: np.array of shape (H, W, 3)
        Image mosaic
    """
    # Determine tile width and height
    N = len(items)
    assert N > 0, 'No items to mosaic!'
    grid_width = grid_width if grid_width else np.ceil(np.sqrt(N)).astype(int)
    grid_height = np.ceil(N * 1. / grid_width).astype(np.int)
    input_size = items[0].shape[:2]
    target_shape = (int(input_size[1] * scale), int(input_size[0] * scale))
    mosaic_items = []
    for j in range(grid_width * grid_height):
        if j < N:
            # Only the first image is scaled, the rest are re-shaped
            # to the same size as the previous image in the mosaic
            im = cv2.resize(items[j], dsize=target_shape)
            mosaic_items.append(im)
        else:
            mosaic_items.append(np.zeros_like(mosaic_items[-1]))

    # Stack W tiles horizontally first, then vertically
    im_pad = lambda im: cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    mosaic_items = [im_pad(im) for im in mosaic_items]
    hstack = [np.hstack(mosaic_items[j:j + grid_width]) for j in range(0, len(mosaic_items), grid_width)]
    mosaic_viz = np.vstack(hstack) if len(hstack) > 1 \
        else hstack[0]
    return mosaic_viz


def render_bbox2d_on_image(img, bboxes2d, instance_masks=None, colors=None, texts=None, line_thickness=4):
    """Render list of bounding box2d on image.

    Parameters
    ----------
    img: np.ndarray
        Image to render bounding boxes onto.

    bboxes2d: np.ndarray (N x 4)
        Array of 2d bounding box (x, y, w, h).

    instance_masks: list
        List of binary instance masks cropped to (w,h).

    colors: list
        List of color tuples.

    texts: list, default: None
        List of str classes.

    line_thickness: int
        Line thickness value for bounding box edges.

    Returns
    -------
    img: np.array
        Image with rendered bounding boxes.
    """
    boxes = [
        np.int32([[bbox2d[0], bbox2d[1]], [bbox2d[0] + bbox2d[2], bbox2d[1]],
                  [bbox2d[0] + bbox2d[2], bbox2d[1] + bbox2d[3]], [bbox2d[0], bbox2d[1] + bbox2d[3]]])
        for bbox2d in bboxes2d
    ]
    if colors is None:
        cv2.polylines(img, boxes, True, RED, thickness=line_thickness)
    else:
        assert len(boxes) == len(colors), 'len(boxes) != len(colors)'
        for idx, box in enumerate(boxes):
            cv2.polylines(img, [box], True, colors[idx], thickness=line_thickness)

    # Add Mask
    if instance_masks is not None:
        assert len(instance_masks) == len(bboxes2d)
        for instance_mask, bboxe2d, color in zip(instance_masks, bboxes2d, colors):
            x1, y1, w, h = bboxe2d
            x2, y2 = int(x1 + w), int(y1 + h)
            x1, y1 = int(x1), int(y1)
            img_color_patch = img[y1:y2, x1:x2, :]
            color_np = np.expand_dims(np.array(color), axis=0)
            mask_color_patch = np.matmul(np.expand_dims(instance_mask, axis=2), color_np)
            img[y1:y2, x1:x2, :] = np.clip(mask_color_patch + img_color_patch, 0, 255)

    # Add texts
    if texts:
        assert len(boxes) == len(texts), 'len(boxes) != len(texts)'
        for idx, box in enumerate(boxes):
            cv2.putText(img, texts[idx], tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2, cv2.LINE_AA)
    return img


def visualize_bounding_box_2d(image, bounding_box_2d, ontology, debug=False):
    """DGP-friendly bounding_box_2d visualization util

    Parameters
    ----------
    image: np.uint8 array
        Image to visualize boxes on in BGR format
        shape: (H, W, 3)

    bounding_box_2d: dgp.proto.annotations_pb2.BoundingBox2DAnnotations
        Bounding box annotations

    ontology: dgp.proto.ontology_pb2.Ontology
        Ontology with which to visualize (for colors and class names)

    debug: bool, default: False
        If True render image until key-press

    Returns
    -------
    viz: np.uint8 array
        BGR visualization with bounding boxes
        shape: (H, W, 3)
    """
    bboxes2d = np.array([(annotation.box.x, annotation.box.y, annotation.box.w, annotation.box.h)
                         for annotation in bounding_box_2d.annotations])
    id_to_color = {item.id: (item.color.b, item.color.g, item.color.r) for item in ontology.items}
    id_to_name = {item.id: item.name for item in ontology.items}
    colors = [id_to_color[annotation.class_id] for annotation in bounding_box_2d.annotations]
    class_names = [id_to_name[annotation.class_id] for annotation in bounding_box_2d.annotations]
    viz = render_bbox2d_on_image(np.copy(image), bboxes2d, colors=colors, texts=class_names)
    if debug:
        cv2.imshow('image', viz)
        cv2.waitKey(DEBUG_WAIT_TIME)
    return viz


def render_pointcloud_on_image(img, camera, Xw, cmap=MPL_JET_CMAP, norm_depth=10, dilation=3):
    """Render pointcloud on image.

    Parameters
    ----------
    img: np.ndarray
        Image to render bounding boxes onto.

    camera: Camera
        Camera object with appropriately set extrinsics wrt world.

    Xw: np.ndarray (N x 3)
        3D point cloud (x, y, z) in the world coordinate.

    cmap: matplotlib.colors.Colormap
        Colormap used for visualizing the inverse depth.

    norm_depth: float, default: 10
        Depth value to normalize (inverse) pointcloud depths in color mapping.

    dilation: int, default: 3
        Dilation factor applied on each point.

    Returns
    -------
    img: np.array
        Image with rendered point cloud.
    """
    # Move point cloud to the camera's (C) reference frame from the world (W)
    Xc = camera.p_cw * Xw
    # Project the points as if they were in the camera's frame of reference
    uv = Camera(K=camera.K).project(Xc)
    # Colorize the point cloud based on depth
    z_c = Xc[:, 2]
    zinv_c = 1. / (z_c + 1e-6)
    zinv_c *= norm_depth
    colors = (cmap(np.clip(zinv_c, 0., 1.0))[:, :3] * 255).astype(np.uint8)

    # Create an empty image to overlay
    H, W, _ = img.shape
    vis = np.zeros_like(img)
    in_view = np.logical_and.reduce([(uv >= 0).all(axis=1), uv[:, 0] < W, uv[:, 1] < H, z_c > 0])
    uv, colors = uv[in_view].astype(int), colors[in_view]
    vis[uv[:, 1], uv[:, 0]] = colors  # pylint: disable=unsupported-assignment-operation

    # Dilate visualization so that they render clearly
    vis = cv2.dilate(vis, np.ones((dilation, dilation)))
    mask = (vis > 0).astype(np.uint8)
    return (1 - mask) * img + mask * vis


def render_radar_pointcloud_on_image(
    img, camera, point_cloud, cmap=MPL_JET_CMAP, norm_depth=10, velocity=None, velocity_scale=1, velocity_max_pix=.05
):
    """Render radar pointcloud on image.

    Parameters
    ----------
    img: np.ndarray
        Image to render bounding boxes onto.

    camera: Camera
        Camera object with appropriately set extrinsics wrt world.

    Xw: np.ndarray (N x 8)
        point cloud in spherical coordinates of radar sensor frame

    cmap: matplotlib.colors.Colormap
        Colormap used for visualizing the inverse depth.

    norm_depth: float, default: 10
        Depth value to normalize (inverse) pointcloud depths in color mapping.

    velocity: numpy array with shape (N,3), default None
        velocity vector of points

    velocity_scale: float
            factor to scale velocity vector by

    velocity_max_pix: float
        Maximum length of velocity vector rendering in percent of image width

    Returns
    -------
    img: np.array
        Image with rendered point cloud.
    """
    if len(point_cloud) == 0:
        return img

    Xc = camera.p_cw * point_cloud
    # Project the points as if they were in the camera's frame of reference
    uv = Camera(K=camera.K).project(Xc)
    # Colorize the point cloud based on depth
    z_c = Xc[:, 2]
    zinv_c = 1. / (z_c + 1e-6)
    zinv_c *= norm_depth
    colors = (cmap(np.clip(zinv_c, 0., 1.0))[:, :3] * 255)

    # Create an empty image to overlay
    H, W, _ = img.shape
    in_view = np.logical_and.reduce([(uv >= 0).all(axis=1), uv[:, 0] < W, uv[:, 1] < H, z_c > 0])

    uv, colors = uv[in_view].astype(int), colors[in_view]

    for row, color in zip(uv, colors):
        cx, cy = row
        cv2.circle(img, (cx, cy), 10, color, thickness=3)

    def clip_norm(v, x):
        M = np.linalg.norm(v)
        if M == 0:
            return v
        return np.clip(M, 0, x) * v / M

    if velocity is not None:
        tail = point_cloud + velocity_scale * velocity
        uv_tail = Camera(K=camera.K).project(camera.p_cw * tail)
        uv_tail = uv_tail[in_view].astype(int)
        for row, row_tail, color in zip(uv, uv_tail, colors):
            v_2d = row_tail - row
            v_2d = clip_norm(v_2d, velocity_max_pix * W)
            cx, cy = row
            cx2, cy2 = row + v_2d.astype(np.int)
            cx2 = np.clip(cx2, 0, W - 1)
            cy2 = np.clip(cy2, 0, H - 1)
            cv2.arrowedLine(img, (cx, cy), (cx2, cy2), color, thickness=2, line_type=cv2.LINE_AA)

    return img


class BEVImage:
    """A class for bird's eye view visualization, which generates a canvas of bird's eye view image,

    The class concerns two types of transformations:
        Extrinsics:
            A pose of sensor wrt the body frame. The inputs of rendering functions (`point_cloud` and `bboxes3d`)
            are in this sensor frame.
        BEV rotation:
            This defines an axis-aligned transformation from the body frame to BEV frame.
            For this, it uses conventional definition of orientations in the body frame:
                "forward" is a unit vector pointing to forward direction in the body frame.
                "left" is a unit vector pointing to left-hand-side in the body frame.
            In BEV frame,
                "forward" matches with right-hand-side of BEV image(x-axis)
                "left" matches with top of BEV image (negative y-axis)

    The rendering is done by chaining the extrinsics and BEV rotation to transform the inputs to
    the BEV camera, and then apply an orthographic transformation.

    Parameters
    ----------
    metric_width: float, default: 100.
        Metric extent of the view in width (X)

    metric_height: float, default: 100.
        Metric extent of the view in height (Y)

    pixels_per_meter: float, default: 10.
        Scale that expresses pixels per meter

    polar_step_size_meters: int, default: 10
        Metric steps at which to draw the polar grid

    extrinsics: Pose, default: Identity pose
        The pose of the sensor wrt the body frame (Sensor frame -> (Vehicle) Body frame).
        The input of rendering functions (i.e. `point_cloud`, `bboxes3d`) are assumed to be in the sensor frame.

    forward, left: tuple[int], defaults: (1., 0., 0), (0., 1., 0.)
        Length-3 orthonormal vectors that represents "forward" and "left" direction in the body frame.
        The default values assumes the most standard body frame; i.e., x: forward, y: left z: up.
        These are used to construct a rotation transformation from the body frame to the BEV frame.

    background_clr: tuple[int], defaults: (0, 0, 0)
        Background color in BGR order.

    center_offset_w: int, default: 0
        Offset in pixels to move ego center in BEV.

    center_offset_h: int, default: 0
        Offset in pixels to move ego center in BEV.
    """
    def __init__(
        self,
        metric_width=100.0,
        metric_height=100.0,
        pixels_per_meter=10.0,
        polar_step_size_meters=10,
        forward=(1, 0, 0),
        left=(0, 1, 0),
        background_clr=(0, 0, 0),
        center_offset_w=0,
        center_offset_h=0,
    ):
        forward, left = np.array(forward, np.float64), np.array(left, np.float64)
        assert np.dot(forward, left) == 0  # orthogonality check.

        self._metric_width = metric_width
        self._metric_height = metric_height
        self._pixels_per_meter = pixels_per_meter
        self._polar_step_size_meters = polar_step_size_meters
        self._forward = forward
        self._left = left
        self._bg_clr = np.array(background_clr)[::-1].reshape(1, 1, 3).astype(np.uint8)

        # Body frame -> BEV frame
        right = -left
        bev_rotation = np.array([forward, right, np.cross(forward, right)])
        bev_rotation = Pose.from_rotation_translation(bev_rotation, tvec=np.zeros(3))
        self._bev_rotation = bev_rotation

        self._center_pixel = (
            int((metric_width * pixels_per_meter) // 2 - pixels_per_meter * center_offset_w),
            int((metric_height * pixels_per_meter) // 2 - pixels_per_meter * center_offset_h)
        )
        self.reset()

    def __repr__(self):
        return 'width: {}, height: {}, data: {}'.format(self._metric_width, self._metric_height, type(self.data))

    def reset(self):
        """Reset the canvas to a blank image with guideline circles of various radii.
        """
        self.data = np.ones(
            (int(self._metric_height * self._pixels_per_meter), int(self._metric_width * self._pixels_per_meter), 3),
            dtype=np.uint8
        ) * self._bg_clr

        # Draw metric polar grid
        for i in range(1, int(max(self._metric_width, self._metric_height)) // self._polar_step_size_meters):
            cv2.circle(
                self.data, self._center_pixel, int(i * self._polar_step_size_meters * self._pixels_per_meter),
                (50, 50, 50), 1
            )

    def render_point_cloud(self, point_cloud, extrinsics=Pose(), color=GRAY):
        """Render point cloud in BEV perspective.

        Parameters
        ----------
        point_cloud: numpy array with shape (N, 3)
            3D cloud points in the sensor coordinate frame.

        extrinsics: Pose, default: Identity pose
            The pose of the pointcloud sensor wrt the body frame (Sensor frame -> (Vehicle) Body frame).

        color: Tuple[int]
            Color in RGB to render the points.
        """

        combined_transform = self._bev_rotation * extrinsics

        pointcloud_in_bev = combined_transform * point_cloud
        point_cloud2d = pointcloud_in_bev[:, :2]

        point_cloud2d[:, 0] = (self._center_pixel[0] + point_cloud2d[:, 0] * self._pixels_per_meter)
        point_cloud2d[:, 1] = (self._center_pixel[1] + point_cloud2d[:, 1] * self._pixels_per_meter)

        H, W = self.data.shape[:2]
        uv = point_cloud2d.astype(np.int32)
        in_view = np.logical_and.reduce([
            (point_cloud2d >= 0).all(axis=1),
            point_cloud2d[:, 0] < W,
            point_cloud2d[:, 1] < H,
        ])
        uv = uv[in_view]
        self.data[uv[:, 1], uv[:, 0], :] = color

    def render_radar_point_cloud(
        self, point_cloud, extrinsics=Pose(), color=RED, velocity=None, velocity_scale=1, velocity_max_pix=.05
    ):
        """Render radar point cloud in BEV perspective.

        Parameters
        ----------
        radar_point_cloud: numpy array with shape (N, 3)
            point cloud in rectangular coordinates of sensor frame

        extrinsics: Pose, default: Identity pose
            The pose of the pointcloud sensor wrt the body frame (Sensor frame -> (Vehicle) Body frame).

        color: Tuple[int]
            Color in RGB to render the points.

        velocity: numpy array with shape (N,3), default None
            velocity vector of points

        velocity_scale: float
            factor to scale velocity vector by

        velocity_max_pix: float
            Maximum length of velocity vector rendering in percent of image width
        """
        combined_transform = self._bev_rotation * extrinsics

        pointcloud_in_bev = combined_transform * point_cloud
        point_cloud2d = pointcloud_in_bev[:, :2]

        point_cloud2d[:, 0] = (self._center_pixel[0] + point_cloud2d[:, 0] * self._pixels_per_meter)
        point_cloud2d[:, 1] = (self._center_pixel[1] + point_cloud2d[:, 1] * self._pixels_per_meter)

        H, W = self.data.shape[:2]
        uv = point_cloud2d.astype(np.int32)
        in_view = np.logical_and.reduce([
            (point_cloud2d >= 0).all(axis=1),
            point_cloud2d[:, 0] < W,
            point_cloud2d[:, 1] < H,
        ])
        uv = uv[in_view]

        for row in uv:
            cx, cy = row
            cv2.circle(self.data, (cx, cy), 7, RED, thickness=1)

        def clip_norm(v, x):
            M = np.linalg.norm(v)
            if M == 0:
                return v
            return np.clip(M, 0, x) * v / M

        if velocity is not None:
            tail = point_cloud + velocity_scale * velocity
            pointcloud_in_bev_tail = combined_transform * tail
            point_cloud2d_tail = pointcloud_in_bev_tail[:, :2]
            point_cloud2d_tail[:, 0] = (self._center_pixel[0] + point_cloud2d_tail[:, 0] * self._pixels_per_meter)
            point_cloud2d_tail[:, 1] = (self._center_pixel[1] + point_cloud2d_tail[:, 1] * self._pixels_per_meter)
            uv_tail = point_cloud2d_tail.astype(np.int32)
            uv_tail = uv_tail[in_view]
            for row, row_tail in zip(uv, uv_tail):
                v_2d = row_tail - row
                v_2d = clip_norm(v_2d, velocity_max_pix * W)

                cx, cy = row
                cx2, cy2 = row + v_2d.astype(np.int)

                cx2 = np.clip(cx2, 0, W - 1)
                cy2 = np.clip(cy2, 0, H - 1)
                color = GREEN
                # If moving away from vehicle change the color (not strictly correct because radar is not a (0,0))
                # TODO: calculate actual radar sensor position
                if np.dot(row - np.array([W / 2, H / 2]), v_2d) > 0:
                    color = (255, 110, 199)
                cv2.arrowedLine(self.data, (cx, cy), (cx2, cy2), color, thickness=1, line_type=cv2.LINE_AA)

    def render_paths(self, paths, extrinsics=Pose(), colors=(GREEN, ), line_thickness=1, tint=1.0):
        """Render object paths on bev.

        Parameters
        ----------
        paths: list[list[Pose]]
            List of object poses in the coordinate frame of the current timestep.

        extrinsics: Pose, default: Identity pose
            The pose of the pointcloud sensor wrt the body frame (Sensor frame -> (Vehicle) Body frame).

        colors: List of RGB tuple, default: [GREEN,]
            Draw path using this color.

        line_thickness: int, default: 1
            Thickness of lines.

        tint: float, default: 1.0
            Mulitiplicative factor applied to color used to darken lines.
        """

        if len(colors) == 1:
            colors = list(colors) * len(paths)

        if tint != 1.0:
            colors = [[int(tint * c) for c in color] for color in colors]

        combined_transform = self._bev_rotation * extrinsics

        for path, color in zip(paths, colors):
            # path should contain a list of Pose objects or None types. None types will be skipped.
            # TODO: add option to interpolate skipped poses.
            path3d = [combined_transform * pose.tvec.reshape(1, 3) for pose in path if pose is not None]
            path2d = np.round(self._pixels_per_meter * np.stack(path3d, 0)[..., :2],
                              0).astype(np.int32).reshape(1, -1, 2)
            offset = np.array(self._center_pixel).reshape(1, 1, 2)  # pylint: disable=E1121
            path2d = path2d + offset
            # TODO: if we group the paths by color we can draw all paths with the same color at once
            cv2.polylines(self.data, path2d, 0, color, line_thickness, cv2.LINE_AA)

    def render_bounding_box_3d(
        self,
        bboxes3d,
        extrinsics=Pose(),
        colors=(GREEN, ),
        side_color_fraction=0.7,
        rear_color_fraction=0.5,
        texts=None,
        line_thickness=2,
        font_scale=0.5,
        font_colors=(WHITE, ),
        markers=None,
        marker_scale=.5,
        marker_colors=(RED, ),
    ):
        """Render bounding box 3d in BEV perspective.

        Parameters
        ----------
        bboxes3d: List of BoundingBox3D
            3D annotations in the sensor coordinate frame.

        extrinsics: Pose, default: Identity pose
            The pose of the pointcloud sensor wrt the body frame (Sensor frame -> (Vehicle) Body frame).

        colors: List of RGB tuple, default: [GREEN,]
            Draw boxes using this color.

        side_color_fraction: float, default: 0.6
            A fraction in brightness of side edge colors of bounding box wrt the front face.

        rear_color_fraction: float, default: 0.3
            A fraction in brightness of rear face colors of bounding box wrt the front face.

        texts: list of str, default: None
            3D annotation category name.

        line_thickness: int, default: 2
            Thickness of lines.

        font_scale: float, default: 0.5
            Font scale used for text labels.

        font_colors: List of RGB tuple, default: [WHITE,]
            Color used for text labels.

        markers: List[int], default: None
            List of opencv markers to draw in bottom right corner of cuboid. Should be one of: 
            cv2.MARKER_CROSS, cv2.MARKER_DIAMOND, cv2.MARKER_SQUARE, cv2.MARKER_STAR, cv2.MARKER_TILTED_CROSS, cv2.MARKER_TRIANGLE_DOWN, cv2.MARKER_TRIANGLE_UP, or None.

        marker_scale: float, default: .5
            Scale factor for markers,

        marker_colors: List of RGB Tuple, default: [RED,]
            Draw markers using this color.
        """

        if len(colors) == 1:
            colors = list(colors) * len(bboxes3d)

        if len(font_colors) == 1:
            font_colors = list(font_colors) * len(bboxes3d)

        if len(marker_colors) == 1:
            marker_colors = list(marker_colors) * len(bboxes3d)

        combined_transform = self._bev_rotation * extrinsics

        # Draw cuboids
        for bidx, (bbox, color) in enumerate(zip(bboxes3d, colors)):
            # Create 3 versions of colors for face coding.
            front_face_color = color
            side_line_color = [int(side_color_fraction * c) for c in color]
            rear_face_color = [int(rear_color_fraction * c) for c in color]

            # Do orthogonal projection and bring into pixel coordinate space
            corners = bbox.corners
            corners_in_bev = combined_transform * corners
            corners2d = corners_in_bev[[0, 1, 5, 4], :2]  # top surface of cuboid

            # Compute the center and offset of the corners
            corners2d[:, 0] = (self._center_pixel[0] + corners2d[:, 0] * self._pixels_per_meter)
            corners2d[:, 1] = (self._center_pixel[1] + corners2d[:, 1] * self._pixels_per_meter)

            center = np.mean(corners2d, axis=0).astype(np.int32)
            corners2d = corners2d.astype(np.int32)

            # Draw front face, side faces and back face
            cv2.line(self.data, tuple(corners2d[0]), tuple(corners2d[1]), front_face_color, line_thickness, cv2.LINE_AA)
            cv2.line(self.data, tuple(corners2d[1]), tuple(corners2d[2]), side_line_color, line_thickness, cv2.LINE_AA)
            cv2.line(self.data, tuple(corners2d[2]), tuple(corners2d[3]), rear_face_color, line_thickness, cv2.LINE_AA)
            cv2.line(self.data, tuple(corners2d[3]), tuple(corners2d[0]), side_line_color, line_thickness, cv2.LINE_AA)

            # Draw white light connecting center and font side.
            cv2.arrowedLine(
                self.data, tuple(center), (
                    (corners2d[0][0] + corners2d[1][0]) // 2,
                    (corners2d[0][1] + corners2d[1][1]) // 2,
                ), WHITE, 1, cv2.LINE_AA
            )

            if texts:
                if texts[bidx] is not None:
                    top_left = np.argmin(np.linalg.norm(corners2d, axis=1))
                    cv2.putText(
                        self.data, texts[bidx], tuple(corners2d[top_left]), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        font_colors[bidx], line_thickness // 2, cv2.LINE_AA
                    )

            if markers:
                if markers[bidx] is not None:
                    bottom_right = np.argmax(np.linalg.norm(corners2d, axis=1))

                    assert markers[bidx] in [
                        cv2.MARKER_CROSS, cv2.MARKER_DIAMOND, cv2.MARKER_SQUARE, cv2.MARKER_STAR,
                        cv2.MARKER_TILTED_CROSS, cv2.MARKER_TRIANGLE_DOWN, cv2.MARKER_TRIANGLE_UP
                    ]

                    cv2.drawMarker(
                        self.data, tuple(corners2d[bottom_right]), marker_colors[bidx], markers[bidx],
                        int(20 * marker_scale), 2, cv2.LINE_AA
                    )

    def render_camera_frustrum(self, intrinsics, extrinsics, width, color=YELLOW, line_thickness=1):
        """
        Visualize the frustrum of camera by drawing two lines connecting the
        camera center and top-left / top-right corners of image plane.

        Parameters
        ----------
        intrinsics: np.ndarray
            3x3 intrinsics matrix

        extrinsics: Pose
            Pose of camera in body frame.

        width: int
            Width of image.

        color: Tuple[int], default: Yellow
            Color in RGB of line.

        line_thickness: int, default: 1
            Thickness of line.
        """

        K_inv = np.linalg.inv(intrinsics)

        top_corners_2d = np.array([[0, 0, 1], [width, 0, 1]], np.float64)

        top_corners_3d = np.dot(top_corners_2d, K_inv.T)
        frustrum_in_cam = np.vstack([np.zeros((1, 3), np.float64), top_corners_3d])
        frustrum_in_body = extrinsics * frustrum_in_cam
        frustrum_in_bev = self._bev_rotation * frustrum_in_body

        # Compute the center and offset of the corners
        frustrum_in_bev = frustrum_in_bev[:, :2]
        frustrum_in_bev[:, 0] = (self._center_pixel[0] + frustrum_in_bev[:, 0] * self._pixels_per_meter)
        frustrum_in_bev[:, 1] = (self._center_pixel[1] + frustrum_in_bev[:, 1] * self._pixels_per_meter)

        frustrum_in_bev[1:] = (100 * (frustrum_in_bev[1:] - frustrum_in_bev[0]) + frustrum_in_bev[0])
        frustrum_in_bev = frustrum_in_bev.astype(np.int32)

        cv2.line(self.data, tuple(frustrum_in_bev[0]), tuple(frustrum_in_bev[1]), color, line_thickness)
        cv2.line(self.data, tuple(frustrum_in_bev[0]), tuple(frustrum_in_bev[2]), color, line_thickness)


def ontology_to_viz_colormap(ontology, void_class_id=255):
    """Grabs semseg viz-ready colormap from DGP Ontology object

    Parameters
    ----------
    ontology: dgp.proto.ontology_pb2.Ontology
        DGP ontology object for which we want to create a viz-friendly colormap look-up

    void_class_id: int, default: 255
        Class ID used to denote VOID or IGNORE in ontology

    Returns
    -------
    colormap: np.int64 array
        Shape is (num_classes, 3), where num_classes includes the VOID class
        `colormap[i, :]` is the BGR triplet for class ID i, with `colormap[-1, :]` being the color for the VOID class

    Notes
    -----
    Class ID's are assumed to be contiguous and 0-indexed, with VOID class as the last item in the ontology
    (if VOID is not in the ontology, then black color is assigned to it and appended to colormap)
    """
    colormap = [color for id, color in ontology.colormap.items()]

    # If ontology does not include void class as the last item, then manually append a color for black
    if ontology.class_ids[-1] != void_class_id:
        colormap.append((0, 0, 0))

    return np.array(colormap)


def visualize_semantic_segmentation_2d(
    semantic_segmentation_2d, ontology, void_class_id=255, image=None, alpha=0.3, debug=False
):
    """Constructs a visualization of a semseg frame (either ground truth or predictions), provided an ontology

    Parameters
    ----------
    semantic_segmentation_2d: np.array
        Per-pixel class ID's for a single input image, with `void_class_id` being the IGNORE class
        shape: (H, W)

    ontology: dgp.proto.ontology_pb2.Ontology
        Ontology under which we want to visualize the semseg frame

    void_class_id: int, default: 255
        ID in `semantic_segmentation_2d` that denotes VOID or IGNORE

    image: np.uint8 array, default: None
        If specified then will blend image into visualization with weight `alpha`

    alpha: float, default: 0.3
        If `image` is specified, then will visualize an image/semseg blend with `alpha` weight given to image

    debug: bool, default: True
        If True then visualize frame to display

    Returns
    -------
    colored_semseg: np.uint8
        Visualization of `semantic_segmentation_2d` frame, *in BGR*
        shape: (H, W, 3)
    """
    height, width = semantic_segmentation_2d.shape

    # Convert colormap to (num_classes, 3), where num_classes includes VOID and last entry is color for VOID
    colormap = ontology_to_viz_colormap(ontology, void_class_id=void_class_id)

    # Color per-pixel predictions using the generated color map
    colored_semseg = np.copy(semantic_segmentation_2d).astype(np.uint8)
    colored_semseg[semantic_segmentation_2d == void_class_id] = len(colormap) - 1
    colored_semseg = colormap[colored_semseg.flatten()]
    colored_semseg = colored_semseg.reshape(height, width, 3).astype(np.uint8)

    if image is not None:
        colored_semseg = (alpha * image + (1 - alpha) * colored_semseg).astype(np.uint8)

    if debug:
        cv2.imshow('image', colored_semseg)
        cv2.waitKey(DEBUG_WAIT_TIME)

    return colored_semseg


def visualize_bev(
    lidar_datums,
    class_colormap,
    show_instance_id_on_bev=False,
    id_to_name=None,
    camera_datums=None,
    camera_colors=None,
    bev_metric_width=100,
    bev_metric_height=100,
    bev_pixels_per_meter=10,
    bev_polar_step_size_meters=10,
    bev_forward=(1, 0, 0),
    bev_left=(0, 1, 0),
    bev_background_clr=(0, 0, 0),
    bev_line_thickness=4,
    bev_font_scale=0.5,
    radar_datums=None,
    instance_colormap=None,
    cuboid_caption_fn=None,
    marker_fn=None,
    marker_scale=.5,
    show_paths_on_bev=False,
    bev_center_offset_w=0,
    bev_center_offset_h=0,
):
    """Create BEV visualization that shows pointcloud, 3D bounding boxes, and optionally camera frustrums.
    Parameters
    ----------
    lidar_datums: List[OrderedDict]
        List of lidar datums as a dictionary.
    class_colormap: Dict
        Mapping from class IDs to RGB colors.
    show_instance_id_on_bev: Bool, default: False
        If True, then show `instance_id` on a corner of 3D bounding boxes in BEV view.
        If False, then show `class_name` instead.
    id_to_name: OrderedDict, default: None
        Mapping from class IDs to class names.
    camera_datums: List[OrderedDict], default: None
        List of camera datums as a dictionary.
    camera_colors: List[Tuple[int]], default: None
        List of RGB colors associated with each camera. The colors are used to draw frustrum.
    bev_*:
        See `BEVImage` for these keyword arguments.
    radar_datums: List[OrderedDict], default: None
        List of radar datums to visualize
    instance_colormap: Dict
        Mapping from instance id to RGB colors.
    cuboid_caption_fn: Callable, BoundingBox3d -> Tuple[String,Tuple[3]]
        Function taking a BoundingBox3d object and returning a tuple with the caption string, and the rgb
        value for that caption. e.g., ( 'car', (255,0,0) )
    marker_fn: Callable, BoundingBox3d -> Tuple[int,Tuple[3]]
        Function taking a BoundingBox3d object and returning a tuple with the caption a marker id, and the rgb
        value for that marker. e.g., ( cv2.MARKER_DIAMOND, (255,0,0) ). Marker should be one of
        cv2.MARKER_CROSS, cv2.MARKER_DIAMOND, cv2.MARKER_SQUARE, cv2.MARKER_STAR, cv2.MARKER_TILTED_CROSS, cv2.MARKER_TRIANGLE_DOWN, cv2.MARKER_TRIANGLE_UP, or None.
    show_paths_on_bev: Bool, default: False
        If true draw a path for each cuboid. Paths are stored in cuboid attributes under the 'path' key, i.e.,
        path = cuboid.attributes['path'], paths are themselves a list of pose objects transformed to the
        correct frame. This method does not handle creating or transforming the paths.
    bev_enter_offset_w: int, default: 0
        Offset in pixels to move ego center in BEV.
    bev_center_offset_h: int, default: 0
        Offset in pixels to move ego center in BEV.

    Returns
    -------
    np.ndarray
        BEV visualization as an image.
    """
    bev = BEVImage(
        bev_metric_width,
        bev_metric_height,
        bev_pixels_per_meter,
        bev_polar_step_size_meters,
        bev_forward,
        bev_left,
        bev_background_clr,
        center_offset_w=bev_center_offset_w,
        center_offset_h=bev_center_offset_h
    )

    # 1. Render pointcloud
    if len(lidar_datums) > 1:
        pc_colors = get_unique_colors(len(lidar_datums))
    else:
        pc_colors = [GRAY]
    for lidar_datum, clr in zip(lidar_datums, pc_colors):
        bev.render_point_cloud(lidar_datum['point_cloud'], lidar_datum['extrinsics'], color=clr)

    # 2. Render radars
    if radar_datums is not None:
        for radar_datum in radar_datums:
            bev.render_radar_point_cloud(
                radar_datum['point_cloud'], radar_datum['extrinsics'], velocity=radar_datum['velocity']
            )

    # 3. Render 3D bboxes.
    for lidar_datum in lidar_datums:
        if 'bounding_box_3d' in lidar_datum:

            if len(lidar_datum['bounding_box_3d']) == 0:
                continue

            if instance_colormap is not None:
                colors = [
                    instance_colormap.get(bbox.instance_id, class_colormap[bbox.class_id])
                    for bbox in lidar_datum['bounding_box_3d']
                ]
            else:
                colors = [class_colormap[bbox.class_id] for bbox in lidar_datum['bounding_box_3d']]

            # If no caption function is supplied, generate one from the instance ids or class ids
            # Caption functions should return a tuple (string, color)
            # TODO: expand to include per caption font size.
            if show_instance_id_on_bev and cuboid_caption_fn is None:
                cuboid_caption_fn = lambda x: (str(x.instance_id), WHITE)
            elif cuboid_caption_fn is None:  # show class names
                cuboid_caption_fn = lambda x: (id_to_name[x.class_id], WHITE)

            labels, font_colors = zip(*[cuboid_caption_fn(bbox3d) for bbox3d in lidar_datum['bounding_box_3d']])

            markers, marker_colors = None, (RED, )
            if marker_fn is not None:
                markers, marker_colors = zip(*[marker_fn(bbox3d) for bbox3d in lidar_datum['bounding_box_3d']])

            bev.render_bounding_box_3d(
                lidar_datum['bounding_box_3d'],
                lidar_datum['extrinsics'],
                colors=colors,
                texts=labels if bev_font_scale > 0 else None,
                line_thickness=bev_line_thickness,
                font_scale=bev_font_scale,
                font_colors=font_colors,
                markers=markers if marker_scale > 0 else None,
                marker_scale=marker_scale,
                marker_colors=marker_colors,
            )

            if show_paths_on_bev:
                # Collect the paths and path colors
                paths, path_colors = zip(
                    *[(bbox.attributes['path'], c)
                      for bbox, c in zip(lidar_datum['bounding_box_3d'], colors)
                      if 'path' in bbox.attributes]
                )
                if len(paths) > 0:
                    bev.render_paths(paths, extrinsics=lidar_datum['extrinsics'], colors=path_colors, line_thickness=1)

    # 4. Render camera frustrums.
    if camera_datums is not None:
        for cam_datum, cam_color in zip(camera_datums, camera_colors):
            bev.render_camera_frustrum(
                cam_datum['intrinsics'],
                cam_datum['extrinsics'],
                cam_datum['rgb'].size[0],
                color=cam_color,
                line_thickness=bev_line_thickness // 2
            )

    return bev.data


def visualize_cameras(
    camera_datums,
    id_to_name,
    lidar_datums=None,
    rgb_resize_factor=1.0,
    # `BoundingBox3D` kwargs
    bbox3d_font_scale=1.0,
    bbox3d_line_thickness=4,
    # `render_pointcloud_on_image` kwargs
    pc_rgb_cmap=MPL_JET_CMAP,
    pc_rgb_norm_depth=10,
    pc_rgb_dilation=3,
    radar_datums=None,
):
    """Create camera visualization that shows 3D bounding boxes, and optionally projected pointcloud.
    Parameters
    ----------
    camera_datums: List[OrderedDict], default: None
        List of camera datums as a dictionary.
    id_to_name: OrderedDict, default: None
        Mapping from class IDs to class names.
    lidar_datums: List[OrderedDict] or None, default: None
        List of lidar datums as a dictionary. If given, then draw pointcloud contained in all datums.
    rgb_resize_factor: float, default: 1.0
        Resize images by this factor before tiling them into a single panel.
    bbox3d_font_scale: float, default: 1.0
        Font scale used for text labels.
    bbox3d_line_thickness: int, default: 4
        Thickness of lines used for drawing 3D bounding boxes.
    pc_rgb_norm_depth: int, default: 10
        Depth value to normalize (inverse) pointcloud depths in color mapping.
    pc_rgb_dilation: int, default: 3
        Dilation factor applied on each point in pointcloud.
    radar_datums: List[OrderedDict], default: None
        List of radar datums to visualize
    Returns
    -------
    rgb_viz: List[np.ndarray]
        List of camera visualization images, one for each camera.
    """
    rgb_viz = []
    for cam_datum in camera_datums:
        rgb = np.array(cam_datum['rgb']).copy()

        if lidar_datums is not None:
            # 1. Render pointcloud
            for lidar_datum in lidar_datums:
                p_LC = cam_datum['extrinsics'].inverse() * lidar_datum['extrinsics']  # lidar -> body -> camera
                rgb = render_pointcloud_on_image(
                    rgb,
                    Camera(K=cam_datum['intrinsics'], p_cw=p_LC),
                    lidar_datum['point_cloud'],
                    cmap=pc_rgb_cmap,
                    norm_depth=pc_rgb_norm_depth,
                    dilation=pc_rgb_dilation
                )

        if radar_datums is not None:
            # 2. Render radar pointcloud
            for radar_datum in radar_datums:
                p_LC = cam_datum['extrinsics'].inverse() * radar_datum['extrinsics']  # radar -> body -> camera
                rgb = render_radar_pointcloud_on_image(
                    rgb,
                    Camera(K=cam_datum['intrinsics'], p_cw=p_LC),
                    radar_datum['point_cloud'],
                    norm_depth=pc_rgb_norm_depth,
                    velocity=radar_datum['velocity']
                )

        # 3. Render 3D bboxes
        # for bbox3d, class_id in zip(cam_datum['bounding_box_3d'], cam_datum['class_ids']):
        if 'bounding_box_3d' in cam_datum:
            for bbox3d in cam_datum['bounding_box_3d']:
                class_name = id_to_name[bbox3d.class_id]
                rgb = bbox3d.render(
                    rgb,
                    Camera(K=cam_datum['intrinsics']),
                    line_thickness=bbox3d_line_thickness,
                    class_name=class_name,
                    font_scale=bbox3d_font_scale,
                )

        rgb_viz.append(cv2.resize(rgb, None, fx=rgb_resize_factor, fy=rgb_resize_factor))

    return rgb_viz


def visualize_agent_bev(
    agent_datums,
    lidar_datums,
    class_colormap,
    show_instance_id_on_bev=False,
    id_to_name=None,
    camera_datums=None,
    camera_colors=None,
    bev_metric_width=100,
    bev_metric_height=100,
    bev_pixels_per_meter=10,
    bev_polar_step_size_meters=10,
    bev_forward=(1, 0, 0),
    bev_left=(0, 1, 0),
    bev_background_clr=(0, 0, 0),
    bev_line_thickness=4,
    bev_font_scale=0.5,
    radar_datums=None,
    instance_colormap=None,
    cuboid_caption_fn=None,
    marker_fn=None,
    marker_scale=.5,
    show_paths_on_bev=False,
    bev_center_offset_w=0,
    bev_center_offset_h=0,
):
    """Create BEV visualization that shows pointcloud, 3D bounding boxes, and optionally camera frustrums.
    Parameters
    ----------
    agent_datums: List[BoundingBox3D]
        List of agent bounding boxes in frame
    lidar_datums: List[OrderedDict]
        List of lidar datums as a dictionary.
    class_colormap: Dict
        Mapping from class IDs to RGB colors.
    show_instance_id_on_bev: Bool, default: False
        If True, then show `instance_id` on a corner of 3D bounding boxes in BEV view.
        If False, then show `class_name` instead.
    id_to_name: OrderedDict, default: None
        Mapping from class IDs to class names.
    camera_datums: List[OrderedDict], default: None
        List of camera datums as a dictionary.
    camera_colors: List[Tuple[int]], default: None
        List of RGB colors associated with each camera. The colors are used to draw frustrum.
    bev_*:
        See `BEVImage` for these keyword arguments.
    radar_datums: List[OrderedDict], default: None
        List of radar datums to visualize
    instance_colormap: Dict
        Mapping from instance id to RGB colors.
    cuboid_caption_fn: Callable, BoundingBox3d -> Tuple[String,Tuple[3]]
        Function taking a BoundingBox3d object and returning a tuple with the caption string, and the rgb
        value for that caption. e.g., ( 'car', (255,0,0) )
    marker_fn: Callable, BoundingBox3d -> Tuple[int,Tuple[3]]
        Function taking a BoundingBox3d object and returning a tuple with the caption a marker id, and the rgb
        value for that marker. e.g., ( cv2.MARKER_DIAMOND, (255,0,0) ). Marker should be one of
        cv2.MARKER_CROSS, cv2.MARKER_DIAMOND, cv2.MARKER_SQUARE, cv2.MARKER_STAR, cv2.MARKER_TILTED_CROSS, cv2.MARKER_TRIANGLE_DOWN, cv2.MARKER_TRIANGLE_UP, or None.
    show_paths_on_bev: Bool, default: False
        If true draw a path for each cuboid. Paths are stored in cuboid attributes under the 'path' key, i.e.,
        path = cuboid.attributes['path'], paths are themselves a list of pose objects transformed to the
        correct frame. This method does not handle creating or transforming the paths.
    bev_enter_offset_w: int, default: 0
        Offset in pixels to move ego center in BEV.
    bev_center_offset_h: int, default: 0
        Offset in pixels to move ego center in BEV.

    Returns
    -------
    np.ndarray
        BEV visualization as an image.
    """
    bev = BEVImage(
        bev_metric_width,
        bev_metric_height,
        bev_pixels_per_meter,
        bev_polar_step_size_meters,
        bev_forward,
        bev_left,
        bev_background_clr,
        center_offset_w=bev_center_offset_w,
        center_offset_h=bev_center_offset_h
    )

    # 1. Render pointcloud
    if len(lidar_datums) > 1:
        pc_colors = get_unique_colors(len(lidar_datums))
    else:
        pc_colors = [GRAY]
    for lidar_datum, clr in zip(lidar_datums, pc_colors):
        bev.render_point_cloud(lidar_datum['point_cloud'], lidar_datum['extrinsics'], color=clr)

    # 2. Render radars
    if radar_datums is not None:
        for radar_datum in radar_datums:
            bev.render_radar_point_cloud(
                radar_datum['point_cloud'], radar_datum['extrinsics'], velocity=radar_datum['velocity']
            )

    # 3. Render 3D bboxes.
    #for agent_datum in agent_datums:
    if instance_colormap is not None:
        colors = [
            instance_colormap.get(agent_datum.instance_id, class_colormap[agent_datum.class_id])
            for agent_datum in agent_datums
        ]
    else:
        colors = [class_colormap[agent_datum.class_id] for agent_datum in agent_datums]

    # If no caption function is supplied, generate one from the instance ids or class ids
    # Caption functions should return a tuple (string, color)
    # TODO: expand to include per caption font size.
    if show_instance_id_on_bev and cuboid_caption_fn is None:
        cuboid_caption_fn = lambda x: (str(x.instance_id), WHITE)
    elif cuboid_caption_fn is None:  # show class names
        cuboid_caption_fn = lambda x: (id_to_name[x.class_id], WHITE)

    labels, font_colors = zip(*[cuboid_caption_fn(agent_datum) for agent_datum in agent_datums])

    markers, marker_colors = None, (RED, )
    if marker_fn is not None:
        markers, marker_colors = zip(*[marker_fn(agent_datum) for agent_datum in agent_datums])

    bev.render_bounding_box_3d(
        agent_datums,
        lidar_datums[0]['extrinsics'],
        colors=colors,
        texts=labels if bev_font_scale > 0 else None,
        line_thickness=bev_line_thickness,
        font_scale=bev_font_scale,
        font_colors=font_colors,
        markers=markers if marker_scale > 0 else None,
        marker_scale=marker_scale,
        marker_colors=marker_colors,
    )

    if show_paths_on_bev:
        # Collect the paths and path colors
        paths, path_colors = zip(
            *[(bbox.attributes['path'], c) for bbox, c in zip(agent_datums, colors) if 'path' in bbox.attributes]
        )
        if len(paths) > 0:
            bev.render_paths(paths, extrinsics=lidar_datums[0]['extrinsics'], colors=path_colors, line_thickness=1)

    # 4. Render camera frustrums.
    if camera_datums is not None:
        for cam_datum, cam_color in zip(camera_datums, camera_colors):
            bev.render_camera_frustrum(
                cam_datum['intrinsics'],
                cam_datum['extrinsics'],
                cam_datum['rgb'].size[0],
                color=cam_color,
                line_thickness=bev_line_thickness // 2
            )

    return bev.data
