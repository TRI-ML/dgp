# Copyright 2019 Toyota Research Institute.  All rights reserved.
"""Visualization tools for a variety of tasks"""
import numpy as np
from matplotlib.cm import get_cmap

import cv2
from dgp.utils.camera import Camera

COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_GRAY = (100, 100, 100)
COLOR_DARKGRAY = (50, 50, 50)
COLOR_WHITE = (255, 255, 255)

class InstanceColorGenerator:
    """A Class that generates unique color based on instance category.

    Parameters
    ----------
    base_colormap: dict
            A dictionary of colors, mapping class_id to color
    """

    def __init__(self, base_colormap, max_dist = 30):
        """Initialization."""
        # record colors used for instance and stuff identification.
        # if the random color generation happen to generate a used
        # color, it will be rejected.
        self.taken_colors = set([0, 0, 0])
        self.base_colormap = base_colormap
        self.max_dist = max_dist
        for base_color in self.base_colormap.values():
            self.taken_colors.add(tuple(base_color))

    def get_color(self, class_id):
        """Generate a random color for instance based on base category color.

        Parameters
        ----------
        class_id: int
            class_id

        Returns
        -------
        color: A unique color to identify a segment.
        """
        def random_color(base, max_dist=self.max_dist):
            new_color = base + np.random.randint(low=-max_dist, high=max_dist + 1, size=3)
            return tuple(np.maximum(0, np.minimum(255, new_color)))
        base_color = self.base_colormap[class_id]
        while True:
            color = random_color(base_color)
            if color not in self.taken_colors:
                self.taken_colors.add(color)
                return color

def visualize_instance_segmentation_2d(
    instance_masks, instance_class_ids, ontology, image_size, void_class_id=255, class_names=None, image=None, alpha=0.3, debug=False, white_edge=False
):
    """Constructs a visualization of a instance segmentation frame (either ground truth or predictions), provided an ontology

    Parameters
    ----------
    instance_masks: List[np.bool]
            (H, W) bool array for each instance in instance annotation

    instance_ids: List[int]
            Instance lass CIDs for each instance in instance annotation

    image: np.uint8 array, default: None
        If specified then will blend image into visualization with weight `alpha` 

    image_size: List[int]
        [H, W] of target image

    alpha: float, default: 0.3
        If `image` is specified, then will visualize an image/semseg blend with `alpha` weight given to image

    debug: bool, default: True
        If True then visualize frame to display

    Returns
    -------
    colored_instance_seg: np.uint8
        Visualization of `instance_segmentation_2d` frame, *in BGR*
        shape: (H, W, 3)
    """
    H, W = image_size

    colormap = ontology_to_viz_colormap(ontology, void_class_id=void_class_id)

    # create dictionary of colormap keyed by class_id.
    colormap_dict = {class_id: color for class_id, color in enumerate(colormap)}
    color_generator = InstanceColorGenerator(colormap_dict)

    # Color per-pixel predictions using the generated color map
    colored_instance_seg = np.zeros([H, W, 3], dtype='uint8')
    for mask, class_id in zip(instance_masks, instance_class_ids):
        colored_instance_seg[mask] = color_generator.get_color(class_id)
        if white_edge:
            thresh = (mask*255).astype('uint8')
            contours, _ = cv2.findContours(thresh,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour_mask = cv2.drawContours(colored_instance_seg, contours, -1, (255, 255, 255), 3)



    if image is not None:
        colored_instance_seg = (alpha * image + (1 - alpha) * colored_instance_seg).astype(np.uint8)

    if class_names is not None:
        for mask, class_name in zip(instance_masks, class_names):
            mask_loc = np.nonzero(mask)
            x, y = mask_loc[0][0], mask_loc[1][0]
            cv2.putText(colored_instance_seg, class_name, (y, x), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

    if debug:
        cv2.imshow('image', colored_instance_seg)
        cv2.waitKey(DEBUG_WAIT_TIME)

    return colored_instance_seg

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
    cv2.rectangle(image, (0, status_ymin), (status_xmax, H), COLOR_DARKGRAY, thickness=-1)
    cv2.putText(image, '%s' % text, (text_offset, H - text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, thickness=1)
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
    mosaic = np.vstack(hstack) if len(hstack) > 1 \
        else hstack[0]
    return mosaic


def render_bbox2d_on_image(img, bboxes2d, colors=None, texts=None):
    """Render list of bounding box2d on image.

    Parameters
    ----------
    img: np.ndarray
        Image to render bounding boxes onto.

    bboxes2d: np.ndarray (N x 4)
        Array of 2d bounding box (x, y, w, h).

    colors: list
        List of color tuples.

    texts: list, default: None
        List of str classes.

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
        cv2.polylines(img, boxes, True, COLOR_RED, thickness=2)
    else:
        assert len(boxes) == len(colors), 'len(boxes) != len(colors)'
        for idx, box in enumerate(boxes):
            cv2.polylines(img, [box], True, colors[idx], thickness=2)

    # Add texts
    if texts:
        assert len(boxes) == len(texts), 'len(boxes) != len(texts)'
        for idx, box in enumerate(boxes):
            cv2.putText(img, texts[idx], tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        COLOR_WHITE, 2, cv2.LINE_AA)
    return img


def render_pointcloud_on_image(img, camera, Xw, colormap='jet', percentile=80):
    """Render pointcloud on image.

    Parameters
    ----------
    img: np.ndarray
        Image to render bounding boxes onto.

    camera: Camera
        Camera object with appropriately set extrinsics wrt world.

    Xw: np.ndarray (N x 3)
        3D point cloud (x, y, z) in the world coordinate.

    colormap: str, default: jet
        Colormap used for visualizing the inverse depth.

    percentile: float, default: 80
        Use this percentile to normalize the inverse depth.

    Returns
    -------
    img: np.array
        Image with rendered point cloud.
    """
    cmap = get_cmap('jet')
    # Move point cloud to the camera's (C) reference frame from the world (W)
    Xc = camera.p_cw * Xw
    # Project the points as if they were in the camera's frame of reference
    uv = Camera(K=camera.K).project(Xc)
    # Colorize the point cloud based on depth
    z_c = Xc[:, 2]
    zinv_c = 1. / (z_c + 1e-6)
    zinv_c /= np.percentile(zinv_c, percentile)
    colors = (cmap(np.clip(zinv_c, 0., 1.0))[:, :3] * 255).astype(np.uint8)

    # Create an empty image to overlay
    H, W, _ = img.shape
    vis = np.zeros_like(img)
    in_view = np.logical_and.reduce([(uv >= 0).all(axis=1), uv[:, 0] < W, uv[:, 1] < H, z_c > 0])
    uv, colors = uv[in_view].astype(int), colors[in_view]
    vis[uv[:, 1], uv[:, 0]] = colors

    # Dilate visualization so that they render clearly
    vis = cv2.dilate(vis, np.ones((5, 5)))
    return np.maximum(vis, img)


class BEVImage:
    """A class for bird's eye view visualization, which generates a canvas of bird's eye view image,
    This assumes that x-right, y-forward, so the projection will be in the first 2 coordinates 0, 1 (i.e. x-y plane)

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

    x-axis: int, default: 0
        Axis corresponding to the right of the BEV image.

    y-axis: int, default: 1
        Axis corresponding to the forward of the BEV image.
    """
    def __init__(
        self, metric_width=100., metric_height=100., pixels_per_meter=10., polar_step_size_meters=10, xaxis=0, yaxis=1
    ):
        assert xaxis != yaxis, 'Provide different x and y axis coordinates'
        self._metric_width = metric_width
        self._metric_height = metric_height
        self._pixels_per_meter = pixels_per_meter
        self._xaxis = xaxis
        self._yaxis = yaxis
        self._center_pixel = (int(metric_width * pixels_per_meter) // 2, int(metric_height * pixels_per_meter) // 2)
        self.data = np.zeros((int(metric_height * pixels_per_meter), int(metric_width * pixels_per_meter), 3),
                             dtype=np.uint8)

        # Draw metric polar grid
        for i in range(1, int(max(self._metric_width, self._metric_height)) // polar_step_size_meters):
            cv2.circle(
                self.data, self._center_pixel, int(i * polar_step_size_meters * self._pixels_per_meter), (50, 50, 50), 1
            )

    def __repr__(self):
        return 'width: {}, height: {}, data: {}'.format(self._metric_width, self._metric_height, type(self.data))

    def render_point_cloud(self, point_cloud):
        """Render point cloud in BEV perspective.

        Parameters
        ----------
        point_cloud: numpy array with shape (N, 3), default: None
            3D cloud points in the sensor coordinate frame.
        """

        # Draw point-cloud
        point_cloud2d = np.vstack([point_cloud[:, self._xaxis], point_cloud[:, self._yaxis]]).T
        point_cloud2d[:, 0] = self._center_pixel[0] + point_cloud2d[:, 0] * self._pixels_per_meter
        point_cloud2d[:, 1] = self._center_pixel[1] - point_cloud2d[:, 1] * self._pixels_per_meter
        H, W = self.data.shape[:2]
        uv = point_cloud2d.astype(np.int32)
        in_view = np.logical_and.reduce([(point_cloud2d >= 0).all(axis=1), point_cloud2d[:, 0] < W, point_cloud2d[:, 1] < H])
        uv = uv[in_view]
        self.data[uv[:, 1], uv[:, 0], :] = 128

    def render_bounding_box_3d(self, bboxes3d, color=None, texts=None):
        """Render bounding box 3d in BEV perspective.

        Parameters
        ----------
        bboxes3d: list of BoundingBox3D, default: None
            3D annotations in the sensor coordinate frame.

        color: RGB tuple, default: None
            If provided, draw boxes using this color instead of red forward/blue back

        texts: list of str, default: None
            3D annotation category name.
        """

        if color is None:
            colors = [COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_GRAY]
        else:
            colors = [color] * 4

        # Draw cuboids
        for bidx, bbox in enumerate(bboxes3d):
            # Do orthogonal projection and bring into pixel coordinate space
            corners = bbox.corners
            corners2d = np.vstack([corners[:, self._xaxis], corners[:, self._yaxis]]).T

            # Compute the center and offset of the corners
            corners2d[:, 0] = self._center_pixel[0] + corners2d[:, 0] * self._pixels_per_meter
            corners2d[:, 1] = self._center_pixel[1] - corners2d[:, 1] * self._pixels_per_meter
            center = np.mean(corners2d, axis=0).astype(np.int32)
            corners2d = corners2d.astype(np.int32)

            # Draw object center and green line towards front face, unless color specified
            cv2.circle(self.data, tuple(center), 1, COLOR_GREEN)
            cv2.line(self.data, tuple(center), ((corners2d[0][0] + corners2d[1][0])//2, \
                (corners2d[0][1] + corners2d[1][1]) // 2), COLOR_WHITE, 2)

            # Draw front face, side faces and back face
            cv2.line(self.data, tuple(corners2d[0]), tuple(corners2d[1]), colors[0], 2)
            cv2.line(self.data, tuple(corners2d[3]), tuple(corners2d[4]), colors[3], 2)
            cv2.line(self.data, tuple(corners2d[1]), tuple(corners2d[5]), colors[3], 2)
            cv2.line(self.data, tuple(corners2d[4]), tuple(corners2d[5]), colors[2], 2)

            if texts:
                cv2.putText(self.data, texts[bidx], tuple(corners2d[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            COLOR_WHITE, 2, cv2.LINE_AA)


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
    """
    assert all([item.id == i for i, item in enumerate(ontology.items[:-1])]) and ontology.items[-1].id == void_class_id
    colormap = [(item.color.b, item.color.g, item.color.r) for item in ontology.items]
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
