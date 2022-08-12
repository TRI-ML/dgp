# Copyright 2021-2022 Toyota Research Institute.  All rights reserved.
import hashlib

import cv2
import numpy as np

import dgp.proto.annotations_pb2 as annotations_pb2
from dgp.utils.camera import Camera
from dgp.utils.colors import BLUE, GRAY, GREEN, RED
from dgp.utils.pose import Pose
from dgp.utils.structures.bounding_box_2d import GENERIC_OBJECT_CLASS


class BoundingBox3D:
    """3D bounding box (cuboid) that is centered at `pose` with extent `sizes`.

    Parameters
    ----------
    pose: dgp.utils.pose.Pose, (default: Pose())
        Pose of the center of the 3D cuboid.

    sizes: np.float32, (default: np.float32([0,0,0]))
        Extents of the cuboid (width, length, height).

    class_id: int, default: GENERIC_OBJECT_CLASS
        Integer class ID (0 reserved for background).

    instance_id: int, default: None
        Unique instance ID for bounding box. If None provided, the ID is a hash of the bounding box
        location and class.

    color: tuple, default: (0, 0, 0)
        RGB tuple for bounding box color.

    attributes: dict, default: None
        Dictionary of attributes associated with bounding box. If None provided,
        defaults to empty dict.

    num_points: int, default: 0
        Number of LIDAR points associated with this bounding box.

    occlusion: int, default: 0
        Occlusion state (KITTI3D style).

    truncation: float, default: 0
        Fraction of truncation of object (KITTI3D style).

    feature_ontology_type: dgp.proto.features.FeatureType
        Type of feature of attributions.

    sample_idx: int
        index of sample in scene
    """
    def __init__(
        self,
        pose,
        sizes=np.float32([0, 0, 0]),
        class_id=GENERIC_OBJECT_CLASS,
        instance_id=None,
        color=(0, 0, 0),
        attributes=None,
        num_points=0,
        occlusion=0,
        truncation=0.0,
        feature_ontology_type=None,
        sample_idx=None
    ):
        assert isinstance(pose, Pose)
        assert isinstance(sizes, np.ndarray)
        assert len(sizes) == 3

        self._pose = pose
        self._sizes = sizes
        self._class_id = class_id
        self._instance_id = instance_id
        self._color = color
        self._attributes = dict(attributes) if attributes is not None else {}
        self._num_points = num_points
        self._occlusion = occlusion
        self._truncation = truncation
        self._feature_ontology_type = feature_ontology_type
        self._sample_idx = sample_idx

    @property
    def pose(self):
        return self._pose

    @property
    def sizes(self):
        return self._sizes

    @property
    def num_points(self):
        return self._num_points

    @property
    def occlusion(self):
        return self._occlusion

    @property
    def truncation(self):
        return self._truncation

    @property
    def class_id(self):
        return self._class_id

    @class_id.setter
    def class_id(self, class_id):
        self._class_id = class_id

    @property
    def instance_id(self):
        if self._instance_id is None:
            return self.hexdigest
        return self._instance_id

    @property
    def attributes(self):
        return self._attributes

    @property
    def feature_ontology_type(self):
        return self._feature_ontology_type

    @property
    def sample_idx(self):
        return self._sample_idx

    @property
    def vectorize(self):
        """Get a np.ndarray of with 10 dimensions representing the 3D bounding box

        Returns
        -------
        box_3d: np.float32 array
            Box with coordinates (pose.quat.qw, pose.quat.qx, pose.quat.qy, pose.quat.qz,
            pose.tvec.x, pose.tvec.y, pose.tvec.z, width, length, height).
        """
        return np.float32([
            self.pose.quat.elements[0],
            self.pose.quat.elements[1],
            self.pose.quat.elements[2],
            self.pose.quat.elements[3],
            self.pose.tvec[0],
            self.pose.tvec[1],
            self.pose.tvec[2],
            self.sizes[0],
            self.sizes[1],
            self.sizes[2],
        ])

    @property
    def corners(self):
        """Get 8 corners of the 3D bounding box.
        Note: The pose is oriented such that x-forward, y-left, z-up.
        This corresponds to L (length) along x, W (width) along y, and H
        (height) along z.


        Returns
        ----------
        corners: np.ndarray (8 x 3)
            Corners of the 3D bounding box.
        """
        W, L, H = self._sizes
        x_corners = L / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = W / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = H / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners)).T
        return self._pose * corners

    @property
    def edges(self):
        """Get the 12 edge links of the 3D bounding box, indexed by the corners
        defined by `self.corners``.

        Returns
        ----------
        edges: np.ndarray (12 x 2)
            Edge links of the 3D bounding box.
        """
        return np.int32([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ])

    def __rmul__(self, other):
        """Rigidly transform BoundingBox3D with a Pose object. (transformed_box3d = pose * bbox_3d)

        Parameters
        ----------
        other: Pose
            See `utils.pose.py`.

        Returns
        ----------
        result: BoundingBox3D
            Bounding box with transformed pose.
        """
        if isinstance(other, Pose):
            return BoundingBox3D(
                other * self._pose, self._sizes, self._class_id, self._instance_id, self._color, self._attributes,
                self._num_points, self._occlusion, self._truncation
            )
        else:
            return NotImplemented

    @property
    def hexdigest(self):
        return hashlib.md5(self.vectorize.tobytes() + bytes(self._class_id)).hexdigest()

    def __eq__(self, other):
        return self.hexdigest == other.hexdigest

    def __repr__(self):
        return '{}[Pose: {}, (W: {}, L: {}, H: {}), Points: {}, Class: {}, InstanceID: {}, Attributes: {}]'.format(
            self.__class__.__name__, self.pose, self.sizes[0], self.sizes[1], self.sizes[2], self.num_points,
            self.class_id, self.instance_id, self.attributes
        )

    def render(self, image, camera, line_thickness=2, class_name=None, font_scale=0.5):
        """Render the bounding box on the image.

        Parameters
        ----------
        image: np.ndarray
            Image (H, W, C) to render the bounding box onto. We assume the input image is in *RGB* format.
            The type must be uint8.

        camera: dgp.utils.camera.Camera
            Camera used to render the bounding box.

        line_thickness: int, optional
            Thickness of bounding box lines. Default: 2.

        class_name: str, optional
            Class name of the bounding box. Default: None.

        font_scale: float, optional
            Font scale used in text labels. Default: 0.5.

        Returns
        ----------
        image: np.uint8 array
            Rendered image (H, W, 3).

        Raises
        ------
        ValueError
            Raised if image is not a 3-channel uint8 numpy array.
        TypeError
            Raised if camera is not an instance of Camera.
        """
        if (
            not isinstance(image, np.ndarray) or image.dtype != np.uint8 or len(image.shape) != 3 or image.shape[2] != 3
        ):
            raise ValueError('`image` needs to be a 3-channel uint8 numpy array')
        if not isinstance(camera, Camera):
            raise TypeError('`camera` should be of type Camera')

        points2d = camera.project(self.corners)
        corners = points2d.T
        if (self.corners[:, 2] <= 0).any():
            return image

        # while preserving ability to debug object orientation easily
        COLORS = [RED, GREEN, BLUE]

        # Draw the sides (first)
        for i in range(4):
            cv2.line(
                image, (int(corners.T[i][0]), int(corners.T[i][1])),
                (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                GRAY,
                thickness=line_thickness
            )
        # Draw front (in red) and back (blue) face.
        cv2.polylines(image, [corners.T[:4].astype(np.int32)], True, RED, thickness=line_thickness)
        cv2.polylines(image, [corners.T[4:].astype(np.int32)], True, BLUE, thickness=line_thickness)

        # Draw axes on centroid
        vx = self.pose.tvec + self.pose.matrix[:3, 0]
        vy = self.pose.tvec + self.pose.matrix[:3, 1]
        vz = self.pose.tvec + self.pose.matrix[:3, 2]
        p1 = camera.project(np.vstack([self.pose.tvec, self.pose.tvec, self.pose.tvec])).astype(np.int32)
        p2 = camera.project(np.vstack([vx, vy, vz])).astype(np.int32)

        for i in range(3):
            cv2.line(image, (p1[i][0], p1[i][1]), (p2[i][0], p2[i][1]), COLORS[i], line_thickness)

        if class_name:
            cv2.putText(
                image, class_name, tuple(p1[0]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA
            )

        return image

    def to_proto(self):
        """Serialize bounding box to proto object.

        NOTE: Does not serialize class or instance information, just box properties.
        To serialize a complete annotation, see `dgp/annotations/bounding_box_3d_annotation.py`

        Returns
        -------
        BoundingBox3D.pb2
            As defined in `proto/annotations.proto`.
        """
        return annotations_pb2.BoundingBox3D(
            pose=self._pose.to_proto(),
            width=self._sizes[0],
            length=self._sizes[1],
            height=self._sizes[2],
            occlusion=self._occlusion,
            truncation=self._truncation,
        )
