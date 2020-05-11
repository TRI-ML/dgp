# Copyright 2019 Toyota Research Institute.  All rights reserved.
"""General-purpose class for rigid-body transformations.
"""
import cv2
import numpy as np
import torch
from pyquaternion import Quaternion

from dgp.proto import geometry_pb2


class BoundingBox3D:
    """3D bounding box (cuboid) that is centered at `pose` with extent `sizes`.
    """
    def __init__(self, pose, sizes=np.float64([0, 0, 0]), num_points=0, occlusion=0, truncation=0.0):
        """Initialize a 3D bounding box with 3D pose and size (W, L, H).

        Parameters
        ----------
        pose: Pose, (default: Pose())
            Pose of the center of the 3D cuboid.

        sizes: np.float64, (default: np.float64([0,0,0]))
            Extents of the cuboid (width, length, height).

        num_points: int
            Number of LIDAR points associated with this bounding box.

        occlusion: int, default: 0
            Occlusion state (KITTI3D style)

        truncation: float, default: 0
            Fraction of truncation of object (KITTI3D style)
        """
        assert isinstance(pose, Pose)
        assert isinstance(sizes, np.ndarray)
        assert len(sizes) == 3
        self.pose = pose
        self.sizes = sizes
        self.num_points = num_points
        self.occlusion = occlusion
        self.truncation = truncation

    def __repr__(self):
        return 'BBOX3D Pose: {}, (W: {}, L: {}, H: {}), Points: {}'.format(
            self.pose, self.sizes[0], self.sizes[1], self.sizes[2], self.num_points
        )

    @classmethod
    def from_torch(cls, tensor):
        """Convert a torch Tensor representation of a bounding box back to
        an instance of BoundingBox3D.

        Returns
        -------
        box_3d: BoundingBox3D
            Instance of BoundingBox3D from torch.FloatTensor
        """
        assert isinstance(tensor, torch.FloatTensor)
        pose = Pose(wxyz=tensor[0:4], tvec=tensor[4:7])
        sizes = tensor[7:10].numpy()
        return cls(pose, sizes)

    @property
    def numpy(self):
        """Get a np.ndarray of with 10 dimensions representing the 3D bounding box

        Returns
        -------
        box_3d: np.float32 array
            Box with coordinates (pose.quat.qw, pose.quat.qx, pose.quat.qy, pose.quat.qz,
            pose.tvec.x, pose.tvec.y, pose.tvec.z, width, length, height)
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
        W, L, H = self.sizes
        x_corners = L / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = W / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = H / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners)).T
        return self.pose * corners

    @property
    def edges(self):
        """Get the 12 edge links of the 3D bounding box, indexed by the corners
        defined by `self.corners``.

        Returns
        ----------
        edges: np.ndarray (12 x 2)
            Edge links of the 3D bounding box.
        """
        return np.int32([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3,
                                                                                                                  7]])

    def render_on_image(self, camera, img):
        """Render the bounding box on the image.

        Parameters
        ----------
        camera: Camera
            Camera used to render the bounding box.

        img: np.ndarray
            Image to render the bounding box onto. We assume the input image is in RGB format

        Returns
        ----------
        img: np.ndarray
            Rendered image.
        """
        points2d = camera.project(self.corners)
        corners = points2d.T
        if (self.corners[:, 2] <= 0).any():
            return img

        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        COLORS = [RED, GREEN, BLUE]

        # Draw the sides (first)
        for i in range(4):
            cv2.line(
                img, (int(corners.T[i][0]), int(corners.T[i][1])), (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                (155, 155, 155),
                thickness=2
            )
        # Draw front (in red) and back (blue) face.
        cv2.polylines(img, [corners.T[:4].astype(np.int32)], True, RED, thickness=2)
        cv2.polylines(img, [corners.T[4:].astype(np.int32)], True, BLUE, thickness=2)

        # Draw axes on centroid
        vx = self.pose.tvec + self.pose.matrix[:3, 0]
        vy = self.pose.tvec + self.pose.matrix[:3, 1]
        vz = self.pose.tvec + self.pose.matrix[:3, 2]
        p1 = camera.project(np.vstack([self.pose.tvec, self.pose.tvec, self.pose.tvec])).astype(np.int32)
        p2 = camera.project(np.vstack([vx, vy, vz])).astype(np.int32)

        for i in range(3):
            cv2.line(img, (p1[i][0], p1[i][1]), (p2[i][0], p2[i][1]), COLORS[i], 2)

        return img


class Pose:
    """SE(3) rigid transform class that allows compounding of 6-DOF poses
    and provides common transformations that are commonly seen in geometric problems.
    """
    def __init__(self, wxyz=np.float64([1, 0, 0, 0]), tvec=np.float64([0, 0, 0])):
        """Initialize a Pose with Quaternion and 3D Position

        Parameters
        ----------
        wxyz: np.float64 or Quaternion or torch.FloatTensor, (default: np.float64([1,0,0,0]))
            Quaternion/Rotation (wxyz)

        tvec: np.float64 or torch.FloatTensor, (default: np.float64([0,0,0]))
            Translation (xyz)
        """
        if isinstance(wxyz, torch.FloatTensor):
            wxyz = wxyz.numpy()
        if isinstance(tvec, torch.FloatTensor):
            tvec = tvec.numpy()

        assert isinstance(wxyz, (np.ndarray, Quaternion))
        assert isinstance(tvec, np.ndarray)

        self.quat = Quaternion(wxyz)
        self.tvec = tvec

    def __repr__(self):
        formatter = {'float_kind': lambda x: '%.2f' % x}
        tvec_str = np.array2string(self.tvec, formatter=formatter)
        return 'wxyz: {}, tvec: ({})'.format(self.quat, tvec_str)

    def copy(self):
        """Return a copy of this pose object.

        Returns
        ----------
        result: Pose
            Copied pose object.
        """
        return self.__class__(Quaternion(self.quat), self.tvec.copy())

    def __mul__(self, other):
        """Left-multiply Pose with another Pose or 3D-Points.

        Parameters
        ----------
        other: Pose or np.ndarray
            1. Pose: Identical to oplus operation.
               (i.e. self_pose * other_pose)
            2. ndarray: transform [N x 3] point set
               (i.e. X' = self_pose * X)

        Returns
        ----------
        result: Pose or np.ndarray
            Transformed pose or point cloud
        """
        if isinstance(other, Pose):
            assert isinstance(other, self.__class__)
            t = self.quat.rotate(other.tvec) + self.tvec
            q = self.quat * other.quat
            return self.__class__(q, t)
        elif isinstance(other, BoundingBox3D):
            return BoundingBox3D(self * other.pose, other.sizes)
        else:
            assert other.shape[-1] == 3, 'Point cloud is not 3-dimensional'
            X = np.hstack([other, np.ones((len(other), 1))]).T
            return (np.dot(self.matrix, X).T)[:, :3]

    def __rmul__(self, other):
        raise NotImplementedError('Right multiply not implemented yet!')

    def inverse(self):
        """Returns a new Pose that corresponds to the
        inverse of this one.

        Returns
        ----------
        result: Pose
            Inverted pose
        """
        qinv = self.quat.inverse
        return self.__class__(qinv, qinv.rotate(-self.tvec))

    @property
    def matrix(self):
        """Returns a 4x4 homogeneous matrix of the form [R t; 0 1]

        Returns
        ----------
        result: np.ndarray
            4x4 homogeneous matrix
        """
        result = self.quat.transformation_matrix
        result[:3, 3] = self.tvec
        return result

    @property
    def rotation_matrix(self):
        """Returns the 3x3 rotation matrix (R)

        Returns
        ----------
        result: np.ndarray
            3x3 rotation matrix
        """
        result = self.quat.transformation_matrix
        return result[:3, :3]

    @property
    def rotation(self):
        """Return the rotation component of the pose as a Quaternion object.

        Returns
        ----------
        self.quat: Quaternion
            Rotation component of the Pose object.
        """
        return self.quat

    @property
    def translation(self):
        """Return the translation component of the pose as a np.ndarray.

        Returns
        ----------
        self.tvec: np.ndarray
            Translation component of the Pose object.
        """
        return self.tvec

    @classmethod
    def from_matrix(cls, transformation_matrix):
        """Initialize pose from 4x4 transformation matrix

        Parameters
        ----------
        transformation_matrix: np.ndarray
            4x4 containing rotation/translation

        Returns
        -------
        Pose
        """
        return cls(wxyz=Quaternion(matrix=transformation_matrix[:3, :3]), tvec=np.float64(transformation_matrix[:3, 3]))

    @classmethod
    def from_pose_proto(cls, pose_proto):
        """Initialize pose from 4x4 transformation matrix

        Parameters
        ----------
        pose_proto: Pose_pb2
            Pose as defined in proto/geometry.proto

        Returns
        -------
        Pose
        """

        rotation = np.float64([
            pose_proto.rotation.qw,
            pose_proto.rotation.qx,
            pose_proto.rotation.qy,
            pose_proto.rotation.qz,
        ])

        translation = np.float64([
            pose_proto.translation.x,
            pose_proto.translation.y,
            pose_proto.translation.z,
        ])
        return cls(wxyz=rotation, tvec=translation)

    def to_proto(self):
        """Convert Pose into pb object.

        Returns
        -------
        pose_0S: Pose_pb2
            Pose as defined in proto/geometry.proto
        """
        pose_0S = geometry_pb2.Pose()
        pose_0S.rotation.qw = self.quat.elements[0]
        pose_0S.rotation.qx = self.quat.elements[1]
        pose_0S.rotation.qy = self.quat.elements[2]
        pose_0S.rotation.qz = self.quat.elements[3]

        pose_0S.translation.x = self.tvec[0]
        pose_0S.translation.y = self.tvec[1]
        pose_0S.translation.z = self.tvec[2]
        return pose_0S

    def __eq__(self, other):
        return self.quat == other.quat and (self.tvec == other.tvec).all()
