# Copyright 2021-2022 Toyota Research Institute.  All rights reserved.
"""General-purpose class for cameras."""
import logging
from functools import lru_cache

import cv2
import numpy as np

from dgp.proto import geometry_pb2
from dgp.utils.pose import Pose


def generate_depth_map(camera, Xw, shape):
    """Render pointcloud on image.

    Parameters
    ----------
    camera: Camera
        Camera object with appropriately set extrinsics wrt world.

    Xw: np.ndarray
        3D point cloud (x, y, z) in the world coordinate. Shape is (N x 3).

    shape: np.ndarray
        Output depth image shape as (H, W).

    Returns
    -------
    depth: np.array
        Rendered depth image.
    """
    assert len(shape) == 2, 'Shape needs to be 2-tuple.'
    # Move point cloud to the camera's (C) reference frame from the world (W)
    Xc = camera.p_cw * Xw
    # Project the points as if they were in the camera's frame of reference
    uv = Camera(K=camera.K).project(Xc).astype(int)
    # Colorize the point cloud based on depth
    z_c = Xc[:, 2]

    # Create an empty image to overlay
    H, W = shape
    depth = np.zeros((H, W), dtype=np.float32)
    in_view = np.logical_and.reduce([(uv >= 0).all(axis=1), uv[:, 0] < W, uv[:, 1] < H, z_c > 0])
    uv, z_c = uv[in_view], z_c[in_view]
    depth[uv[:, 1], uv[:, 0]] = z_c
    return depth


def pbobject_from_camera_matrix(K, distortion=None):
    """Convert camera intrinsic matrix into pb object.

    Parameters
    ----------
    K: np.ndarray
        Camera Intrinsic Matrix

    distortion: dict[str, float]
        Dictionary of distortion params i.e, k1,k2,p1,p2,k3,k4,xi,alpha etc

    Returns
    -------
    intrinsics: geometry_pb2.CameraIntrinsics
        Camera Intrinsic object
    """
    intrinsics = geometry_pb2.CameraIntrinsics()
    if len(K):
        intrinsics.fx = K[0, 0]
        intrinsics.fy = K[1, 1]
        intrinsics.cx = K[0, 2]
        intrinsics.cy = K[1, 2]
        intrinsics.skew = K[0, 1]

    if distortion is not None:
        for k, v in distortion.items():
            # TODO: assert the proto contains this value
            setattr(intrinsics, k, v)

    return intrinsics


def camera_matrix_from_pbobject(intrinsics):
    """Convert CameraIntrinsics pbobject to 3x3 camera matrix

    Parameters
    ----------
    intrinsics: CameraIntrinsics_pb2
        Protobuf containing cx, cy, fx, fy, skew

    Returns
    -------
    K: np.ndarray
        Camera matrix
    """
    K = np.eye(3)
    K[0, 0] = intrinsics.fx
    K[1, 1] = intrinsics.fy
    K[0, 2] = intrinsics.cx
    K[1, 2] = intrinsics.cy
    K[0, 1] = intrinsics.skew
    return K


class Distortion:
    """Distortion via distortion parameters or full distortion map"""
    def __init__(self, D=np.zeros(5, np.float32)):
        assert isinstance(D, np.ndarray)
        self.D = D

    def distorion_map(self):
        raise NotImplementedError()

    @property
    def coefficients(self):
        return self.D


class Camera:
    """Camera class with well-defined intrinsics and extrinsics."""
    def __init__(self, K, D=None, p_cw=None):
        """Initialize camera with identity pose.

        Parameters
        ----------
        K: np.ndarray (3x3)
            Camera calibration matrix.

        D: np.ndarray (5,) or (H x W) or Dict[str,float]
            Distortion parameters or distortion map or dictionary of distortion values

        p_cw: dgp.utils.pose.Pose
            Pose from world to camera frame.
        """
        self.K = K
        # TODO: refactor this class to support other camera models. This assumes Brown Conrady
        self.D = Distortion() if D is None else D
        self.p_cw = Pose() if p_cw is None else p_cw
        assert isinstance(self.K, np.ndarray)
        assert isinstance(self.p_cw, Pose)

    @property
    def rodrigues(self):
        """Retrieve the Rodrigues rotation.

        Returns
        -------
        rvec: np.ndarray (3,)
            Rodrigues rotation
        """
        R = self.p_cw.rotation_matrix
        rvec, _ = cv2.Rodrigues(R)
        return rvec

    @classmethod
    def from_params(cls, fx, fy, cx, cy, p_cw=None, distortion=None):
        """Create camera batch from calibration parameters.

        Parameters
        ----------
        fx: float
            Camera focal length along x-axis.

        fy: float
            Camera focal length along y-axis.

        cx: float
            Camera x-axis principal point.

        cy: float
            Camera y-axis principal point.

        p_cw: Pose
            Pose from world to camera frame.

        distortion: dict[str, float], optional
            Optional dictionary of distortion parameters k1,k2,.. etc. Default: None.

        Returns
        -------
        Camera
            Camera object with relevant intrinsics.
        """
        # TODO: add skew
        K = np.float32([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])
        return cls(K=K, D=distortion, p_cw=p_cw)

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    @property
    def P(self):
        """Projection matrix"""
        Rt = self.p_cw.matrix[:3]
        return self.K.dot(Rt)

    @property
    @lru_cache()
    def Kinv(self):
        """Analytic inverse camera intrinsic (K^-1)

        Returns
        ----------
        Kinv: np.ndarray (33)
            Inverse camera matrix K^-1.
        """
        Kinv = self.K.copy()
        Kinv[0, 0] = 1. / self.fx
        Kinv[1, 1] = 1. / self.fy
        Kinv[0, 2] = -1. * self.cx / self.fx
        Kinv[1, 2] = -1. * self.cy / self.fy
        return Kinv

    def transform(self, X, frame):
        # Transform 3D points into the camera reference frame
        if frame == 'c':
            Xc = X
        elif frame == 'w':
            Xc = self.p_cw * X
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))
        return Xc

    def project(self, Xw):
        """Project 3D points from specified reference frame onto image plane

        Parameters
        ----------
        Xw: np.ndarray
            3D spatial coordinates for each pixel in the world reference frame. Shape is (N x 3).

        Returns
        -------
        x: np.ndarray (N x 2)
            2D image coordinates for each pixel in the specified
           reference frame.
        """
        _, C = Xw.shape
        assert C == 3

        # Since self.D can be a distoriton object or a dictionary, handle the appropriate case and
        # throw a warning about the model being used. This currenty does not support fisheye.
        distortion = np.zeros(5, dtype=np.float32)
        if isinstance(self.D, Distortion):
            distortion = self.D.coefficients
        elif isinstance(self.D, dict):
            logging.warning('Using Brown-Conrady (Opencv default) distortion model for projection.')
            k1 = self.D.get('k1', 0.0)
            k2 = self.D.get('k2', 0.0)
            p1 = self.D.get('p1', 0.0)
            p2 = self.D.get('p2', 0.0)
            k3 = self.D.get('k3', 0.0)
            distortion = np.array([k1, k2, p1, p2, k3])

        uv, _ = cv2.projectPoints(Xw, self.rodrigues, self.p_cw.tvec, self.K, distortion)
        return uv.reshape(-1, 2)

    @staticmethod
    def scale_intrinsics(K, x_scale, y_scale):
        """Scale intrinsic matrix (B33, or 33) given x and y-axes scales.
        Note: This function works for both torch and numpy.

        Parameters
        ----------
        K: np.ndarray
            Camera calibration matrix. Shape is (3 x 3).

        x_scale: float
            x-axis scale factor.

        y_scale: float
            y-axis scale factor.

        Returns
        -------
        np.array of shape (3x3)
            Scaled camera intrinsic matrix
        """
        K[..., 0, 0] *= x_scale
        K[..., 1, 1] *= y_scale
        K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
        K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
        return K

    def unproject(self, points_in_2d):
        """Project points from 2D into 3D given known camera intrinsics

        Parameters
        ----------
        points_in_2d: np.ndarray
            Array of shape (N, 2)

        Returns
        -------
        points_in_3d: np.ndarray
            Array of shape (N, 3) of points projected into 3D
        """
        logging.warning('unproject currently does not consider distortion parameters')
        rays = cv2.undistortPoints(points_in_2d[:, None], self.K, None)
        return cv2.convertPointsToHomogeneous(rays).reshape(-1, 3).astype(np.float32)

    def in_frustum(self, X, height, width):
        """Given a set of 3D points, return a boolean index of points in the camera frustum

        Parameters
        ----------
        X: np.ndarray
            3D spatial coordinates for each pixel IN THE CAMERA FRAME. Shape is (N x 3).

        height: int
            Height of image.

        width: int
            Width of image.

        Returns
        -------
        inside_frustum: np.ndarray
            Bool array for X which are inside frustum
        """
        if not len(X):
            return np.array([], dtype=bool)

        corners = np.asarray([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)], dtype=np.float32)
        rays = self.unproject(corners)
        rays /= np.linalg.norm(rays, axis=1)[:, None]

        # Build the 4 plane normals that all point towards into frustum interior
        top = np.cross(rays[0], rays[1])
        right = np.cross(rays[1], rays[2])
        bottom = np.cross(rays[2], rays[3])
        left = np.cross(rays[3], rays[0])
        frustum = np.stack((top, right, bottom, left))

        inside_frustum = np.logical_and.reduce(frustum @ X.T > 0, axis=0)
        return inside_frustum
