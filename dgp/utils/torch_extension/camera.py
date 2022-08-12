# Copyright 2019 Toyota Research Institute.  All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgp.utils.torch_extension.pose import Pose, QuaternionPose
from dgp.utils.torch_extension.stn import image_grid


def construct_K(fx, fy, cx, cy, dtype=torch.float, device=None):
    """Create camera intrinsics from focal length and focal center

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
    dtype: str
        Tensor dtype
    device: str
        Tensor device

    Returns
    -------
    torch.FloatTensor
        Camera intrinsic matrix (33)
    """
    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=dtype, device=device)


def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics matrix given x and y-axes scales.
    Note: This function works for both torch and numpy.

    Parameters
    ----------
    K: torch.FloatTensor, np.ndarray
        An intrinsics matrix to scale. Shape is B33 or 33.
    x_scale: float
        x-axis scale factor.
    y_scale: float
        y-axis scale factor.

    Returns
    -------
    torch.FloatTensor, np.ndarray
        Scaled camera intrinsic matrix. Shape is B33 or 33.
    """
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K


class Camera(nn.Module):
    """Fully-differentiable camera class whose extrinsics operate on the
    appropriate pose manifold. Supports fully-differentiable 3D-to-2D, 2D-to-3D
    projection/back-projections, scaled camera and inverse warp
    functionality.

    Note: This class implements the batched camera class, where a batch of
    camera intrinsics (K) and extrinsics (Tcw) are used for camera projection,
    back-projection.

    Attributes
    ----------
    K: torch.FloatTensor (B33)
        Camera calibration matrix.
    p_cw: dgp.utils.torch_extension.pose.Pose or dgp.utils.torch_extension.pose.QuaternionPose
        Pose from world to camera frame.
    """
    def __init__(self, K, D=None, p_cw=None):
        super().__init__()
        self.K = K
        if D is not None:
            raise NotImplementedError('No support for camera distortion')
        self.D = None
        self.p_cw = Pose.identity(len(K)) \
                   if p_cw is None else p_cw
        assert len(self.p_cw) == len(self.K)
        assert isinstance(self.K, torch.Tensor)
        assert issubclass(type(self.p_cw), (Pose, QuaternionPose)), 'p_cw needs to be a Pose type'

    def to(self, *args, **kwargs):
        """Move camera object to specified device.

        Parameters
        ----------
        *args: tuple
            Positional arguments to a to() call to move a tensor-like object to a particular device.
        **kwargs: dict
            Keyword arguments to a to() call to move a tensor-like object to a particular device.
        """
        self.K = self.K.to(*args, **kwargs)
        self.p_cw = self.p_cw.to(*args, **kwargs)
        return self

    @classmethod
    def from_params(cls, fx, fy, cx, cy, p_cw=None, B=1):
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
            Pose from world to camera frame, with a batch size of 1.
        B: int
            Batch size for p_cw and K

        Returns
        ----------
        Camera
            Camera object with relevant intrinsics and batch size of B.
        """
        if p_cw is not None:
            assert issubclass(type(p_cw), (Pose, QuaternionPose)), 'p_cw needs to be a Pose type'
            assert len(p_cw) == 1
            p_cw = p_cw.repeat([B, 1, 1])
        return cls(K=construct_K(fx, fy, cx, cy).repeat([B, 1, 1]), p_cw=p_cw)

    def scaled(self, x_scale, y_scale=None):
        """Scale the camera by specified factor.

        Parameters
        ----------
        x_scale: float
            x-axis scale factor.
        y_scale: float
            y-axis scale factor.

        Returns
        ----------
        Camera
            Scaled camera object.
        """
        if y_scale is None:
            y_scale = x_scale
        if x_scale == 1. and y_scale == 1.:
            return self
        K = scale_intrinsics(self.K.clone(), x_scale, y_scale)
        return Camera(K, p_cw=self.p_cw)

    @property
    def fx(self):
        return self.K[:, 0, 0]

    @property
    def fy(self):
        return self.K[:, 1, 1]

    @property
    def cx(self):
        return self.K[:, 0, 2]

    @property
    def cy(self):
        return self.K[:, 1, 2]

    @property
    def P(self):
        """Projection matrix.

        Returns
        ----------
        P: torch.Tensor (B34)
            Projection matrix
        """
        Rt = self.p_cw.matrix[:, :3]
        return self.K.bmm(Rt)

    @property
    def Kinv(self):
        """Analytic inverse camera intrinsic (K^-1)

        Returns
        ----------
        Kinv: torch.FloatTensor (B33)
            Batch of inverse camera matrices K^-1.
        """
        assert tuple(self.K.shape[-2:]) == (3, 3)
        Kinv = self.K.clone()
        Kinv[:, 0, 0] = 1. / self.fx
        Kinv[:, 1, 1] = 1. / self.fy
        Kinv[:, 0, 2] = -1. * self.cx / self.fx
        Kinv[:, 1, 2] = -1. * self.cy / self.fy
        return Kinv

    def transform(self, X, frame):
        """Transform 3D points into the camera reference frame.

        Parameters
        ----------
        X: torch.FloatTensor
            Points reference in the `frame` reference frame. Shape must be B3*.

        frame: str
            Reference frame in which the output 3-D points are specified.
            Options are 'c' and 'w' that correspond to camera and world
            reference frames.

        Returns
        -------
        Xc: torch.FloatTensor (B3*)
            Transformed 3D points into the camera reference frame.

        Raises
        ------
        ValueError
            Raised if frame is an unsupported reference frame.
        """
        if frame == 'c':
            Xc = X
        elif frame == 'w':
            Xc = self.p_cw * X
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))
        return Xc

    def reconstruct(self, depth, frame='c'):
        """Back-project to 3D in specified reference frame, given depth map

        Parameters
        ----------
        depth: torch.FloatTensor
            Depth image. Shape must be B1HW.

        frame: str, optional
            Reference frame in which the output 3-D points are specified. Default: 'c'.

        Returns
        -------
        X: torch.FloatTensor
            Batch of 3D spatial coordinates for each pixel in the specified
            reference frame. Shape will be B3HW.
        """
        B, C, H, W = depth.shape
        assert C == 1

        # Generate image grid (B3HW) for camera projection
        grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)
        flat_grid = grid.view(B, 3, -1)  # B3(HW)

        # Estimate the outward rays in the camera frame
        # B3(HW) = B33 * B3(HW) -> B3HW
        xnorm = (self.Kinv.bmm(flat_grid)).view(B, 3, H, W)

        # Scale rays to metric depth and transform into the specified frame of
        # reference.
        Xr = xnorm * depth
        return self.transform(Xr, frame=frame)

    def project(self, X, frame='w', shape=None):
        """Project 3D points from specified reference frame onto image plane
        TODO: Support sparse point cloud projection.

        Parameters
        ----------
        X: torch.FloatTensor
            Batch of 3D spatial coordinates for each pixel in the specified
            reference frame. Shape must be B3HW or B3N.

        frame: str, optional
            Reference frame in which the input points (X) are specified. Default: 'w'.

        shape: tuple, optional
            Optional image shape. Default: None.

        Returns
        -------
        x: torch.FloatTensor (B2HW or B2N)
            Batch of normalized 2D image coordinates for each pixel in the specified
            reference frame. Normalized points range from (-1,1).

        Raises
        ------
        ValueError
            Raised if the shape of X is unsupported or if the frame type is unknown.
        """
        # If X.dim == 3, i.e. X shape is B3N, then we expect the image shape to be provided.
        # normalize and assume that X can be either B3HW or B3N
        if X.dim() == 3:
            assert shape is not None, 'Image shape needs to be provided to project X'
            H, W = shape
            B, C, N = X.shape
            output_shape = (N, )
        elif X.dim() == 4:
            B, C, H, W = X.shape
            output_shape = (H, W)
        else:
            raise ValueError('Unknown shape for input 3d points: {}, expected 3 or 4 dims'.format(X.shape))
        assert C == 3

        # If only a single camera intrinsic is provided, then all points are
        # projected with the same K, otherwise, use specific K for each point
        # cloud.
        K = self.K.repeat([B, 1, 1]) if len(self.K) == 1 else self.K
        assert len(K) == len(X)

        # Transform 3D points into the camera reference frame
        if frame == 'c':
            Xc = K.bmm(X.view(B, 3, -1))
        elif frame == 'w':
            Xc = K.bmm((self.p_cw * X).view(B, 3, -1))
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))
        X = Xc[:, 0]
        Y = Xc[:, 1]
        Z = Xc[:, 2].clamp(min=1e-5)

        # B(HW)
        Xnorm = 2 * (X / Z) / (W - 1) - 1.
        Ynorm = 2 * (Y / Z) / (H - 1) - 1.

        # TODO: Avoid out-of-bound pixels
        return torch.stack([Xnorm, Ynorm], dim=1).view(B, 2, *output_shape)

    def unproject(self, uv):
        """Project points from 2D into 3D given known camera intrinsics

        Parameters
        ----------
        uv: torch.FloatTensor
            Input un-normalized points in 2D (B, 2, N)

        Returns
        -------
        rays: torch.FloatTensor (B, 3, N)
            Rays projected out into 3D
        """
        B, C, N = uv.shape
        assert C == 2
        ones = torch.ones((B, 1, N), device=uv.device, dtype=uv.dtype)
        flat_grid = torch.cat([uv, ones], dim=1).view(B, 3, -1)

        # Estimate the outward rays in the camera frame
        rays = (self.Kinv.bmm(flat_grid)).view(B, 3, N)
        return rays
