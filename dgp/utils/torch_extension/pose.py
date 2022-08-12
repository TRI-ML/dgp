# Copyright 2019 Toyota Research Institute.  All rights reserved.
"""Torch utilities for rigid-body pose manipulation.

Some of the rotation/quaternion utilities have been borrowed from:
https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
https://github.com/arraiyopensource/kornia/blob/master/kornia/geometry/conversions.py
"""
import torch
import torch.nn.functional as F


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).

    Parameters
    ----------
    q: torch.FloatTensor
        Input quaternion to use for rotation. Shape is B4.

    r: torch.FloatTensor
        Second quaternion to use for rotation composition. Shape is B4.

    Returns
    ----------
    rotated_q: torch.FloatTensor (B4)
        Composed quaternion rotation.
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).

    Parameters
    ----------
    q: torch.FloatTensor
        Input quaternion to use for rotation. Shape is B4.

    v: torch.FloatTensor
        Input vector to rotate with. Shape is B3.

    Returns
    -------
    vector: torch.FloatTensor (B3)
        Rotated vector.
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qinv(q):
    """Returns the quaternion conjugate. (w, x, y, z)

    Parameters
    ----------
    q: torch.FloatTensor
        Input quaternion to invert. Shape is B4.

    Returns
    -------
    quaternion: torch.FloatTensor (B4)
        Inverted quaternion.
    """
    q[..., -3:] *= -1
    return q


def quaternion_to_rotation_matrix(quaternion):
    """Converts a quaternion to a rotation matrix.
    The quaternion should be in (w, x, y, z) format.

    Parameters
    ----------
    quaternion: torch.FloatTensor
        Input quaternion to convert. Shape is B4.

    Returns
    ----------
    rotation_matrix: torch.FloatTensor (B33)
        Batched rotation matrix.

    Raises
    ------
    TypeError
        Raised if quaternion is not a torch.Tensor.
    ValueError
        Raised if the shape of quaternion is not supported.
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape))
    # normalize the input quaternion
    quaternion_norm = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    one = torch.tensor(1.)

    matrix = torch.stack([
        one - (tyy + tzz), txy - twz, txz + twy, txy + twz, one - (txx + tzz), tyz - twx, txz - twy, tyz + twx, one -
        (txx + tyy)
    ],
                         dim=-1).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-8):
    """Convert 3x3 rotation matrix to 4d quaternion vector.
    The quaternion vector has components in (w, x, y, z) format.

    Parameters
    ----------
    rotation_matrix: torch.FloatTensor
        Input rotation matrix to convert. Shape is B33.

    eps: float, optional
        Epsilon value to avoid zero division. Default: 1e-8.

    Returns
    ----------
    quaternion: torch.FloatTensor (B4)
        Batched rotation in quaternion.

    Raises
    ------
    TypeError
        Raised if rotation_matrix is not a torch.Tensor.
    ValueError
        Raised if the shape of rotation_matrix is unsupported.
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix)))

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError("Input size must be a (*, 3, 3) tensor. Got {}".format(rotation_matrix.shape))

    def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(numerator.dtype).tiny  # type: ignore
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec = rotation_matrix.view(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt(trace + 1.0) * 2.  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat([qw, qx, qy, qz], dim=-1)

    def cond_1():
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat([qw, qx, qy, qz], dim=-1)

    def cond_2():
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat([qw, qx, qy, qz], dim=-1)

    def cond_3():
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat([qw, qx, qy, qz], dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion = torch.where(trace > 0., trace_positive_cond(), where_1)
    return quaternion


def normalize_quaternion(quaternion, eps=1e-12):
    """Normalizes a quaternion.
    The quaternion should be in (w, x, y, z) format.

    Parameters
    ----------
    quaternion: torch.FloatTensor
        Input quaternion to normalize. Shape is B4.

    eps: float, optional
        Epsilon value to avoid zero division. Default: 1e-12.

    Returns
    ----------
    normalized_quaternion: torch.FloatTensor (B4)
        Normalized quaternion.

    Raises
    ------
    TypeError
        Raised if quaternion is not a torch.Tensor.
    ValueError
        Raised if the shape of quaternion is not supported.
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape))
    return F.normalize(quaternion, p=2, dim=-1, eps=eps)


def invert_pose(T01):
    """Invert homogeneous matrix as a rigid transformation
       T^-1 = [R^T | -R^T * t]

    Parameters
    ----------
    T01: torch.FloatTensor
        Input batch of transformation tensors. Shape is B44.

    Returns
    ----------
    T10: torch.FloatTensor (B44)
        Inverted batch of transformation tensors.
    """
    Tinv = torch.eye(4, device=T01.device, dtype=T01.dtype).repeat([len(T01), 1, 1])
    Tinv[:, :3, :3] = torch.transpose(T01[:, :3, :3], -2, -1)
    Tinv[:, :3, -1] = torch.bmm(-1. * Tinv[:, :3, :3], T01[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv


class Pose:
    """Generic rigid-body transformation class that operates on the
    appropriately defined manifold.

    Parameters
    ----------
    value: torch.FloatTensor (44, B44)
        Input transformation tensors either batched (B44) or as a single value (44).

    Attributes
    ----------
    value: torch.FloatTensor (B44)
        Input transformation tensor batched (B44)
    """
    def __init__(self, value):
        assert tuple(value.shape[-2:]) == (4, 4)
        # If (4,4) tensor is passed, convert to (1,4,4)
        if value.dim() == 2:
            value = value.unsqueeze(0)
        assert value.dim() == 3
        self.value = value

    def __repr__(self):
        return self.value.__repr__()

    def __len__(self):
        """Batch size of pose tensor"""
        return len(self.value)

    def copy(self):
        raise NotImplementedError()

    @classmethod
    def identity(cls, B=1, device=None, dtype=torch.float):
        """Batch of identity matrices.

        Parameters
        ----------
        B: int, optional
            Batch size. Default: 1.

        device: str, optional
            A device for a tensor-like object; ex: "cpu". Default: None.

        dtype: optional
            A data type for a tensor-like object. Default: torch.float.

        Returns
        ----------
        Pose
            Batch of identity transformation poses.
        """
        return cls(torch.eye(4, device=device, dtype=dtype).repeat([B, 1, 1]))

    @property
    def matrix(self):
        """Returns the batched homogeneous matrix as a tensor

        Returns
        ----------
        result: torch.FloatTensor (B44)
            Bx4x4 homogeneous matrix
        """
        return self.value

    @property
    def rotation_matrix(self):
        """Returns the 3x3 rotation matrix (R)

        Returns
        ----------
        result: torch.FloatTensor (B33)
            Bx3x3 rotation matrix
        """
        return self.value[..., :3, :3]

    @property
    def translation(self):
        """Return the translation component of the pose as a torch.Tensor.

        Returns
        ----------
        tvec: torch.FloatTensor (B3)
            Translation component of the Pose object.
        """
        return self.value[..., :3, -1]

    def repeat(self, *args, **kwargs):
        """Repeat the Pose tensor

        Parameters
        ----------
        *args: tuple
            Positional arguments to a repeat() call to repeat a tensor-like object along particular dimensions.
        **kwargs: dict
            Keyword arguments to a repeat() call to repeat a tensor-like object along particular dimension.
        """
        self.value = self.value.repeat(*args, **kwargs)
        return self

    def to(self, *args, **kwargs):
        """Move object to specified device

        Parameters
        ----------
        *args: tuple
            Positional arguments to a to() call to move a tensor-like object to a particular device.
        **kwargs: dict
            Keyword arguments to a to() call to move a tensor-like object to a particular device.
        """
        self.value = self.value.to(*args, **kwargs)
        return self

    def __mul__(self, other):
        """Matrix multiplication overloading for pose-pose and pose-point
        transformations.

        Parameters
        ----------
        other: Pose or torch.FloatTensor
            Either Pose, or 3-D points torch.FloatTensor (B3N or B3HW).

        Returns
        -------
        Pose
            Transformed pose, or 3-D points via rigid-transform on the manifold,
            with same type as other.

        Raises
        ------
        ValueError
            Raised if the shape of other is unsupported.
        NotImplementedError
            Raised if other is neither a Pose nor a torch.Tensor.
        """
        if isinstance(other, Pose):
            return self.transform_pose(other)
        # jscpd:ignore-start
        elif isinstance(other, torch.Tensor):
            if other.shape[1] == 3 and other.dim() > 2:
                assert other.dim() == 3 or other.dim() == 4
                return self.transform_points(other)
            else:
                raise ValueError('Unknown tensor dimensions {}'.format(other.shape))
        else:
            raise NotImplementedError()
        # jscpd:ignore-end

    def __rmul__(self, other):
        """
        Raises
        ------
        NotImplementedError
            Unconditionally.
        """
        raise NotImplementedError('Right multiply not implemented yet!')

    def transform_pose(self, other):
        """Left-multiply (oplus) rigid-body transformation.

        Parameters
        ----------
        other: Pose
            Pose to left-multiply with (self * other)

        Returns
        ----------
        Pose
            Transformed Pose via rigid-transform on the manifold.
        """
        assert tuple(other.value.shape[-2:]) == (4, 4)
        return Pose(self.value.bmm(other.value))

    def transform_points(self, X0):
        """Transform 3-D points from one frame to another via rigid-body transformation.

        Parameters
        ----------
        X0: torch.FloatTensor
            3-D points in torch.FloatTensor (shaped either B3N or B3HW).

        Returns
        ----------
        torch.FloatTensor (B3N or B3HW)
           Transformed 3-D points with the same shape as X0.
        """
        assert X0.shape[1] == 3
        B = len(X0)
        shape = X0.shape[2:]
        X1 = self.value[:, :3, :3].bmm(X0.view(B, 3, -1)) + self.value[:, :3, -1].unsqueeze(-1)
        return X1.view(B, 3, *shape)

    def inverse(self):
        """Invert homogeneous matrix as a rigid transformation.

        Returns
        ----------
        Pose
           Pose batch inverted on the appropriate manifold.
        """
        return Pose(invert_pose(self.value))


class QuaternionPose:
    """Derived Pose class that operates on the quaternion manifold instead.

    Parameters
    ----------
    wxyz: torch.FloatTensor (4, B4)
        Input quaternion tensors either batched (B4) or as a single value (4,).
    tvec: torch.FloatTensor (3, B3)
        Input translation tensors either batched (B3) or as a single value (3,).
    """
    def __init__(self, wxyz, tvec):
        assert wxyz.dim() == tvec.dim(), ('Quaternion and translation dimensions are different')
        assert len(wxyz) == len(tvec), ('Quaternion and translation batch sizes are different')
        # If (d) tensor is passed, convert to (B,d)
        if wxyz.dim() == 1:
            wxyz = wxyz.unsqueeze(0)
            tvec = tvec.unsqueeze(0)
        assert wxyz.dim() == 2
        self.quat = wxyz
        self.tvec = tvec

    def __len__(self):
        """Batch size of pose tensor"""
        return len(self.quat)

    def __repr__(self):
        return 'QuaternionPose: B={}, [qw, qx, qy, qz, x, y, z]'.format(
            len(self), torch.cat([self.quat, self.tvec], dim=-1)
        )

    @classmethod
    def identity(cls, B=1, device=None, dtype=torch.float):
        """Batch of identity matrices.

        Parameters
        ----------
        B: int, optional
            Batch size. Default: 1.

        device: str
            A device to send a tensor-like object to. Ex: "cpu".

        dtype: optional
            A data type for a tensor-like object. Default: torch.float.

        Returns
        ----------
        Pose
            Batch of identity transformation poses.
        """
        return cls(
            torch.tensor([1., 0., 0., 0., 0., 0., 0.], device=device, dtype=dtype).repeat([B, 1]),
            torch.tensor([0., 0., 0.], device=device, dtype=dtype).repeat([B, 1])
        )

    @classmethod
    def from_matrix(cls, value):
        """Create a batched QuaternionPose from a batched homogeneous matrix.

        Parameters
        ----------
        value: torch.FloatTensor
            Batched homogeneous matrix. Shape is B44.

        Returns
        ----------
        pose: QuaternionPosec
            QuaternionPose batch. Batch dimension is shape B.
        """
        if value.dim() == 2:
            value = value.unsqueeze(0)
        wxyz = rotation_matrix_to_quaternion(value[..., :3, :3].contiguous())
        tvec = value[..., :3, -1]
        return cls(wxyz, tvec)

    @property
    def matrix(self):
        """Returns the batched homogeneous matrix as a tensor

        Returns
        ----------
        result: torch.FloatTensor (B44)
            Bx4x4 homogeneous matrix
        """
        R = quaternion_to_rotation_matrix(self.quat)
        T = torch.eye(4, device=R.device, dtype=R.dtype).repeat([len(self), 1, 1])
        T[:, :3, :3] = R
        T[:, :3, -1] = self.tvec
        return T

    @property
    def rotation_matrix(self):
        """Returns the 3x3 rotation matrix (R)

        Returns
        ----------
        result: torch.FloatTensor (B33)
            Bx3x3 rotation matrix
        """
        return quaternion_to_rotation_matrix(self.quat)

    @property
    def translation(self):
        """Return the translation component of the pose as a torch.Tensor.

        Returns
        ----------
        tvec: torch.FloatTensor (B3)
            Translation component of the Pose object.
        """
        return self.tvec

    def repeat(self, B):
        """Repeat the QuaternionPose tensor

        Parameters
        ----------
        B: int
            The size of the batch dimension.
        """
        self.quat = self.quat.repeat([B, 1])
        self.tvec = self.tvec.repeat([B, 1])
        assert self.quat.dim() == self.tvec.dim() == 2, (
            'Attempting to repeat along the batch dimension failed, quat/tvec dims: {}/{}'.format(
                self.quat.dim(), self.tvec.dim()
            )
        )
        return self

    def to(self, *args, **kwargs):
        """Move object to specified device

        Parameters
        ----------
        *args: tuple
            Positional arguments to a to() call to move a tensor-like object to a particular device.
        **kwargs: dict
            Keyword arguments to a to() call to move a tensor-like object to a particular device.
        """
        self.quat = self.quat.to(*args, **kwargs)
        self.tvec = self.tvec.to(*args, **kwargs)
        return self

    def __mul__(self, other):
        """Matrix multiplication overloading for pose-pose and pose-point
        transformations.

        Parameters
        ----------
        other: QuaternionPose or torch.FloatTensor
            Either Pose, or 3-D points torch.FloatTensor (B3N or B3HW).

        Returns
        ----------
        Pose
            Transformed pose, or 3-D points via rigid-transform on the manifold,
            with same type as other.

        Raises
        ------
        ValueError
            Raised if other.shape is not supported.
        NotImplementedError
            Raised if other is neither a QuaternionPose or a torch.Tensor.
        """
        if isinstance(other, QuaternionPose):
            return self.transform_pose(other)
        # jscpd:ignore-start
        elif isinstance(other, torch.Tensor):
            if other.shape[1] == 3 and other.dim() > 2:
                assert other.dim() == 3 or other.dim() == 4
                return self.transform_points(other)
            else:
                raise ValueError('Unknown tensor dimensions {}'.format(other.shape))
        else:
            raise NotImplementedError()
        # jscpd:ignore-end

    def transform_pose(self, other):
        """Left-multiply (oplus) rigid-body transformation.


        Parameters
        ----------
        other: QuaternionPose
            Pose to left-multiply with (self * other)

        Returns
        ----------
        QuaternionPose
           Transformed Pose via rigid-transform on the manifold.
        """
        assert isinstance(other, QuaternionPose), ('Other pose is not QuaternionPose')
        tvec = qrot(self.quat, other.tvec) + self.tvec
        quat = qmul(self.quat, other.quat)
        return self.__class__(quat, tvec)

    def transform_points(self, X0):
        """Transform 3-D points from one frame to another via rigid-body transformation.

        Note: This function can be modified to do batched rotation operation
        with quaternions directly.

        Parameters
        ----------
        X0: torch.FloatTensor
            3-D points in torch.FloatTensor (shaped either B3N or B3HW).

        Returns
        ----------
        torch.FloatTensor (B3N or B3HW)
           Transformed 3-D points with the same shape as X0.
        """
        assert X0.shape[1] == 3 and len(X0) == len(self), (
            'Batch sizes do not match pose={}, X={}, '.format(self.__repr__(), X0.shape)
        )
        B, shape = len(X0), X0.shape[2:]
        R = quaternion_to_rotation_matrix(self.quat)
        X1 = R.bmm(X0.view(B, 3, -1)) + self.tvec.unsqueeze(-1)
        return X1.view(B, 3, *shape)

    def inverse(self):
        """Invert T=[trans, quaternion] as a rigid transformation.
        Returns:
           QuaternionPose: Pose batch inverted on the appropriate manifold.
        """
        Qinv = qinv(self.quat)
        Tinv = qrot(Qinv, -self.tvec)
        return QuaternionPose(Qinv, Tinv)
