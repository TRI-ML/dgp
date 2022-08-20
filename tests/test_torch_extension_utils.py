# Copyright 2019 Toyota Research Institute. All rights reserved.
import functools
import unittest

import numpy as np
import torch
from pyquaternion import Quaternion

from dgp.utils.pose import Pose as NumpyPose
from dgp.utils.testing import assert_raises, assert_true
from dgp.utils.torch_extension.camera import Camera, image_grid
from dgp.utils.torch_extension.pose import (
    Pose,
    QuaternionPose,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)


def make_random_quaternion():
    q_wxyz = np.float32(Quaternion.random().elements)
    return q_wxyz


class TestTorchUtilities(unittest.TestCase):
    def setUp(self):
        def make_random_pvec():
            q_wxyz = make_random_quaternion()
            tvec = np.random.rand(3).astype(np.float32)
            return np.hstack([q_wxyz, tvec])

        def pvec2mat(pvec):
            q_wxyz, tvec = torch.from_numpy(pvec[:4]), torch.from_numpy(pvec[4:])
            R = quaternion_to_rotation_matrix(q_wxyz)
            Rt = torch.cat([R, tvec.reshape(3, 1)], 1)
            T = torch.eye(4, device=Rt.device)
            T[:3] = Rt
            return T

        self.pvecs = [make_random_pvec() for _ in range(3)]
        self.tfs = [pvec2mat(pvec) for pvec in self.pvecs]

    def test_quaternion_rotation_conversions(self):
        q_wxyz = torch.from_numpy(make_random_quaternion())
        R = quaternion_to_rotation_matrix(q_wxyz)
        q_wxyz_ = rotation_matrix_to_quaternion(R)
        # Check if either q == q' or q == -q' (since q == -q represents the
        # same rotation)
        assert_true(
            np.allclose(q_wxyz.numpy(), q_wxyz_.numpy(), atol=1e-6)
            or np.allclose(q_wxyz.numpy(), -q_wxyz_.numpy(), atol=1e-6)
        )

    def test_pose_utils_equivalence(self):
        """Test pose transform equivalance with dgp.utils.geometry"""
        poses_np = [NumpyPose(wxyz=pvec[:4], tvec=pvec[4:]) for pvec in self.pvecs]
        poses = [Pose(tf) for tf in self.tfs]
        qposes = [QuaternionPose.from_matrix(tf) for tf in self.tfs]

        # Check if the pose construction and conversion to homogeneous matrices
        # are consistent across all implementations
        for p1, p2, p3 in zip(poses_np, poses, qposes):
            assert_true(np.allclose(p1.matrix, p2.matrix.numpy(), atol=1e-6))
            assert_true(np.allclose(p1.matrix, p3.matrix.numpy(), atol=1e-6))

    def test_pose_utils(self):
        """Test pose class in dgp.utils.torch_extension.Pose and dgp.utils.torch_extension.QuaternionPose"""

        # Test pose transforms
        npposes = [NumpyPose(wxyz=pvec[:4], tvec=pvec[4:]) for pvec in self.pvecs]
        poses = [Pose(tf) for tf in self.tfs]
        qposes = [QuaternionPose.from_matrix(tf) for tf in self.tfs]

        # Test matrix composition of transformations
        final_pose_np = functools.reduce(lambda x, y: x @ y, [tf.numpy() for tf in self.tfs])
        final_pose_torch = functools.reduce(lambda x, y: x @ y, self.tfs)
        assert_true(np.allclose(final_pose_np, final_pose_torch.numpy(), atol=1e-6))

        # Test Pose manifold composition of transformations
        final_pose_NumpyPose = functools.reduce(lambda x, y: x * y, npposes)
        final_pose_Pose = functools.reduce(lambda x, y: x * y, poses)
        final_pose_QuaternionPose = functools.reduce(lambda x, y: x * y, qposes)
        assert_true(np.allclose(final_pose_np, final_pose_NumpyPose.matrix, atol=1e-6))
        assert_true(np.allclose(final_pose_np, final_pose_Pose.matrix.numpy(), atol=1e-6))
        assert_true(np.allclose(final_pose_np, final_pose_QuaternionPose.matrix.numpy(), atol=1e-6))

        def make_random_points(B=1, N=100):
            return torch.from_numpy(np.random.rand(B, 3, N)).type(torch.float)

        # Test single point cloud transformations for some implementations
        X = make_random_points()
        Xt_ = X[0].numpy()
        X_ = Xt_.T

        # Test point cloud transformations
        X1 = final_pose_Pose * X
        X2 = final_pose_QuaternionPose * X
        X3 = final_pose_NumpyPose * X_
        X4 = final_pose_np.dot(np.vstack([Xt_, np.ones((1, len(X_)))]))

        assert_true(np.allclose(X1.numpy(), X2.numpy(), atol=1e-6))
        assert_true(np.allclose(X1.squeeze().numpy().T, X3, atol=1e-6))
        assert_true(np.allclose(X1.squeeze().numpy(), X4[:3, :], atol=1e-6))

    @unittest.skip("sparse_uv2d broken")
    def test_camera_utils(self):
        """Test camera class in dgp.utils.torch_extension.Camera"""
        fx = fy = 500.
        B, H, W = 10, 480, 640
        cx = W / 2 - 0.5
        cy = H / 2 - 0.5
        inv_depth = torch.rand((B, 1, H, W))

        # Create a camera at identity and reconstruct point cloud from depth
        cam = Camera.from_params(fx, fy, cx, cy, p_cw=None, B=B)
        X = cam.reconstruct(1. / (inv_depth + 1e-6))
        assert_true(tuple(X.shape) == (B, 3, H, W))

        # Project the point cloud back into the image
        uv_pred = cam.project(X)
        assert_true(tuple(uv_pred.shape) == (B, 2, H, W))

        # Image grid and the projection should be identical since we
        # reconstructed and projected without any rotation/translation.
        grid = image_grid(B, H, W, inv_depth.dtype, inv_depth.device, normalized=True)
        uv = grid[:, :2]
        assert_true(np.allclose(uv.numpy(), uv_pred.numpy(), atol=1e-6))

        # Backproject ray from the sampled 2d image points
        sparse_uv2d = image_grid(B, H, W, inv_depth.dtype, inv_depth.device, normalized=False)[:, :2, ::10, ::10]
        sparse_uv2d = sparse_uv2d.contiguous().view(B, 2, -1)

        # Unproject to 3d rays (x, y, 1): B3N
        sparse_rays = cam.unproject(sparse_uv2d)
        sparse_inv_depth = torch.rand((B, 1, sparse_uv2d.shape[-1]))
        sparse_X = sparse_rays * sparse_inv_depth.repeat([1, 3, 1])
        assert_true(tuple(sparse_X.shape) == (B, 3, sparse_uv2d.shape[-1]))

        # Check if cam.project() without input shape raises an error
        with assert_raises(AssertionError) as _:
            sparse_uv2d_pred = cam.project(sparse_rays)

        # Camera project provides uv in normalized coordinates
        sparse_uv2d_pred = cam.project(sparse_X, shape=(H, W))

        # Normalize uv2d
        sparse_uv2d_norm = sparse_uv2d.clone()
        sparse_uv2d_norm[:, 0] = 2 * sparse_uv2d[:, 0] / (W - 1) - 1.
        sparse_uv2d_norm[:, 1] = 2 * sparse_uv2d[:, 1] / (H - 1) - 1.
        assert_true(np.allclose(sparse_uv2d_norm.numpy(), sparse_uv2d_pred.numpy(), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
