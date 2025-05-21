import numpy as np
import pytest

from dgp.utils.math import Covariance3D
from dgp.utils.pose import Pose

from tests.pose_fixtures import (  # noqa: F401, isort: skip
    dummy_pose, dummy_pose_inverse, step_angle,
)


@pytest.fixture
def dummy_cov3d() -> Covariance3D:
    return Covariance3D(data=np.float32([
        10.0,
        0.0,
        0.0,
        5.0,
        0.0,
        2.0,
    ]))


def test_covariance_rotation(
    dummy_cov3d: Covariance3D,
    dummy_pose: Pose,  # noqa: F811
    dummy_pose_inverse: Pose,  # noqa: F811
    step_angle: float,  # noqa: F811
):
    curr_cov1 = dummy_cov3d
    curr_cov2 = dummy_cov3d
    # Adjusted due to numerical errors.
    rtol = 1e-6
    atol = 1e-6
    num_rotations = int(round(360.0 / step_angle))
    for _ in range(num_rotations):
        # Counter clockwise rotation.
        curr_cov1 = dummy_pose * curr_cov1
        curr_cov2 = curr_cov2 * dummy_pose_inverse
        # Clockwise rotation.
        np.testing.assert_allclose(curr_cov1.mat3, curr_cov2.mat3, rtol=rtol, atol=atol)
    np.testing.assert_allclose(curr_cov1.mat3, dummy_cov3d.mat3, rtol=rtol, atol=atol)
    np.testing.assert_allclose(curr_cov2.mat3, dummy_cov3d.mat3, rtol=rtol, atol=atol)
