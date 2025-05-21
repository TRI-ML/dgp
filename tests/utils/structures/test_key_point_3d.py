import numpy as np
import pytest

from dgp.utils.math import Covariance3D
from dgp.utils.pose import Pose
from dgp.utils.structures.key_point_3d import ProbabilisticKeyPoint3D
from tests.pose_fixtures import dummy_pose, step_angle  # noqa: F401


@pytest.fixture
def cov_data() -> np.ndarray:
    return np.array([10, 2, 3, 10, 2, 10], dtype=np.float32)


@pytest.fixture()
def key_point(cov_data: np.ndarray) -> ProbabilisticKeyPoint3D:
    return ProbabilisticKeyPoint3D(
        point=np.asarray([1, 2, 3], dtype=np.float32),
        covariance=Covariance3D(cov_data),
    )


def test_cov3(cov_data):
    cov3 = Covariance3D(cov_data)
    np.testing.assert_allclose(cov3._get_array(cov3._get_mat(cov_data)), cov_data)


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


def test_probabilistic_key_point_3d(
    key_point: ProbabilisticKeyPoint3D,
    step_angle: float,  # noqa: F811
    dummy_pose: Pose,  # noqa: F811
) -> None:
    np.testing.assert_equal(key_point.hexdigest, "be2da15917b1820742ef5b67b4e38d74")

    rotated_point1 = key_point
    rotated_point2 = key_point
    num_rotations = int(round(360.0 / step_angle))
    for _ in range(num_rotations):
        rotated_point1 = dummy_pose * rotated_point1
        rotated_point2 = rotated_point2 * dummy_pose
        np.testing.assert_almost_equal(rotated_point1.xyz, rotated_point2.xyz)
