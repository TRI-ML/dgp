import numpy as np
import pytest

from dgp.utils.pose import Pose
from dgp.utils.structures.key_line_3d import KeyLine3D, ProbabilisticKeyLine3D
from dgp.utils.structures.key_point_3d import ProbabilisticKeyPoint3D

from tests.pose_fixtures import dummy_pose, dummy_pose_inverse, step_angle  # noqa: F401, isort: skip


@pytest.fixture
def key_line():
    k = np.float32([[0.5, 2, -1], [-4, 0, 3], [0, -1, 2], [0.25, 1.25, -0.25], [100, 1, 200]])
    return KeyLine3D(k)


def test_keyline_class_id(key_line):
    assert key_line.class_id == 1


def test_keyline_instance_id(key_line):
    assert key_line.instance_id == "6b144d77fb6c1f915f56027b4fe34f5e"


def test_keyline_color(key_line):
    assert key_line.color == (0, 0, 0)


def test_keyline_attributes(key_line):
    assert key_line.attributes == {}


def test_keyline_hexdigest(key_line):
    assert key_line.hexdigest == "6b144d77fb6c1f915f56027b4fe34f5e"


def test_keyline_to_proto(key_line):
    assert len(key_line.to_proto()) == 5


@pytest.fixture
def cov_data() -> np.ndarray:
    return np.array([10, 2, 3, 10, 2, 10], dtype=np.float32)


def test_key_line_3d(key_line: KeyLine3D, dummy_pose: Pose, dummy_pose_inverse: Pose):  # noqa: F811
    transformed_key_line = dummy_pose * key_line
    got_key_line = dummy_pose_inverse * transformed_key_line
    np.testing.assert_allclose(key_line.xyz, got_key_line.xyz, rtol=1e-6, atol=1e-6)


def test_probabilistic_key_line_3d(cov_data: np.ndarray, dummy_pose: Pose, dummy_pose_inverse: Pose):  # noqa: F811
    p1 = ProbabilisticKeyPoint3D(np.asarray([1, 1, 1], dtype=np.float32), covariance=cov_data)
    p2 = ProbabilisticKeyPoint3D(np.asarray([2, 2, 2], dtype=np.float32), covariance=cov_data)
    key_line_3d = ProbabilisticKeyLine3D(points=[p1, p2])
    np.testing.assert_equal(key_line_3d.cov3[0], p1.cov3)
    np.testing.assert_equal(key_line_3d.cov3[1], p2.cov3)
    # NOTE: xyz is in column major.
    np.testing.assert_equal(key_line_3d.xyz.T[0], p1.xyz)
    np.testing.assert_equal(key_line_3d.xyz.T[1], p2.xyz)
    np.testing.assert_equal(key_line_3d.hexdigest, "bd11fdd87cebea6bc362631de2288bee")

    transformed_key_line_3d = dummy_pose * key_line_3d
    got_key_line_3d = dummy_pose_inverse * transformed_key_line_3d

    for point1, point2 in zip(key_line_3d, got_key_line_3d):
        np.testing.assert_almost_equal(point1.xyz, point2.xyz)
        np.testing.assert_almost_equal(point1.cov3.arr6, point2.cov3.arr6)
