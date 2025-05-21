import numpy as np
import pytest
from pyquaternion import Quaternion

from dgp.utils.pose import Pose


@pytest.fixture
def step_angle() -> float:
    return 30.0


@pytest.fixture
def dummy_pose(step_angle: float) -> Pose:
    theta = np.radians(step_angle)
    q = Quaternion._from_axis_angle(axis=np.float32([0.0, 0.0, 1.0]), angle=theta)
    return Pose.from_rotation_translation(
        rotation_matrix=q.rotation_matrix,
        tvec=np.float32([1, 2, 3]),
    )


@pytest.fixture
def dummy_pose_inverse(dummy_pose: Pose) -> Pose:
    return dummy_pose.inverse()
