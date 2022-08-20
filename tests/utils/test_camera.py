import numpy as np
import pytest
from numpy.lib.ufunclike import fix

from dgp.utils.camera import Camera


@pytest.fixture
def camera():
    k = np.float32([
        [.5, 0, 2],
        [0, .25, 3],
        [0, 0, 1],
    ])
    d = np.zeros(shape=(5, ))
    return Camera(K=k, D=d)


@pytest.fixture
def camera2():
    return Camera.from_params(.5, .25, 2, 3)


def test_camera_fx(camera, camera2):
    assert camera.fx == camera2.fx


def test_camera_fy(camera, camera2):
    assert camera.fy == camera2.fy


def test_camera_cx(camera, camera2):
    assert camera.cx == camera2.cx


def test_camera_cy(camera, camera2):
    assert camera.cy == camera2.cy


def test_camera_Kinv(camera):
    assert np.allclose(np.matmul(camera.Kinv, camera.K), np.eye(3))
