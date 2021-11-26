import numpy as np
from numpy.lib.ufunclike import fix
from dgp.utils.camera import Camera
def test_camera_creation():
    k= np.ones(shape=(3,3))
    d = np.ones(shape=(5,))
    camera = Camera(K=k,D=d)
    camera2 = Camera.from_params(1,1,1,1)
    assert camera.fx == camera2.fx
    assert camera.fy == camera2.fy
    assert camera.cx == camera2.cx
    assert camera.cy == camera2.cy
    assert camera.P.any()
    assert camera.Kinv.any()