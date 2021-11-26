import numpy as np

from dgp.annotations.depth_annotation import DenseDepthAnnotation


def test_create_depth_annotation():
    depth_array = np.ones(shape=(10,10))
    annotation = DenseDepthAnnotation(depth_array)
    annotation.render()
    assert annotation.hexdigest[:8] == "fe0e420a"