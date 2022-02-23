import os

import numpy as np
from dgp.annotations.key_line_2d_annotation import KeyLine2DAnnotation
from tests import TEST_DATA_DIR


def test_create_key_line_annotation():
    key_line_array = np.ones(shape=(10, 10))
    annotation = KeyLine2DAnnotation(key_line_array)
    assert annotation.hexdigest[:8] == "f430921"



def test_key_line_save():
    key_line_array = np.ones(shape=(10, 10))
    annotation = KeyLine2DAnnotation(key_line_array)
    annotation.save(".")
    filepath = "./123478fc23409ba80973214bc.npz"
    assert os.path.exists(filepath)
    os.remove(filepath)
