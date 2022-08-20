import numpy as np
import pytest
from numpy.lib.ufunclike import fix

from dgp.utils.structures.key_line_2d import KeyLine2D


@pytest.fixture
def keyline():
    k = np.float32([
        [.5, 2],
        [0, 3],
        [0, 1],
        [.25, -.25],
        [100, 1],
    ])
    return KeyLine2D(k)


def test_keyline_class_id(keyline):
    assert keyline.class_id == 1


def test_keyline_instance_id(keyline):
    assert keyline.instance_id == 'fec66e3031932ead7efc8c0e5090ffac'


def test_keyline_color(keyline):
    assert keyline.color == (0, 0, 0)


def test_keyline_attributes(keyline):
    assert keyline.attributes == {}


def test_keyline_hexdigest(keyline):
    assert keyline.hexdigest == 'fec66e3031932ead7efc8c0e5090ffac'


def test_keyline_to_proto(keyline):
    assert len(keyline.to_proto()) == 5
