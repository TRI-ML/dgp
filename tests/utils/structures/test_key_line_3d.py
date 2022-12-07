import numpy as np
import pytest
from numpy.lib.ufunclike import fix

from dgp.utils.structures.key_line_3d import KeyLine3D


@pytest.fixture
def keyline():
    k = np.float32([[.5, 2, -1], [-4, 0, 3], [0, -1, 2], [.25, 1.25, -.25], [100, 1, 200]])
    return KeyLine3D(k)


def test_keyline_class_id(keyline):
    assert keyline.class_id == 1


def test_keyline_instance_id(keyline):
    assert keyline.instance_id == '6b144d77fb6c1f915f56027b4fe34f5e'


def test_keyline_color(keyline):
    assert keyline.color == (0, 0, 0)


def test_keyline_attributes(keyline):
    assert keyline.attributes == {}


def test_keyline_hexdigest(keyline):
    assert keyline.hexdigest == '6b144d77fb6c1f915f56027b4fe34f5e'


def test_keyline_to_proto(keyline):
    assert len(keyline.to_proto()) == 5
