# Copyright 2021 Toyota Research Institute.  All rights reserved.
import hashlib

import numpy as np

import dgp.proto.annotations_pb2 as annotations_pb2

GENERIC_OBJECT_CLASS = 1


class KeyLine2D:
    """2D keyline object.

    Parameters
    ----------
    line: np.ndarray[np.float32]
        Array of N*2 floats describing keyline coordinates [x, y].

    class_id: int, default: GENERIC_OBJECT_CLASS
        Integer class ID

    instance_id: int, default: None
        Unique instance ID for keyline. If None provided, the ID is a hash of the keyline
        location and class.

    color: tuple, default: (0, 0, 0)
        RGB tuple for keyline color

    attributes: dict, default: None
        Dictionary of attributes associated with keyline. If None provided,
        defaults to empty dict.

    """
    def __init__(self, line, class_id=GENERIC_OBJECT_CLASS, instance_id=None, color=(0, 0, 0), attributes=None):
        assert line.dtype in (np.float32, np.float64)
        assert line.shape[1] == 2

        self._point = line
        self.x = []
        self.y = []
        for point in line:
            self.x.append(point[0])
            self.y.append(point[1])

        self._class_id = class_id
        self._instance_id = instance_id
        self._color = color
        self._attributes = dict(attributes) if attributes is not None else {}

    @property
    def xy(self):
        return np.array([self.x, self.y], dtype=np.float32)

    @property
    def class_id(self):
        return self._class_id

    @property
    def instance_id(self):
        if self._instance_id is None:
            return self.hexdigest
        return self._instance_id

    @property
    def color(self):
        return self._color

    @property
    def attributes(self):
        return self._attributes

    @property
    def hexdigest(self):
        return hashlib.md5(self.xy.tobytes() + bytes(self._class_id)).hexdigest()

    def __repr__(self):
        return "{}[{}, Class: {}, Attributes: {}]".format(
            self.__class__.__name__, list(self.xy), self.class_id, self.attributes
        )

    def __eq__(self, other):
        return self.hexdigest == other.hexdigest

    def to_proto(self):
        """Serialize keyline to proto object.

        NOTE: Does not serialize class or instance information, just line geometry.
        To serialize a complete annotation, see `dgp/annotations/key_line_2d_annotation.py`

        Returns
        -------
        KeyLine2D.pb2
            As defined in `proto/annotations.proto`
        """
        return [annotations_pb2.KeyPoint2D(x=int(self.x[j]), y=int(self.y[j])) for j, k in enumerate(self.x)]
