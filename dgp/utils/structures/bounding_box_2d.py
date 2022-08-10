# Copyright 2021 Toyota Research Institute.  All rights reserved.
import hashlib

import numpy as np

import dgp.proto.annotations_pb2 as annotations_pb2

GENERIC_OBJECT_CLASS = 1


class BoundingBox2D:
    """2D bounding box object.

    Parameters
    ----------
    box: np.ndarray[np.float32]
        Array of 4 floats describing bounding box coordinates. Can be either ([l, t, r, b] or [l, t, w, h]).

    class_id: int, default: GENERIC_OBJECT_CLASS
        Integer class ID (0 reserved for background).

    instance_id: int, default: None
        Unique instance ID for bounding box. If None provided, the ID is a hash of the bounding box
        location and class.

    color: tuple, default: (0, 0, 0)
        RGB tuple for bounding box color

    attributes: dict, default: None
        Dictionary of attributes associated with bounding box. If None provided,
        defaults to empty dict.

    mode: str, default: ltwh
        One of "ltwh" or "ltrb". Corresponds to "[left, top, width, height]" representation
        or "[left, top, right, bottom]"

    Raises
    ------
    Exception
        Raised if the value of mode is unsupported.
    """
    def __init__(
        self, box, class_id=GENERIC_OBJECT_CLASS, instance_id=None, color=(0, 0, 0), attributes=None, mode="ltwh"
    ):
        assert box.dtype in (np.float32, np.float64)
        assert box.shape[0] == 4
        assert class_id != 0, "0 is reserved for background, your class must have a different ID"

        self._box = box

        self.l = box[0]
        self.t = box[1]
        if mode == "ltwh":
            self.w = box[2]
            self.h = box[3]
        elif mode == "ltrb":
            self.w = box[2] - box[0]
            self.h = box[3] - box[1]
        else:
            raise Exception(f"Bounding box must be initialized as 'ltrb' or 'ltwh', cannot recognize {mode}")

        self._class_id = class_id
        self._instance_id = instance_id
        self._color = color
        self._attributes = dict(attributes) if attributes is not None else {}

    def intersection_over_union(self, other):
        """Compute intersection over union of this box against other(s).

        Parameters
        ----------
        other: BoundingBox2D
            A separate BoundingBox2D instance to compute IoU against.

        Raises
        ------
        NotImplementedError
            Unconditionally
        """
        raise NotImplementedError

    @property
    def ltrb(self):
        return np.array([self.l, self.t, self.l + self.w, self.t + self.h], dtype=np.float32)

    @property
    def ltwh(self):
        return np.array([self.l, self.t, self.w, self.h], dtype=np.float32)

    @property
    def class_id(self):
        return self._class_id

    @class_id.setter
    def class_id(self, class_id):
        self._class_id = class_id

    @property
    def instance_id(self):
        if self._instance_id is None:
            return self.hexdigest
        return self._instance_id

    @property
    def area(self):
        return self.w * self.h

    @property
    def color(self):
        return self._color

    @property
    def attributes(self):
        return self._attributes

    @property
    def hexdigest(self):
        return hashlib.md5(self.ltrb.tobytes() + bytes(self._class_id)).hexdigest()

    def __repr__(self):
        return "{}[{}, Class: {}, Attributes: {}]".format(
            self.__class__.__name__, list(self.ltrb), self.class_id, self.attributes
        )

    def __eq__(self, other):
        return self.hexdigest == other.hexdigest

    def render(self, image):
        """Render bounding boxes on an image.

        Parameters
        ----------
        image: PIL.Image or np.ndarray
            Background image on which boxes are rendered

        Returns
        -------
        image: PIL.Image or np.ndarray
            Image with boxes rendered
        """
        raise NotImplementedError

    def to_proto(self):
        """Serialize bounding box to proto object.

        NOTE: Does not serialize class or instance information, just box geometry.
        To serialize a complete annotation, see `dgp/annotations/bounding_box_2d_annotation.py`

        Returns
        -------
        BoundingBox2D.pb2
            As defined in `proto/annotations.proto`
        """
        return annotations_pb2.BoundingBox2D(x=int(self.l), y=int(self.t), w=int(self.w), h=int(self.h))
