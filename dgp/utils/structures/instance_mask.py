# Copyright 2021 Toyota Research Institute.  All rights reserved.
import hashlib
from collections import OrderedDict

import numpy as np
import pycocotools.mask as mask_util

GENERIC_OBJECT_CLASS = 1


class InstanceMask2D():
    """2D instance mask object.

    Parameters
    ----------
    mask: np.ndarray[bool, np.uint8, np.int64]
        2D boolean array describiing instance mask.

    class_id: int, default: GENERIC_OBJECT_CLASS
        Integer class ID (0 reserved for background).

    instance_id: int, default: None
        Unique instance ID for instance mask. If None provided, the ID is a hash of the instance mask,
        location and class.

    attributes: dict, default: None
        Dictionary of attributes associated with instance mask. If None provided,
        defaults to empty dict.
    """
    def __init__(self, mask, class_id=GENERIC_OBJECT_CLASS, instance_id=None, color=(0, 0, 0), attributes=None):
        assert mask.dtype in (bool, np.uint8, np.int64)
        self._bitmask = mask

        self._class_id = class_id
        self._instance_id = instance_id
        self._color = color
        self._attributes = dict(attributes) if attributes is not None else {}

    def intersection_over_union(self, other):
        """Compute intersection over union of this box against other(s).

        Parameters
        ----------
        other: InstanceMask2D
            Another instance of InstanceMask2D to compute IoU against.

        Raises
        ------
        NotImplementedError
            Unconditionally.
        """
        raise NotImplementedError

    @property
    def bitmask(self):
        return self._bitmask

    @property
    def class_id(self):
        return self._class_id

    @class_id.setter
    def class_id(self, class_id):
        self._class_id = class_id

    @property
    def color(self):
        return self._color

    @property
    def instance_id(self):
        if self._instance_id is None:
            return self.hexdigest
        return self._instance_id

    @property
    def area(self):
        """Compute intersection over union of this box against other(s)."""
        return np.sum(self.bitmask)

    @property
    def attributes(self):
        return self._attributes

    @property
    def hexdigest(self):
        return hashlib.md5(self.bitmask.tobytes() + bytes(self._class_id) + bytes(self._instance_id)).hexdigest()

    def __repr__(self):
        return "{}[Class: {}, Attributes: {}]".format(self.__class__.__name__, self.class_id, self.attributes)

    def __eq__(self, other):
        return self.hexdigest == other.hexdigest

    def render(self, image):
        """Render instance masks on an image.

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

    @property
    def rle(self):
        _rle = mask_util.encode(np.array(self.bitmask, np.uint8, order='F'))
        return RLEMask(_rle['size'], _rle['counts'])


class RLEMask():
    """Container of RLE-encoded mask.

    See https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py for RLE format.

    Parameters
    ----------
    size: list[int] or np.ndarray[int]
        Height and width of mask.
    counts: list[int]
        Count-encoding of RLE format.
    """
    def __init__(self, size, counts):
        self.size = size
        self.counts = counts

    def to_dict(self):
        return OrderedDict([('size', self.size), ('counts', self.counts)])
