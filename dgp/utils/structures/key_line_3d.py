# Copyright 2022 Woven Planet. All rights reserved.
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import dgp.proto.annotations_pb2 as annotations_pb2
from dgp.utils.math import Covariance3D
from dgp.utils.pose import Pose
from dgp.utils.structures.key_point_3d import ProbabilisticKeyPoint3D

GENERIC_OBJECT_CLASS_ID = 1


class KeyLine3D:
    """3D keyline object.

    Parameters
    ----------
    line: np.ndarray[np.float32]
        Array of (N,3) floats describing keyline coordinates [x, y, z].

    class_id: int, default: GENERIC_OBJECT_CLASS_ID
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
    def __init__(self, line, class_id=GENERIC_OBJECT_CLASS_ID, instance_id=None, color=(0, 0, 0), attributes=None):
        assert line.dtype in (np.float32, np.float64)
        assert line.shape[1] == 3

        self._point = line
        self.x = []
        self.y = []
        self.z = []
        for point in line:
            self.x.append(point[0])
            self.y.append(point[1])
            self.z.append(point[2])

        self._class_id = class_id
        self._instance_id = instance_id
        self._color = color
        self._attributes = dict(attributes) if attributes is not None else {}

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)

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
        return hashlib.md5(self.xyz.tobytes() + bytes(self._class_id)).hexdigest()

    def __repr__(self):
        return "{}[{}, Class: {}, Attributes: {}]".format(
            self.__class__.__name__, list(self.xyz), self.class_id, self.attributes
        )

    def __eq__(self, other):
        return self.hexdigest == other.hexdigest

    def to_proto(self):
        """Serialize keyline to proto object.

        Does not serialize class or instance information, just line geometry.
        To serialize a complete annotation, see `dgp/annotations/key_line_3d_annotation.py`

        Returns
        -------
        KeyLine3D.pb2
            As defined in `proto/annotations.proto`
        """
        return [
            annotations_pb2.KeyPoint3D(x=float(self.x[j]), y=float(self.y[j]), z=float(self.z[j]))
            for j, _ in enumerate(self.x)
        ]

    def __mul__(self, pose: Pose) -> "KeyLine3D":
        return KeyLine3D(
            line=pose * self.xyz.T,
            class_id=self.class_id,
            instance_id=self.instance_id,
            attributes=self.attributes,
        )

    def __rmul__(self, pose: Pose) -> "KeyLine3D":
        return self.__mul__(pose)


class ProbabilisticKeyLine3D:
    """3D probabilistic key line object

    Parameters
    ----------
    points: List[ProbabilisticKeyPoint3D]
        List of probabilistic 3D key points.

    class_id: int, default: GENERIC_OBJECT_CLASS_ID
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
    def __init__(
        self,
        points: List[ProbabilisticKeyPoint3D],
        class_id: int = GENERIC_OBJECT_CLASS_ID,
        instance_id: Optional[int] = None,
        color: Tuple[int, int, int] = (0, 0, 0),
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self._points = points
        self._class_id = class_id
        self._instance_id = instance_id
        self._color = color
        self._attributes = dict(attributes) if attributes is not None else {}

    def __eq__(self, other: "ProbabilisticKeyLine3D") -> bool:
        return self.hexdigest == other.hexdigest

    @property
    def points(self) -> List[ProbabilisticKeyPoint3D]:
        return self._points

    @property
    def xyz(self) -> np.ndarray:
        """Gets the xyz coordinates in column major"""
        return np.array([point.xyz for point in self._points], dtype=np.float32).T

    @property
    def cov3(self) -> List[Covariance3D]:
        return [point.cov3 for point in self._points]

    @property
    def __len__(self) -> int:
        return len(self._points)

    def __getitem__(self, index: int) -> ProbabilisticKeyPoint3D:
        return self._points[index]

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
    def attributes(self) -> Union[Dict[str, Any], None]:
        return self._attributes

    @property
    def hexdigest(self):
        cov_bytes = np.asarray([point.cov3.arr6 for point in self.points]).tobytes()
        return hashlib.md5(
            self.xyz.tobytes() + cov_bytes + bytes(self._class_id) + str(self.attributes).encode("utf8")
        ).hexdigest()

    def __mul__(self, pose: Pose) -> "ProbabilisticKeyLine3D":
        return ProbabilisticKeyLine3D(
            points=[pose * point for point in self.points],
            class_id=self.class_id,
            instance_id=self.instance_id,
            color=self.color,
            attributes=self.attributes,
        )

    def __rmul__(self, pose: Pose) -> "ProbabilisticKeyLine3D":
        return self.__mul__(pose)
