# Copyright 2022 Woven Planet. All rights reserved.
import hashlib
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

import dgp.proto.annotations_pb2 as annotations_pb2
from dgp.utils.math import Covariance3D
from dgp.utils.pose import Pose

GENERIC_OBJECT_CLASS_ID = 1


class KeyPoint3D:
    """3D keypoint object.

    Parameters
    ----------
    point: np.ndarray[np.float32]
        Array of 3 floats describing keypoint coordinates [x, y, z].

    class_id: int, default: GENERIC_OBJECT_CLASS_ID
        Integer class ID (0 reserved for background).

    instance_id: int, default: None
        Unique instance ID for keypoint. If None provided, the ID is a hash of the keypoint
        location and class.

    color: tuple, default: (0, 0, 0)
        RGB tuple for keypoint color

    attributes: dict, default: None
        Dictionary of attributes associated with keypoint. If None provided,
        defaults to empty dict.
    """
    def __init__(self, point, class_id=GENERIC_OBJECT_CLASS_ID, instance_id=None, color=(0, 0, 0), attributes=None):
        assert point.dtype in (np.float32, np.float64)
        assert point.shape[0] == 3

        self._point = point

        self.x = point[0]
        self.y = point[1]
        self.z = point[2]

        self._class_id = class_id
        self._instance_id = instance_id
        self._color = color
        self._attributes = dict(attributes) if attributes is not None else {}

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @property
    def xyzw(self):
        """Homogeneous coordinates."""
        return np.array([self.x, self.y, self.z, 1.0], dtype=np.float32)

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
        """Serialize keypoint to proto object.

        Does not serialize class or instance information, just point geometry.
        To serialize a complete annotation, see `dgp/annotations/key_point_3d_annotation.py`

        Returns
        -------
        KeyPoint3D.pb2
            As defined in `proto/annotations.proto`
        """
        return annotations_pb2.KeyPoint3D(x=float(self.x), y=float(self.y), z=float(self.z))


class ProbabilisticKeyPoint3D(KeyPoint3D):
    """3D key point object.

    Parameters
    ----------
    point: np.ndarray[np.float32]
        Array of 3 floats describing key point coordinates [x, y, z].

    covariance: np.ndarray[np.float32] | Covariance3D
        Array of 6 floats of shape (6,) or 9 floats of shape (3, 3) or Covariance3D object,
        describing the covariance matrix for this point.

    class_id: int, default: GENERIC_OBJECT_CLASS_ID
        Integer class ID (0 reserved for background).

    instance_id: int, default: None
        Unique instance ID for key point. If None provided, the ID is a hash of the key point
        location and class.

    color: tuple, default: (0, 0, 0)
        RGB tuple for key point color

    attributes: dict, default: None
        Dictionary of attributes associated with key point. If None provided,
        defaults to empty dict.
    """
    def __init__(
        self,
        point: np.ndarray,
        covariance: Union[np.ndarray, Covariance3D],
        class_id: int = GENERIC_OBJECT_CLASS_ID,
        instance_id: Optional[int] = None,
        color: Tuple[int, int, int] = (0, 0, 0),
        attributes: Optional[Dict[str, Any]] = None,
    ):
        KeyPoint3D.__init__(
            self,
            point=point,
            class_id=class_id,
            instance_id=instance_id,
            color=color,
            attributes=attributes,
        )
        if isinstance(covariance, Covariance3D):
            self._cov3 = covariance
        else:
            self._cov3 = Covariance3D(data=covariance)

    @property
    def cov3(self) -> Covariance3D:
        return self._cov3

    @property
    def hexdigest(self):
        return hashlib.md5(
            self.xyz.tobytes() + self.cov3.arr6.tobytes() + bytes(self._class_id) + str(self.attributes).encode("utf8")
        ).hexdigest()

    def __mul__(self, pose: Pose) -> "ProbabilisticKeyPoint3D":
        return ProbabilisticKeyPoint3D(
            point=(self.xyzw @ pose.matrix.T)[:3],
            covariance=pose * self.cov3,
            class_id=self.class_id,
            instance_id=self.instance_id,
            attributes=self.attributes,
        )

    def __rmul__(self, pose: Pose) -> "ProbabilisticKeyPoint3D":
        return ProbabilisticKeyPoint3D(
            point=(pose.matrix @ self.xyzw)[:3],
            covariance=self.cov3 * pose,
            class_id=self.class_id,
            instance_id=self.instance_id,
            attributes=self.attributes,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(xyz={self.xyz}, covariance={self._cov3}, class={self.class_id}, attributes={self.attributes})"

    def to_proto(self):
        """Serialize keypoint to proto object.

        Does not serialize class or instance information, just point geometry.
        To serialize a complete annotation, see `dgp/annotations/key_point_3d_annotation.py`

        Returns
        -------
        ProbabilisticKeyPoint3D.pb2
            As defined in `proto/annotations.proto`
        """
        return annotations_pb2.ProbabilisticKeyPoint3D(
            x=float(self.x),
            y=float(self.y),
            z=float(self.z),
            var_x=float(self.cov3.var_x),
            cov_xy=float(self.cov3.cov_xy),
            cov_xz=float(self.cov3.cov_xz),
            var_y=float(self.cov3.var_y),
            cov_yz=float(self.cov3.cov_yz),
            var_z=float(self.cov3.var_z),
        )
