# Copyright 2022 Woven Planet.  All rights reserved.
"""Wicker conversion methods"""
# pylint: disable=arguments-renamed
# pylint: disable=unused-argument
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
import base64
import io
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image
from wicker.schema import BytesField, LongField, NumpyField, StringField

from dgp.annotations import (
    ONTOLOGY_REGISTRY,
    BoundingBox2DAnnotationList,
    BoundingBox3DAnnotationList,
    DenseDepthAnnotation,
    KeyLine2DAnnotationList,
    KeyLine3DAnnotationList,
    KeyPoint2DAnnotationList,
    KeyPoint3DAnnotationList,
    PanopticSegmentation2DAnnotation,
    SemanticSegmentation2DAnnotation,
    ontology,
)
from dgp.annotations.ontology import Ontology
from dgp.utils.pose import Pose
from dgp.utils.protobuf import open_ontology_pbobject

WICKER_RAW_NONE_VALUE = b'\x00\x00\x00\x00'

# NOTE: all of these unwicker methods would be vastly simpler if we modify dgp annotations to support
# saving/loading into file like objects. TODO: (chris.ochoa)


class WickerSerializer(ABC):
    @abstractmethod
    def schema(self, name: str, data: Any) -> Any:
        """Returns a wicker schema object.

        Parameters
        ----------
        name: str
            The name of the key in the wicker schema.

        data: Any
            Example raw data to serialize. Some wicker types need additional information,
            for example numpy arrays need to know the shape.

        Returns
        -------
        schema: The schema object.
        """
        raise NotImplementedError

    def serialize(self, data: Any) -> Any:
        """Convert data for consumption by native wicker types.

        Parameters
        ----------
        data: Any
            The raw data to convert.

        Returns
        -------
        data: Any
            The converted data. This should be consumable by the schema object returned by self.schema.
        """
        return data

    def unserialize(self, raw: Any) -> Any:
        """Convert raw wicker data to a useable type.

        Parameters
        ----------
        raw: Any
            The raw wicker data. Input depends on self.schema. Raw will be a  string for wicker StringField,
            a numpy array for NumpyField or bytes for ByteField etc.

        Returns
        -------
        output: Any
            The converted output
        """
        return raw


class StringSerializer(WickerSerializer):
    def schema(self, name: str, data: str):
        return StringField(name)


class LongSerializer(WickerSerializer):
    def schema(self, name: str, data: int):
        return LongField(name, data)


class DistortionSerializer(WickerSerializer):
    def schema(self, name: str, data: Any):
        return StringField(name)

    def serialize(self, data: Dict[str, float]) -> str:
        return json.dumps(data)

    def unserialize(self, raw: str) -> Dict[str, float]:
        return json.loads(raw)


class PoseSerializer(WickerSerializer):
    def schema(self, name: str, data: Any):
        return NumpyField(name, shape=(4, 4), dtype='float64')

    def serialize(self, pose: Pose) -> np.ndarray:
        return pose.matrix.astype(np.float64)

    def unserialize(self, raw: np.ndarray) -> Pose:
        return Pose.from_matrix(raw)


class IntrinsicsSerializer(WickerSerializer):
    def schema(self, name: str, data: Any):
        return NumpyField(name, shape=(3, 3), dtype='float32')

    def serialize(self, K: np.ndarray) -> np.ndarray:
        return K.astype(np.float32)


class RGBSerializer(WickerSerializer):
    def schema(self, name: str, data: Any):
        return BytesField(name, is_heavy_pointer=True)

    def serialize(self, rgb: Image.Image) -> bytes:
        rgb_bytes = io.BytesIO()
        rgb.save(rgb_bytes, "JPEG")
        return rgb_bytes.getvalue()

    def unserialize(self, raw: np.ndarray) -> Image.Image:
        return Image.open(io.BytesIO(raw))


class PointCloudSerializer(WickerSerializer):
    def schema(self, name: str, data: Any):
        # Support shapes like (N,3) for points, (N,) for offsets, (N,3,3) for covariances etc
        shp = (-1, *data.shape[1:])
        return NumpyField(name, shape=shp, dtype='float32', is_heavy_pointer=True)

    def serialize(self, point_cloud: np.ndarray) -> np.ndarray:
        return point_cloud.astype(np.float32)


class BoundingBox2DSerializer(WickerSerializer):
    def __init__(self, ):
        super().__init__()
        self._ontology = None

    @property
    def ontology(self) -> Ontology:
        return self._ontology

    @ontology.setter
    def ontology(self, ontology: Ontology):
        self._ontology = ontology

    def schema(self, name: str, data: Any):
        return BytesField(name, required=False, is_heavy_pointer=True)

    def serialize(self, annotation: Optional[BoundingBox2DAnnotationList]) -> bytes:
        if annotation is None:
            return WICKER_RAW_NONE_VALUE
        return annotation.to_proto().SerializeToString()

    def unserialize(self, raw: bytes) -> BoundingBox2DAnnotationList:
        if raw == WICKER_RAW_NONE_VALUE or self.ontology is None:
            return None

        return BoundingBox2DAnnotationList.load(raw, self.ontology)


class BoundingBox3DSerializer(WickerSerializer):
    def __init__(self, ):
        super().__init__()
        self.ontology = None

    @property
    def ontology(self) -> Ontology:
        return self._ontology

    @ontology.setter
    def ontology(self, ontology: Ontology):
        self._ontology = ontology

    def schema(self, name: str, data: Any):
        return BytesField(name, required=False, is_heavy_pointer=True)

    def serialize(self, annotation: Optional[BoundingBox3DAnnotationList]) -> bytes:
        if annotation is None:
            return WICKER_RAW_NONE_VALUE
        return annotation.to_proto().SerializeToString()

    def unserialize(self, raw: bytes) -> BoundingBox3DAnnotationList:
        if raw == WICKER_RAW_NONE_VALUE or self.ontology is None:
            return None

        return BoundingBox3DAnnotationList.load(raw, self.ontology)


class SemanticSegmentation2DSerializer(WickerSerializer):
    def __init__(self, ):
        super().__init__()
        self.ontology = None

    @property
    def ontology(self) -> Ontology:
        return self._ontology

    @ontology.setter
    def ontology(self, ontology: Ontology):
        self._ontology = ontology

    def schema(self, name: str, data: Any):
        return BytesField(name, required=False, is_heavy_pointer=True)

    def serialize(self, annotation: SemanticSegmentation2DAnnotation) -> bytes:
        if annotation is None:
            return WICKER_RAW_NONE_VALUE

        # Save the image as PNG
        segmentation_image = annotation._convert_contiguous_to_class()
        _, buffer = cv2.imencode(".png", segmentation_image)
        return io.BytesIO(buffer).getvalue()

    def unserialize(self, raw: bytes) -> SemanticSegmentation2DAnnotation:
        if raw == WICKER_RAW_NONE_VALUE or self.ontology is None:
            return None
        return SemanticSegmentation2DAnnotation.load(raw, self.ontology)

    def set_ontology(self, ontology: Ontology):
        self.ontology = ontology


class InstanceSegmentation2DSerializer(WickerSerializer):
    def __init__(self, ):
        super().__init__()
        self.ontology = None

    @property
    def ontology(self) -> Ontology:
        return self._ontology

    @ontology.setter
    def ontology(self, ontology: Ontology):
        self._ontology = ontology

    def schema(self, name: str, data: Any):
        return BytesField(name, required=False, is_heavy_pointer=True)

    def serialize(self, annotation: PanopticSegmentation2DAnnotation) -> bytes:
        if annotation is None:
            return WICKER_RAW_NONE_VALUE

        _, buffer = cv2.imencode(".png", annotation.panoptic_image)
        panoptic_image = io.BytesIO(buffer).getvalue()
        index_to_label = json.dumps(annotation.index_to_label).encode('utf-8')
        ann_bytes = {
            'panoptic_image': base64.b64encode(panoptic_image).decode(),
            'index_to_label': base64.b64encode(index_to_label).decode()
        }
        return json.dumps(ann_bytes).encode('utf-8')

    def unserialize(self, raw: bytes) -> PanopticSegmentation2DAnnotation:
        if raw == WICKER_RAW_NONE_VALUE or self.ontology is None:
            return None

        raw_dict = json.loads(raw)
        raw_bytes = io.BytesIO(base64.b64decode(raw_dict['panoptic_image']))
        panoptic_image = cv2.imdecode(np.frombuffer(raw_bytes.getbuffer(), np.uint8), cv2.IMREAD_UNCHANGED)
        if len(panoptic_image.shape) == 3:
            _L = panoptic_image
            label_map = _L[:, :, 2] + 256 * _L[:, :, 1] + 256 * 256 * _L[:, :, 0]
            panoptic_image = label_map.astype(PanopticSegmentation2DAnnotation.DEFAULT_PANOPTIC_IMAGE_DTYPE)

        raw_bytes = io.BytesIO(base64.b64decode(raw_dict['index_to_label']))
        index_to_label = json.loads(raw_bytes.getvalue())

        return PanopticSegmentation2DAnnotation(self.ontology, panoptic_image, index_to_label)


class DepthSerializer():
    def schema(self, name: str, data: DenseDepthAnnotation):
        return NumpyField(name, shape=data.depth.shape, dtype='float32', is_heavy_pointer=True)

    def serialize(self, depth: DenseDepthAnnotation) -> np.ndarray:
        return depth.depth.astype(np.float32)

    def unserialize(self, raw: bytes) -> DenseDepthAnnotation:
        return DenseDepthAnnotation(raw)


class OntologySerializer():
    def __init__(self, ontology_type: str):
        super().__init__()
        self.ontology_type = ontology_type

    def schema(self, name: str, data: Ontology):
        return BytesField(name)

    def serialize(self, ontology: Ontology) -> bytes:
        return ontology.to_proto().SerializeToString()

    def unserialize(self, raw: bytes) -> Ontology:
        if raw == WICKER_RAW_NONE_VALUE:
            return None

        ontology = open_ontology_pbobject(raw)
        return ONTOLOGY_REGISTRY[self.ontology_type](ontology)


class KeyPoint2DSerializer(WickerSerializer):
    def __init__(self, ):
        super().__init__()
        self._ontology = None

    @property
    def ontology(self) -> Ontology:
        return self._ontology

    @ontology.setter
    def ontology(self, ontology: Ontology):
        self._ontology = ontology

    def schema(self, name: str, data: Any):
        return BytesField(name, required=False, is_heavy_pointer=True)

    def serialize(self, annotation: Optional[KeyPoint2DAnnotationList]) -> bytes:
        if annotation is None:
            return WICKER_RAW_NONE_VALUE
        return annotation.to_proto().SerializeToString()

    def unserialize(self, raw: bytes) -> KeyPoint2DAnnotationList:
        if raw == WICKER_RAW_NONE_VALUE or self.ontology is None:
            return None

        return KeyPoint2DAnnotationList.load(raw, self.ontology)


class KeyLine2DSerializer(WickerSerializer):
    def __init__(self, ):
        super().__init__()
        self._ontology = None

    @property
    def ontology(self) -> Ontology:
        return self._ontology

    @ontology.setter
    def ontology(self, ontology: Ontology):
        self._ontology = ontology

    def schema(self, name: str, data: Any):
        return BytesField(name, required=False, is_heavy_pointer=True)

    def serialize(self, annotation: Optional[KeyLine2DAnnotationList]) -> bytes:
        if annotation is None:
            return WICKER_RAW_NONE_VALUE
        return annotation.to_proto().SerializeToString()

    def unserialize(self, raw: bytes) -> KeyLine2DAnnotationList:
        if raw == WICKER_RAW_NONE_VALUE or self.ontology is None:
            return None

        return KeyLine2DAnnotationList.load(raw, self.ontology)


class KeyPoint3DSerializer(WickerSerializer):
    def __init__(self, ):
        super().__init__()
        self._ontology = None

    @property
    def ontology(self) -> Ontology:
        return self._ontology

    @ontology.setter
    def ontology(self, ontology: Ontology):
        self._ontology = ontology

    def schema(self, name: str, data: Any):
        return BytesField(name, required=False, is_heavy_pointer=True)

    def serialize(self, annotation: Optional[KeyPoint3DAnnotationList]) -> bytes:
        if annotation is None:
            return WICKER_RAW_NONE_VALUE
        return annotation.to_proto().SerializeToString()

    def unserialize(self, raw: bytes) -> KeyPoint3DAnnotationList:
        if raw == WICKER_RAW_NONE_VALUE or self.ontology is None:
            return None

        return KeyPoint3DAnnotationList.load(raw, self.ontology)


class KeyLine3DSerializer(WickerSerializer):
    def __init__(self, ):
        super().__init__()
        self._ontology = None

    @property
    def ontology(self) -> Ontology:
        return self._ontology

    @ontology.setter
    def ontology(self, ontology: Ontology):
        self._ontology = ontology

    def schema(self, name: str, data: Any):
        return BytesField(name, required=False, is_heavy_pointer=True)

    def serialize(self, annotation: Optional[KeyLine3DAnnotationList]) -> bytes:
        if annotation is None:
            return WICKER_RAW_NONE_VALUE
        return annotation.to_proto().SerializeToString()

    def unserialize(self, raw: bytes) -> KeyLine3DAnnotationList:
        if raw == WICKER_RAW_NONE_VALUE or self.ontology is None:
            return None

        return KeyLine3DAnnotationList.load(raw, self.ontology)
