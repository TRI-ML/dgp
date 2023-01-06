# Copyright 2022 Woven Planet. All rights reserved.
import numpy as np

from dgp.annotations.base_annotation import Annotation
from dgp.annotations.ontology import KeyPointOntology
from dgp.proto.annotations_pb2 import (
    KeyPoint3DAnnotation,
    KeyPoint3DAnnotations,
)
from dgp.utils.protobuf import (
    generate_uid_from_pbobject,
    parse_pbobject,
    save_pbobject_as_json,
)
from dgp.utils.structures.key_point_3d import KeyPoint3D


class KeyPoint3DAnnotationList(Annotation):
    """Container for 3D keypoint annotations.

    Parameters
    ----------
    ontology: KeyPointOntology
        Ontology for 3D keypoint tasks.

    pointlist: list[KeyPoint3D]
        List of KeyPoint3D objects. See `dgp/utils/structures/key_point_3d` for more details.
    """
    def __init__(self, ontology, pointlist):
        super().__init__(ontology)
        assert isinstance(self._ontology, KeyPointOntology), "Trying to load annotation with wrong type of ontology!"
        for point in pointlist:
            assert isinstance(
                point, KeyPoint3D
            ), f"Can only instantate an annotation from a list of KeyPoint3D, not {type(point)}"
        self._pointlist = pointlist

    @classmethod
    def load(cls, annotation_file, ontology):
        """Load annotation from annotation file and ontology.

        Parameters
        ----------
        annotation_file: str or bytes
            Full path to annotation or bytestring

        ontology: KeyPointOntology
            Ontology for 3D keypoint tasks.

        Returns
        -------
        KeyPoint3DAnnotationList
            Annotation object instantiated from file.
        """
        _annotation_pb2 = parse_pbobject(annotation_file, KeyPoint3DAnnotations)
        pointlist = [
            KeyPoint3D(
                point=np.float32([ann.point.x, ann.point.y, ann.point.z]),
                class_id=ontology.class_id_to_contiguous_id[ann.class_id],
                color=ontology.colormap[ann.class_id],
                attributes=getattr(ann, "attributes", {}),
            ) for ann in _annotation_pb2.annotations
        ]
        return cls(ontology, pointlist)

    def to_proto(self):
        """Return annotation as pb object.

        Returns
        -------
        KeyPoint3DAnnotations
            Annotation as defined in `proto/annotations.proto`
        """
        return KeyPoint3DAnnotations(
            annotations=[
                KeyPoint3DAnnotation(
                    class_id=self._ontology.contiguous_id_to_class_id[point.class_id],
                    point=point.to_proto(),
                    attributes=point.attributes
                ) for point in self._pointlist
            ]
        )

    def save(self, save_dir):
        """Serialize Annotation object and saved to specified directory. Annotations are saved in format <save_dir>/<sha>.<ext>

        Parameters
        ----------
        save_dir: str
            Directory in which annotation is saved.

        Returns
        -------
        output_annotation_file: str
            Full path to saved annotation
        """
        return save_pbobject_as_json(self.to_proto(), save_path=save_dir)

    def __len__(self):
        return len(self._pointlist)

    def __getitem__(self, index):
        """Return a single 3D keypoint"""
        return self._pointlist[index]

    def render(self):
        """Batch rendering function for keypoints."""
        raise NotImplementedError

    @property
    def xyz(self):
        """Return points as (N, 3) np.ndarray in format ([x, y, z])"""
        return np.array([point.xyz for point in self._pointlist], dtype=np.float32)

    @property
    def class_ids(self):
        """Return class ID for each point, with ontology applied:
        0 is background, class IDs mapped to a contiguous set.
        """
        return np.array([point.class_id for point in self._pointlist], dtype=np.int64)

    @property
    def attributes(self):
        """Return a list of dictionaries of attribut name to value."""
        return [point.attributes for point in self._pointlist]

    @property
    def instance_ids(self):
        return np.array([point.instance_id for point in self._pointlist], dtype=np.int64)

    @property
    def hexdigest(self):
        """Reproducible hash of annotation."""
        return generate_uid_from_pbobject(self.to_proto())
