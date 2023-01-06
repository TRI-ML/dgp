# Copyright 2021 Toyota Research Institute.  All rights reserved.
import numpy as np

from dgp.annotations.base_annotation import Annotation
from dgp.annotations.ontology import KeyPointOntology
from dgp.proto.annotations_pb2 import (
    KeyPoint2DAnnotation,
    KeyPoint2DAnnotations,
)
from dgp.utils.protobuf import (
    generate_uid_from_pbobject,
    parse_pbobject,
    save_pbobject_as_json,
)
from dgp.utils.structures.key_point_2d import KeyPoint2D


class KeyPoint2DAnnotationList(Annotation):
    """Container for 2D keypoint annotations.

    Parameters
    ----------
    ontology: KeyPointOntology
        Ontology for 2D keypoint tasks.

    pointlist: list[KeyPoint2D]
        List of KeyPoint2D objects. See `dgp/utils/structures/key_point_2d` for more details.
    """
    def __init__(self, ontology, pointlist):
        super().__init__(ontology)
        assert isinstance(self._ontology, KeyPointOntology), "Trying to load annotation with wrong type of ontology!"
        for point in pointlist:
            assert isinstance(
                point, KeyPoint2D
            ), f"Can only instantate an annotation from a list of KeyPoint2D, not {type(point)}"
        self.pointlist = pointlist

    @classmethod
    def load(cls, annotation_file, ontology):
        """Load annotation from annotation file and ontology.

        Parameters
        ----------
        annotation_file: str or bytes
            Full path to annotation or bytestring

        ontology: KeyPointOntology
            Ontology for 2D keypoint tasks.

        Returns
        -------
        KeyPoint2DAnnotationList
            Annotation object instantiated from file.
        """
        _annotation_pb2 = parse_pbobject(annotation_file, KeyPoint2DAnnotations)
        pointlist = [
            KeyPoint2D(
                point=np.float32([ann.point.x, ann.point.y]),
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
        KeyPoint2DAnnotations
            Annotation as defined in `proto/annotations.proto`
        """
        return KeyPoint2DAnnotations(
            annotations=[
                KeyPoint2DAnnotation(
                    class_id=self._ontology.contiguous_id_to_class_id[point.class_id],
                    point=point.to_proto(),
                    attributes=point.attributes
                ) for point in self.pointlist
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
        return len(self.pointlist)

    def __getitem__(self, index):
        """Return a single 2D keypoint"""
        return self.pointlist[index]

    def render(self):
        """TODO: Batch rendering function for keypoints."""
        raise NotImplementedError

    @property
    def xy(self):
        """Return points as (N, 2) np.ndarray in format ([x, y])"""
        return np.array([point.xy for point in self.pointlist], dtype=np.float32)

    @property
    def class_ids(self):
        """Return class ID for each point, with ontology applied:
        0 is background, class IDs mapped to a contiguous set.
        """
        return np.array([point.class_id for point in self.pointlist], dtype=np.int64)

    @property
    def attributes(self):
        """Return a list of dictionaries of attribut name to value."""
        return [point.attributes for point in self.pointlist]

    @property
    def instance_ids(self):
        return np.array([point.instance_id for point in self.pointlist], dtype=np.int64)

    @property
    def hexdigest(self):
        """Reproducible hash of annotation."""
        return generate_uid_from_pbobject(self.to_proto())
