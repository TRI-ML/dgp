# Copyright 2021 Toyota Research Institute.  All rights reserved.
import numpy as np

from dgp.annotations.base_annotation import Annotation
from dgp.annotations.ontology import KeyLineOntology
from dgp.proto.annotations_pb2 import (KeyLine2DAnnotation, KeyLine2DAnnotations)
from dgp.utils.protobuf import (generate_uid_from_pbobject, open_pbobject, save_pbobject_as_json)
from dgp.utils.structures.key_point_2d import KeyLine2D


class KeyLine2DAnnotationList(Annotation):
    """Container for 2D keyline annotations.

    Parameters
    ----------
    ontology: KeyLineOntology
        Ontology for 2D keyline tasks.

    pointlist: list[KeyLine2D]
        List of KeyLine2D objects. See `dgp/utils/structures/key_point_2d` for more details.
    """
    def __init__(self, ontology, pointlist):
        super().__init__(ontology)
        assert isinstance(self._ontology, KeyLineOntology), "Trying to load annotation with wrong type of ontology!"
        for point in pointlist:
            assert isinstance(
                point, KeyLine2D
            ), f"Can only instantate an annotation from a list of KeyLine2D, not {type(point)}"
        self.pointlist = pointlist

    @classmethod
    def load(cls, annotation_file, ontology):
        """Load annotation from annotation file and ontology.

        Parameters
        ----------
        annotation_file: str
            Full path to annotation

        ontology: KeyLineOntology
            Ontology for 2D keyline tasks.

        Returns
        -------
        KeyLine2DAnnotationList
            Annotation object instantiated from file.
        """
        _annotation_pb2 = open_pbobject(annotation_file, KeyLine2DAnnotations)
        pointlist = [
            KeyLine2D(
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
        KeyLine2DAnnotations
            Annotation as defined in `proto/annotations.proto`
        """
        return KeyLine2DAnnotations(
            annotations=[
                KeyLine2DAnnotation(
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
        """Return a single 2D keyline"""
        return self.pointlist[index]

    def render(self):
        """TODO: Batch rendering function for keylines."""
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
