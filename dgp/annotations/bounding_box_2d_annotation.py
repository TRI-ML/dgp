# Copyright 2021 Toyota Research Institute.  All rights reserved.
import numpy as np

from dgp.annotations.base_annotation import Annotation
from dgp.annotations.ontology import BoundingBoxOntology
from dgp.proto.annotations_pb2 import (
    BoundingBox2DAnnotation,
    BoundingBox2DAnnotations,
)
from dgp.utils.protobuf import (
    generate_uid_from_pbobject,
    parse_pbobject,
    save_pbobject_as_json,
)
from dgp.utils.structures.bounding_box_2d import BoundingBox2D


class BoundingBox2DAnnotationList(Annotation):
    """Container for 2D bounding box annotations.

    Parameters
    ----------
    ontology: BoundingBoxOntology
        Ontology for 2D bounding box tasks.

    boxlist: list[BoundingBox2D]
        List of BoundingBox2D objects. See `dgp/utils/structures/bounding_box_2d` for more details.
    """
    def __init__(self, ontology, boxlist):
        super().__init__(ontology)
        assert isinstance(self._ontology, BoundingBoxOntology), "Trying to load annotation with wrong type of ontology!"
        for box in boxlist:
            assert isinstance(
                box, BoundingBox2D
            ), f"Can only instantate an annotation from a list of BoundingBox2D, not {type(box)}"
        self.boxlist = boxlist

    @classmethod
    def load(cls, annotation_file, ontology):
        """Load annotation from annotation file and ontology.

        Parameters
        ----------
        annotation_file: str or bytes
            Full path to annotation or bytestring

        ontology: BoundingBoxOntology
            Ontology for 2D bounding box tasks.

        Returns
        -------
        BoundingBox2DAnnotationList
            Annotation object instantiated from file.
        """
        _annotation_pb2 = parse_pbobject(annotation_file, BoundingBox2DAnnotations)
        boxlist = [
            BoundingBox2D(
                box=np.float32([ann.box.x, ann.box.y, ann.box.w, ann.box.h]),
                class_id=ontology.class_id_to_contiguous_id[ann.class_id],
                instance_id=ann.instance_id,
                color=ontology.colormap[ann.class_id],
                attributes=getattr(ann, "attributes", {}),
            ) for ann in _annotation_pb2.annotations
        ]
        return cls(ontology, boxlist)

    def to_proto(self):
        """Return annotation as pb object.

        Returns
        -------
        BoundingBox2DAnnotations
            Annotation as defined in `proto/annotations.proto`
        """
        return BoundingBox2DAnnotations(
            annotations=[
                BoundingBox2DAnnotation(
                    class_id=self._ontology.contiguous_id_to_class_id[box.class_id],
                    box=box.to_proto(),
                    area=int(box.area),
                    instance_id=box.instance_id,
                    attributes=box.attributes
                ) for box in self.boxlist
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
        return len(self.boxlist)

    def __getitem__(self, index):
        """Return a single 3D bounding box"""
        return self.boxlist[index]

    def render(self):
        """TODO: Batch rendering function for bounding boxes."""
        raise NotImplementedError

    @property
    def ltrb(self):
        """Return boxes as (N, 4) np.ndarray in format ([left, top, right, bottom])"""
        return np.array([box.ltrb for box in self.boxlist], dtype=np.float32)

    @property
    def ltwh(self):
        """Return boxes as (N, 4) np.ndarray in format ([left, top, width, height])"""
        return np.array([box.ltwh for box in self.boxlist], dtype=np.float32)

    @property
    def class_ids(self):
        """Return class ID for each box, with ontology applied:
        0 is background, class IDs mapped to a contiguous set.
        """
        return np.array([box.class_id for box in self.boxlist], dtype=np.int64)

    @property
    def attributes(self):
        """Return a list of dictionaries of attribute name to value."""
        return [box.attributes for box in self.boxlist]

    @property
    def instance_ids(self):
        return np.array([box.instance_id for box in self.boxlist], dtype=np.int64)

    @property
    def hexdigest(self):
        """Reproducible hash of annotation."""
        return generate_uid_from_pbobject(self.to_proto())
