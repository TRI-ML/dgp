# Copyright 2021 Toyota Research Institute.  All rights reserved.
import numpy as np

from dgp.annotations.base_annotation import Annotation
from dgp.annotations.ontology import KeyLineOntology
from dgp.proto.annotations_pb2 import KeyLine2DAnnotation, KeyLine2DAnnotations
from dgp.utils.protobuf import (
    generate_uid_from_pbobject,
    parse_pbobject,
    save_pbobject_as_json,
)
from dgp.utils.structures.key_line_2d import KeyLine2D
from dgp.utils.structures.key_point_2d import KeyPoint2D


class KeyLine2DAnnotationList(Annotation):
    """Container for 2D keyline annotations.

    Parameters
    ----------
    ontology: KeyLineOntology
        Ontology for 2D keyline tasks.

    linelist: list[KeyLine2D]
        List of KeyLine2D objects. See `dgp/utils/structures/key_line_2d` for more details.
    """
    def __init__(self, ontology, linelist):
        super().__init__(ontology)
        assert isinstance(self._ontology, KeyLineOntology), "Trying to load annotation with wrong type of ontology!"
        for line in linelist:
            assert isinstance(
                line, KeyLine2D
            ), f"Can only instantate an annotation from a list of KeyLine2D, not {type(line)}"
        self.linelist = linelist

    @classmethod
    def load(cls, annotation_file, ontology):
        """Load annotation from annotation file and ontology.

        Parameters
        ----------
        annotation_file: str or bytes
            Full path to annotation or bytestring

        ontology: KeyLineOntology
            Ontology for 2D keyline tasks.

        Returns
        -------
        KeyLine2DAnnotationList
            Annotation object instantiated from file.
        """
        _annotation_pb2 = parse_pbobject(annotation_file, KeyLine2DAnnotations)
        linelist = [
            KeyLine2D(
                line=np.float32([[vertex.x, vertex.y] for vertex in ann.vertices]),
                class_id=ontology.class_id_to_contiguous_id[ann.class_id],
                color=ontology.colormap[ann.class_id],
                attributes=getattr(ann, "attributes", {}),
            ) for ann in _annotation_pb2.annotations
        ]
        return cls(ontology, linelist)

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
                    class_id=self._ontology.contiguous_id_to_class_id[line.class_id],
                    vertices=[
                        KeyPoint2D(
                            point=np.float32([x, y]),
                            class_id=line.class_id,
                            instance_id=line.instance_id,
                            color=line.color,
                            attributes=line.attributes
                        ).to_proto() for x, y in zip(line.x, line.y)
                    ],
                    attributes=line.attributes
                ) for line in self.linelist
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
            Full path to saved annotation.
        """
        return save_pbobject_as_json(self.to_proto(), save_path=save_dir)

    def __len__(self):
        return len(self.linelist)

    def __getitem__(self, index):
        """Return a single 2D keyline"""
        return self.linelist[index]

    def render(self):
        """TODO: Batch rendering function for keylines."""
        raise NotImplementedError

    @property
    def xy(self):
        """Return lines as (N, 2) np.ndarray in format ([x, y])"""
        return np.array([line.xy.tolist() for line in self.linelist], dtype=np.float32)

    @property
    def class_ids(self):
        """Return class ID for each line, with ontology applied:
        class IDs mapped to a contiguous set.
        """
        return np.array([line.class_id for line in self.linelist], dtype=np.int64)

    @property
    def attributes(self):
        """Return a list of dictionaries of attribut name to value."""
        return [line.attributes for line in self.linelist]

    @property
    def instance_ids(self):
        return np.array([line.instance_id for line in self.linelist], dtype=np.int64)

    @property
    def hexdigest(self):
        """Reproducible hash of annotation."""
        return generate_uid_from_pbobject(self.to_proto())
