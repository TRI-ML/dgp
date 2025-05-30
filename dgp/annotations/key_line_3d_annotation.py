# Copyright 2022 Woven Planet. All rights reserved.
from typing import Any, Dict, List, Union

import numpy as np

from dgp.annotations.base_annotation import Annotation
from dgp.annotations.ontology import KeyLineOntology
from dgp.proto.annotations_pb2 import KeyLine3DAnnotation, KeyLine3DAnnotations
from dgp.proto.annotations_pb2 import \
    ProbabilisticKeyLine3DAnnotation as ProtoProbabilisticKeyLine3DAnnotation
from dgp.proto.annotations_pb2 import \
    ProbabilisticKeyLine3DAnnotations as ProtoProbabilisticKeyLine3DAnnotations
from dgp.proto.annotations_pb2 import \
    ProbabilisticKeyPoint3D as ProtoProbabilisticKeyPoint3D
from dgp.utils.protobuf import (
    generate_uid_from_pbobject,
    parse_pbobject,
    save_pbobject_as_json,
)
from dgp.utils.structures.key_line_3d import KeyLine3D, ProbabilisticKeyLine3D
from dgp.utils.structures.key_point_3d import (
    KeyPoint3D,
    ProbabilisticKeyPoint3D,
)


class KeyLine3DAnnotationList(Annotation):
    """Container for 3D keyline annotations.

    Parameters
    ----------
    ontology: KeyLineOntology
        Ontology for 3D keyline tasks.

    linelist: list[KeyLine3D]
        List of KeyLine3D objects. See `dgp/utils/structures/key_line_3d` for more details.
    """
    def __init__(self, ontology, linelist):
        super().__init__(ontology)
        assert isinstance(self._ontology, KeyLineOntology), "Trying to load annotation with wrong type of ontology!"
        for line in linelist:
            assert isinstance(
                line, KeyLine3D
            ), f"Can only instantate an annotation from a list of KeyLine3D, not {type(line)}"
        self._linelist = linelist

    @classmethod
    def load(cls, annotation_file, ontology):
        """Load annotation from annotation file and ontology.

        Parameters
        ----------
        annotation_file: str or bytes
            Full path to annotation or bytestring

        ontology: KeyLineOntology
            Ontology for 3D keyline tasks.

        Returns
        -------
        KeyLine3DAnnotationList
            Annotation object instantiated from file.
        """
        _annotation_pb2 = parse_pbobject(annotation_file, KeyLine3DAnnotations)
        linelist = [
            KeyLine3D(
                line=np.float32([[vertex.x, vertex.y, vertex.z] for vertex in ann.vertices]),
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
        KeyLine3DAnnotations
            Annotation as defined in `proto/annotations.proto`
        """
        return KeyLine3DAnnotations(
            annotations=[
                KeyLine3DAnnotation(
                    class_id=self._ontology.contiguous_id_to_class_id[line.class_id],
                    vertices=[
                        KeyPoint3D(
                            point=np.float32([x, y, z]),
                            class_id=line.class_id,
                            instance_id=line.instance_id,
                            color=line.color,
                            attributes=line.attributes,
                        ).to_proto() for x, y, z in zip(line.x, line.y, line.z)
                    ],
                    attributes=line.attributes,
                ) for line in self._linelist
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
        return len(self._linelist)

    def __getitem__(self, index):
        """Return a single 3D keyline"""
        return self._linelist[index]

    def render(self):
        """Batch rendering function for keylines."""
        raise NotImplementedError

    @property
    def xyz(self):
        """Return lines as (N, 3) np.ndarray in format ([x, y, z])"""
        return np.array([line.xyz.tolist() for line in self._linelist], dtype=np.float32)

    @property
    def class_ids(self):
        """Return class ID for each line, with ontology applied:
        class IDs mapped to a contiguous set.
        """
        return np.array([line.class_id for line in self._linelist], dtype=np.int64)

    @property
    def attributes(self):
        """Return a list of dictionaries of attribute name to value."""
        return [line.attributes for line in self._linelist]

    @property
    def instance_ids(self):
        return np.array([line.instance_id for line in self._linelist], dtype=np.int64)

    @property
    def hexdigest(self):
        """Reproducible hash of annotation."""
        return generate_uid_from_pbobject(self.to_proto())


class ProbabilisticKeyLine3DAnnotationList(Annotation):
    """Container for probabilistic 3D key line annotations.

    Parameters
    ----------
    ontology: KeyLineOntology
        Ontology for 3D key line tasks.

    linelist: list[ProbabilisticKeyLine3D]
        List of ProbabilisticKeyLine3D objects. See `dgp/utils/structures/key_line_3d` for more details.
    """
    def __init__(self, ontology: KeyLineOntology, linelist: List[ProbabilisticKeyLine3D]) -> None:
        Annotation.__init__(self, ontology)
        assert isinstance(self._ontology, KeyLineOntology), "Trying to load annotation with wrong type of ontology!"
        for line in linelist:
            assert isinstance(
                line, ProbabilisticKeyLine3D
            ), f"Can only instantiate an annotation from a list of ProbabilisticKeyLine3D, not {type(line)}!"
        self._linelist = linelist

    @classmethod
    def load(
        cls, annotation_file: Union[str, bytes], ontology: KeyLineOntology
    ) -> "ProbabilisticKeyLine3DAnnotationList":
        """Load annotation from annotation file and ontology.

        Parameters
        ----------
        annotation_file: str or bytes
            Full path to annotation or bytestring

        ontology: KeyLineOntology
            Ontology for 3D key line tasks.

        Returns
        -------
        KeyLine3DAnnotationList
            Annotation object instantiated from file.
        """
        _annotation_pb2 = parse_pbobject(
            annotation_file,
            ProtoProbabilisticKeyLine3DAnnotations,
        )
        annotation_list = [
            ProbabilisticKeyLine3D(
                points=[
                    ProbabilisticKeyPoint3D(
                        point=np.asarray([
                            vert.x,
                            vert.y,
                            vert.z,
                        ]),
                        covariance=np.asarray([
                            vert.var_x,
                            vert.cov_xy,
                            vert.cov_xz,
                            vert.var_y,
                            vert.cov_yz,
                            vert.var_z,
                        ]),
                    ) for vert in annotation.vertices
                ],
                class_id=ontology.class_id_to_contiguous_id[annotation.class_id],
                instance_id=annotation.key,
                color=ontology.colormap[annotation.class_id],
                attributes=getattr(annotation, "attributes", {}),
            ) for annotation in _annotation_pb2.annotations
        ]
        return cls(ontology=ontology, linelist=annotation_list)

    def to_proto(self):
        """Return annotation as pb object.

        Returns
        -------
        KeyLine3DAnnotations
            Annotation as defined in `proto/annotations.proto`
        """
        return ProtoProbabilisticKeyLine3DAnnotations(
            annotations=[
                ProtoProbabilisticKeyLine3DAnnotation(
                    class_id=self._ontology.contiguous_id_to_class_id[line.class_id],
                    vertices=[
                        ProtoProbabilisticKeyPoint3D(
                            x=point.x,
                            y=point.y,
                            z=point.z,
                            var_x=point.cov3.var_x,
                            cov_xy=point.cov3.cov_xy,
                            cov_xz=point.cov3.cov_xz,
                            var_y=point.cov3.var_y,
                            cov_yz=point.cov3.cov_yz,
                            var_z=point.cov3.var_z,
                        ) for point in line
                    ],
                    key=line.instance_id,
                    attributes=line.attributes,
                ) for line in self._linelist
            ]
        )

    def save(self, save_dir: str) -> None:
        """Serialize Annotation object and saved to specified directory.

        Annotations are saved in format <save_dir>/<sha>.<ext>

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

    def __len__(self) -> int:
        return len(self._linelist)

    def __getitem__(self, index: int) -> ProbabilisticKeyLine3D:
        """Return a single 3D keyline"""
        return self._linelist[index]

    def render(self):
        """Batch rendering function for keylines."""
        raise NotImplementedError

    @property
    def xyz(self) -> np.ndarray:
        """Return lines as (N, 3) np.ndarray in format ([x, y, z])"""
        return np.array([line.xyz.tolist() for line in self._linelist], dtype=np.float32)

    @property
    def class_ids(self) -> np.ndarray:
        """Return class ID for each line, with ontology applied:
        class IDs mapped to a contiguous set.
        """
        return np.array([line.class_id for line in self._linelist], dtype=np.int64)

    @property
    def attributes(self) -> List[Dict[str, Any]]:
        """Return a list of dictionaries of attribute name to value."""
        return [line.attributes for line in self._linelist]

    @property
    def instance_ids(self) -> np.ndarray:
        return np.array([line.instance_id for line in self._linelist], dtype=np.int64)

    @property
    def hexdigest(self) -> str:
        """Reproducible hash of annotation."""
        return generate_uid_from_pbobject(self.to_proto())
