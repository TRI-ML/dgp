# Copyright 2021 Toyota Research Institute.  All rights reserved.
import numpy as np

from dgp.annotations.base_annotation import Annotation
from dgp.annotations.ontology import BoundingBoxOntology
from dgp.proto.annotations_pb2 import (
    BoundingBox3DAnnotation,
    BoundingBox3DAnnotations,
)
from dgp.utils.camera import Camera
from dgp.utils.pose import Pose
from dgp.utils.protobuf import (
    generate_uid_from_pbobject,
    parse_pbobject,
    save_pbobject_as_json,
)
from dgp.utils.structures.bounding_box_3d import BoundingBox3D


class BoundingBox3DAnnotationList(Annotation):
    """Container for 3D bounding box annotations.

    Parameters
    ----------
    ontology: BoundingBoxOntology
        Ontology for 3D bounding box tasks.

    boxlist: list[BoundingBox3D]
        List of BoundingBox3D objects. See `utils/structures/bounding_box_3d`
        for more details.
    """
    def __init__(self, ontology, boxlist):
        super().__init__(ontology)
        assert isinstance(self._ontology, BoundingBoxOntology), "Trying to load annotation with wrong type of ontology!"

        for box in boxlist:
            assert isinstance(
                box, BoundingBox3D
            ), f"Can only instantiate an annotation from a list of BoundingBox3D, not {type(box)}"
        self.boxlist = boxlist

    @classmethod
    def load(cls, annotation_file, ontology):
        """Load annotation from annotation file and ontology.

        Parameters
        ----------
        annotation_file: str or bytes
            Full path to annotation or bytestring

        ontology: BoundingBoxOntology
            Ontology for 3D bounding box tasks.

        Returns
        -------
        BoundingBox3DAnnotationList
            Annotation object instantiated from file.
        """
        _annotation_pb2 = parse_pbobject(annotation_file, BoundingBox3DAnnotations)
        boxlist = [
            BoundingBox3D(
                pose=Pose.load(ann.box.pose),
                sizes=np.float32([ann.box.width, ann.box.length, ann.box.height]),
                class_id=ontology.class_id_to_contiguous_id[ann.class_id],
                instance_id=ann.instance_id,
                color=ontology.colormap[ann.class_id],
                attributes=getattr(ann, "attributes", {}),
                num_points=ann.num_points,
                occlusion=ann.box.occlusion,
                truncation=ann.box.truncation
            ) for ann in _annotation_pb2.annotations
        ]
        return cls(ontology, boxlist)

    def to_proto(self):
        """Return annotation as pb object.

        Returns
        -------
        BoundingBox3DAnnotations
            Annotation as defined `proto/annotations.proto`
        """
        return BoundingBox3DAnnotations(
            annotations=[
                BoundingBox3DAnnotation(
                    class_id=self._ontology.contiguous_id_to_class_id[box.class_id],
                    box=box.to_proto(),
                    instance_id=box.instance_id,
                    attributes=box.attributes,
                    num_points=box.num_points
                ) for box in self.boxlist
            ]
        )

    def save(self, save_dir):
        """Serialize Annotation object and saved to specified directory. Annotations are saved in format <save_dir>/<sha>.<ext>

        Parameters
        ----------
        save_dir: str
            A pathname to a directory to save the annotation object into.

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

    def render(self, image, camera, line_thickness=2, font_scale=0.5):
        """Render the 3D boxes in this annotation on the image in place

        Parameters
        ----------
        image: np.uint8
            Image (H, W, C) to render the bounding box onto. We assume the input image is in *RGB* format.
            Element type must be uint8.

        camera: dgp.utils.camera.Camera
            Camera used to render the bounding box.

        line_thickness: int, optional
            Thickness of bounding box lines. Default: 2.

        font_scale: float, optional
            Font scale used in text labels. Default: 0.5.

        Raises
        ------
        ValueError
            Raised if image is not a 3-channel uint8 numpy array.
        TypeError
            Raised if camera is not an instance of Camera.
        """
        if (
            not isinstance(image, np.ndarray) or image.dtype != np.uint8 or len(image.shape) != 3 or image.shape[2] != 3
        ):
            raise ValueError('`image` needs to be a 3-channel uint8 numpy array')
        if not isinstance(camera, Camera):
            raise TypeError('`camera` should be of type Camera')
        for box in self.boxlist:
            box.render(
                image,
                camera,
                line_thickness=line_thickness,
                class_name=self._ontology.contiguous_id_to_name[box.class_id],
                font_scale=font_scale
            )

    @property
    def poses(self):
        """Get poses for bounding boxes in annotation."""
        return [box.pose for box in self.boxlist]

    @property
    def sizes(self):
        return np.float32([box.sizes for box in self.boxlist])

    @property
    def class_ids(self):
        """Return class ID for each box, with ontology applied:
        0 is background, class IDs mapped to a contiguous set.
        """
        return np.int64([box.class_id for box in self.boxlist])

    @property
    def attributes(self):
        """Return a list of dictionaries of attribute name to value."""
        return [box.attributes for box in self.boxlist]

    @property
    def instance_ids(self):
        return np.int64([box.instance_id for box in self.boxlist])

    @property
    def hexdigest(self):
        """Reproducible hash of annotation."""
        return generate_uid_from_pbobject(self.to_proto())

    def project(self, camera):
        """Project bounding boxes into a camera and get back 2D bounding boxes in the frustum.

        Parameters
        ----------
        camera: Camera
            The Camera instance to project into.

        Raises
        ------
        NotImplementedError
            Unconditionally.
        """
        raise NotImplementedError
