# Copyright 2021 Toyota Research Institute.  All rights reserved.
import io
import os

import cv2
import numpy as np

from dgp.annotations.base_annotation import Annotation
from dgp.annotations.ontology import SemanticSegmentationOntology
from dgp.utils.dataset_conversion import (
    generate_uid_from_semantic_segmentation_2d_annotation,
)


class SemanticSegmentation2DAnnotation(Annotation):
    """Container for semantic segmentation annotation.

    Parameters
    ----------
    ontology: SemanticSegmentationOntology
        Ontology for semantic segmentation tasks.

    segmentation_image: np.array
        Numpy uint8 array encoding segmentation labels.
    """
    def __init__(self, ontology, segmentation_image):
        super().__init__(ontology)
        assert isinstance(
            self._ontology, SemanticSegmentationOntology
        ), "Trying to load annotation with wrong type of ontology!"
        assert isinstance(segmentation_image, np.ndarray)
        assert segmentation_image.dtype == np.uint8
        assert len(segmentation_image.shape) == 2
        self._segmentation_image = segmentation_image

    @classmethod
    def load(cls, annotation_file, ontology):
        """Load annotation from annotation file and ontology.

        Parameters
        ----------
        annotation_file: str or bytes
            Full path to annotation or bytestring

        ontology: SemanticSegmentationOntology
            Ontology for semantic segmentation tasks.

        Returns
        -------
        SemanticSegmentation2DAnnotation
            Annotation object instantiated from file.
        """
        if isinstance(annotation_file, bytes):
            raw_bytes = io.BytesIO(annotation_file)
            segmentation_image = cv2.imdecode(np.frombuffer(raw_bytes.getbuffer(), np.uint8), cv2.IMREAD_UNCHANGED)
        else:
            segmentation_image = cv2.imread(annotation_file, cv2.IMREAD_UNCHANGED)

        if len(segmentation_image.shape) == 3:
            # ParllelDomain used RGBA image, and uses only R-channel.
            # TODO: discuss with PD on changing this to single-channel np.uint8 image.
            segmentation_image = segmentation_image[:, :, 2].copy(order='C')
        # Pixels of value VOID_ID are not remapped to a label.
        not_ignore = segmentation_image != ontology.VOID_ID
        segmentation_image[not_ignore] = ontology.label_lookup[segmentation_image[not_ignore]]
        return cls(ontology, segmentation_image)

    def _convert_contiguous_to_class(self):
        """Helper function to run pre processing prior to saving

        Returns
        -------
        segmentation_image: np.array
            A copy of self._segmentation_image with contiguous_id mapped back to class_id for saving
        """
        # Convert the segmentation image back to original class IDs
        reverse_label_lookup = np.ones(self.ontology.VOID_ID + 1, dtype=np.uint8) * self.ontology.VOID_ID
        for contiguous_id, class_id in self.ontology.contiguous_id_to_class_id.items():
            reverse_label_lookup[contiguous_id] = class_id

        # Create a copy and map IDs back to original set
        segmentation_image = np.copy(self._segmentation_image)
        not_ignore = segmentation_image != self.ontology.VOID_ID
        segmentation_image[not_ignore] = reverse_label_lookup[segmentation_image[not_ignore]]
        return segmentation_image

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
        segmentation_image = self._convert_contiguous_to_class()

        # Save the image as PNG
        output_annotation_file = os.path.join(
            save_dir, f"{generate_uid_from_semantic_segmentation_2d_annotation(segmentation_image)}.png"
        )
        cv2.imwrite(output_annotation_file, segmentation_image)
        return output_annotation_file

    def render(self):
        """TODO: Rendering function for semantic segmentation images."""
        raise NotImplementedError

    @property
    def label(self):
        return self._segmentation_image

    @property
    def hexdigest(self):
        """Reproducible hash of annotation."""
        return generate_uid_from_semantic_segmentation_2d_annotation(self._segmentation_image)
