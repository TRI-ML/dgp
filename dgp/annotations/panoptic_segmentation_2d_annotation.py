# Copyright 2021 Toyota Research Institute.  All rights reserved.
import json
import locale
import os

import cv2
import numpy as np

from dgp import INSTANCE_SEGMENTATION_2D_FOLDER
from dgp.annotations.base_annotation import Annotation
from dgp.annotations.ontology import BoundingBoxOntology
from dgp.proto import annotations_pb2
from dgp.utils.dataset_conversion import (
    generate_uid_from_instance_segmentation_2d_annotation,
)
from dgp.utils.structures import InstanceMask2D


class PanopticSegmentation2DAnnotation(Annotation):
    """Container for 2D panoptic segmentation annotations

    Parameters
    ----------
    ontology: dgp.annotations.BoundingBoxOntology
        Bounding box ontology that will be used to load annotations

    panoptic_image: np.uint16 array
        Single-channel image with value at [i, j] corresponding to the instance ID of the object the pixel belongs to
        for thing pixels and the class ID for stuff pixels. Shape (H, W)

    index_to_label: dict[str->Union(dict, List[dict])]
        Maps from each class name to either:
            1. If class is stuff:
                a single dict - there is only *one* (potentially empty) segment in an image for each stuff class,
                with fields:
                    'index': int
                    'attributes': dict
            2. If class is thing:
                a list of such dicts, one for each instance of the thing class

        For example, an entry in annotation['index_to_label']['Car'], which is a *list*, can look like:
            {
                'index': 21,
                'attributes': {
                    'EmergencyVehicle': 'No',
                    'IsTowing': 'No'
                }
            }
        Then if we load the image at `image_uri`, we would expect all pixels with value 21 to belong to
        this one instance of the 'Car' class.

    panoptic_image_dtype: type, default: np.uint16
        Numpy data type (e.g. np.uint16, np.uint32, etc.) of panoptic image.

    Notes
    -----
    For now only using this annotation object for instance segmentation, so a BoundingBoxOntology is sufficient

    In the future, we probably want to wrap a panoptic annotation into a
    PanopticSegmentation2DAnnotationPB(panoptic_image=<image_path>, index_to_label=<json_path>) proto message
    and then we can `.load` from this proto (and serialize to it in `.save`).

    For now we simply assume, by convention, that a JSON index_to_label file exists along with the panoptic_image file,
    and in this way stay a bit more flexible about what the PanopticSegmentation2DAnnotationPB object should
    look like (e.g. if index_to_label is defined as a proto message).
    """
    DEFAULT_PANOPTIC_IMAGE_DTYPE = np.uint16

    def __init__(self, ontology, panoptic_image, index_to_label, panoptic_image_dtype=DEFAULT_PANOPTIC_IMAGE_DTYPE):
        if panoptic_image_dtype not in (np.uint16, np.uint32):
            raise ValueError("'panoptic_image_dtype' must be either np.uint16 or np.uint32.")
        super().__init__(ontology)
        if not isinstance(self._ontology, BoundingBoxOntology):
            raise TypeError('Trying to load annotation with wrong type of ontology!')
        if (
            not isinstance(panoptic_image, np.ndarray) or not panoptic_image.dtype == panoptic_image_dtype
            or len(panoptic_image.shape) != 2
        ):
            raise ValueError('`panoptic_image` needs to be a single-channel uint16 numpy array')

        # TODO: check type and structure of `index_to_label`

        self.panoptic_image = panoptic_image
        self.index_to_label = index_to_label
        self._masklist = self.parse_panoptic_image()
        self._panoptic_image_dtype = panoptic_image_dtype

    @property
    def masklist(self):
        return self._masklist

    @property
    def panoptic_image_dtype(self):
        return self._panoptic_image_dtype

    def parse_panoptic_image(self):
        """Parses `self.panoptic_image` to produce instance_masks, class_names, and instance_ids

        Returns
        -------
        instance_masks: list[InstanceMask2D]
            Instance mask for each instance in panoptic annotation.

        Raises
        ------
        ValueError
            Raised if an instance ID, parsed from a label, is negative.
        """
        instance_masks = []
        for class_name, labels in self.index_to_label.items():
            if isinstance(labels, list):
                for label in labels:

                    # We use the ID provided by Scale directly as the instance_id. We require this to
                    # be non-negative as otherwise it breaks our convention that only stuff pixels in
                    # `instance_image` are 0
                    instance_id = label['index']
                    if instance_id < 0:
                        raise ValueError('`index` field of a thing class is expected to be non-negative')

                    # Mask for pixels belonging to this instance
                    color = self.ontology.colormap[self.ontology.name_to_id[class_name]]
                    instance_mask = InstanceMask2D(
                        self.panoptic_image == instance_id,
                        class_id=self.ontology.name_to_contiguous_id[class_name],
                        instance_id=instance_id,
                        color=color,
                        attributes=label['attributes']
                    )
                    instance_masks.append(instance_mask)

        return instance_masks

    @property
    def instances(self):
        """
        Returns
        -------
        np.ndarray:
            (N, H, W) bool array for each instance in panoptic annotation.
            N is the number of instances; H, W are the height and width of the image.
        """
        if self.masklist:
            return np.stack([mask.bitmask for mask in self.masklist], axis=0)
        else:
            return np.array([], self._panoptic_image_dtype)

    @property
    def class_names(self):
        """
        Returns
        -------
        List[str]
            Class name for each instance in panoptic annotation
        """
        return [self.ontology.contiguous_id_to_name[class_id] for class_id in self.class_ids]

    @property
    def class_ids(self):
        """
        Returns
        -------
        List[int]
            Contiguous class ID for each instance in panoptic annotation
        """
        return [mask.class_id for mask in self.masklist]

    @property
    def instance_ids(self):
        """
        Returns
        -------
        instance_ids: List[int]
            Instance IDs for each instance in panoptic annotation
        """
        return [mask.instance_id for mask in self.masklist]

    @classmethod
    def load(cls, annotation_file, ontology, panoptic_image_dtype=DEFAULT_PANOPTIC_IMAGE_DTYPE):
        """Loads annotation from file into a canonical format for consumption in __getitem__ function in BaseDataset.
        Format/data structure for annotations will vary based on task.

        Parameters
        ----------
        annotation_file: str
            Full path to panoptic image. `index_to_label` JSON is expected to live at the same path with '.json' ending

        ontology: Ontology
            Ontology for given annotation

        panoptic_image_dtype: type, optional
            Numpy data type (e.g. np.uint16, np.uint32, etc) of panoptic image. Default: np.uint16.
        """
        panoptic_image = cv2.imread(annotation_file, cv2.IMREAD_UNCHANGED)
        if len(panoptic_image.shape) == 3:
            # ParallelDomain uses RGB image for now.
            # TODO: discuss with PD on changing this to np.uint16 image.
            _L = panoptic_image
            label_map = _L[:, :, 2] + 256 * _L[:, :, 1] + 256 * 256 * _L[:, :, 0]
            panoptic_image = label_map.astype(panoptic_image_dtype)
        with open('{}.json'.format(os.path.splitext(annotation_file)[0]), encoding=locale.getpreferredencoding()) as _f:
            index_to_label = json.load(_f)
        return cls(ontology, panoptic_image, index_to_label, panoptic_image_dtype)

    @classmethod
    def from_masklist(cls, masklist, ontology, mask_shape=None, panoptic_image_dtype=DEFAULT_PANOPTIC_IMAGE_DTYPE):
        """Instantiate PanopticSegmentation2DAnnotation from a list of `InstanceMask2D`.

        CAVEAT: This constructs *instance segmentation* annotation, not panoptic annotation. In the following example,
        ```
        annotation_1 = PanopticSegmentation2DAnnotation.load(PANOPTIC_LABEL_IMAGE, ontology)
        annotation_2 = PanopticSegmentation2DAnnotation.from_masklist(annotation_1.masklist, ontology)
        ```
        - all pixels of "stuff" classes in `annotation_1.panoptic_image` are replaced with `ontology.VOID_ID`
          in annotation_2.panoptic_image, and
        - all "stuff" classes in `annotation_1.index_to_label` are removed in `annotation_2.index_to_label`.

        Parameters
        ----------
        masklist: list[InstanceMask2D]
            Instance masks used to create an annotation object.

        ontology: dgp.annotations.BoundingBoxOntology
            Bounding box ontology used to load annotations.

        mask_shape: list[int]
            Height and width of the mask. Only used to create an empty panoptic image when masklist is empty.

        panoptic_image_dtype: type, optional
            Numpy data type (e.g. np.uint16, np.uint32, etc) of panoptic image. Default: np.uint16.
        """
        if not masklist:
            panoptic_image = np.ones(mask_shape, panoptic_image_dtype) * ontology.VOID_ID
        else:
            panoptic_image = np.ones_like(masklist[0].bitmask, panoptic_image_dtype) * ontology.VOID_ID
        index_to_label = {class_name: [] for class_name in ontology.class_names}
        for instance_mask in masklist:
            assert np.unique(panoptic_image[instance_mask.bitmask]) == [ontology.VOID_ID], \
                    "No overlapping between instance masks is allowed."
            panoptic_image[instance_mask.bitmask] = instance_mask.instance_id
            index_to_label[ontology.contiguous_id_to_name[instance_mask.class_id]].append({
                'index':
                instance_mask.instance_id,
                'attributes':
                instance_mask.attributes
            })
        # sort index list for each label.
        for v in index_to_label.values():
            v.sort(key=lambda x: x['index'])
        return cls(ontology, panoptic_image, index_to_label, panoptic_image_dtype)

    def render(self):
        """TODO: Return a rendering of the annotation"""
        raise NotImplementedError

    @property
    def hexdigest(self):
        """Reproducible hash of annotation."""
        # NOTE: for now just hashing `self.panoptic_image`. Could have a hash
        # that's a combination of this and `self.index_to_label`
        return generate_uid_from_instance_segmentation_2d_annotation(self.panoptic_image)

    def save(self, save_dir, datum=None):
        """Serialize Annotation object and save into a  specified datum.

        Parameters
        ----------
        save_dir: str
            If `datum` is given, then annotations will be saved to <save_dir>/<datum.id.name>/<hexdigest>.{png,json}.
            Otherwise, annotations will be saved to <save_dir>/<hexdigest>.{png,json}.

        datum: dgp.proto.sample_pb2.Datum
            Datum to which we will append annotation

        Returns
        -------
        panoptic_image_path: str
            Full path to the output panoptic image file.
        """
        if datum is None:
            panoptic_image_path = os.path.join(save_dir, '{}.png'.format(self.hexdigest))
        else:
            panoptic_image_filename = os.path.join(
                INSTANCE_SEGMENTATION_2D_FOLDER, datum.id.name, '{}.png'.format(self.hexdigest)
            )
            # NOTE: path is to `panoptic_image` (convention is that a JSON file with same name also exists)
            datum.datum.image.annotations[annotations_pb2.INSTANCE_SEGMENTATION_2D] = panoptic_image_filename
            panoptic_image_path = os.path.join(save_dir, panoptic_image_filename)
        os.makedirs(os.path.dirname(panoptic_image_path), exist_ok=True)
        cv2.imwrite(panoptic_image_path, self.panoptic_image)

        index_to_label_path = '{}.json'.format(os.path.splitext(panoptic_image_path)[0])
        with open(index_to_label_path, 'w', encoding=locale.getpreferredencoding()) as _f:
            json.dump(self.index_to_label, _f)

        return panoptic_image_path

    def __len__(self):
        return len(self.masklist)

    def __getitem__(self, index):
        """Return a single 2D instance mask."""
        return self.masklist[index]
