# Copyright 2021 Toyota Research Institute.  All rights reserved.
from collections import OrderedDict

import numpy as np

from dgp.annotations import (
    BoundingBox2DAnnotationList,
    BoundingBox3DAnnotationList,
    PanopticSegmentation2DAnnotation,
)
from dgp.annotations.transforms import BaseTransform
from dgp.utils.structures import BoundingBox2D


class InstanceMaskVisibilityFilter(BaseTransform):
    """Given a multi-modal camera data, select instances whose instance masks appear big enough *at least in one camera*.

    For example, even when an object is mostly truncated in one camera, if it looks big enough in a neighboring
    camera in the multi-modal sample, it will be included in the annotations. In the transformed dataset item, all detection
    annotations (i.e. `bounding_box_3d`, `bounding_box_2d`, `instance_segmentation_2d') contain a single set of instances.

    Parameters
    ----------
    camera_datum_names: list[str]
        Names of camera datums to be used in visibility computation.
        The definition of "visible" is that an instance has large mask at least in one of these cameras.

    min_mask_size: int, default: 300
        Minimum number of foreground pixels in instance mask for an instance to be added to annotations.

    use_amodal_bbox2d_annotations: bool, default: False
        If True, then use "amodal" bounding box (i.e. the box includes occluded/truncated parts) for 2D bounding box annotation.
        If False, then use "modal" bounding box (i.e. tight bounding box of instance mask.)
    """
    def __init__(self, camera_datum_names, min_mask_size=300, use_amodal_bbox2d_annotations=False):
        self._camera_datum_names = camera_datum_names
        self._min_mask_size = min_mask_size
        self._use_amodal_bbox2d_annotations = use_amodal_bbox2d_annotations

    def transform_sample(self, sample):
        """Main entry point for filtering a multimodal sample using instance masks.

        Parameters
        ----------
        sample: list[OrderedDict]
            Multimodal sample as returned by `__getitem__()` of `_SynchronizedDataset`.

        Returns
        -------
        new_sample: list[OrderedDict]
            Multimodal sample with all detection annotations are filtered.

        Raises
        ------
        ValueError
            Raised if a 2D or 3D bounding box instance lacks any required instance IDs.
        """
        cam_datums = [datum for datum in sample if datum['datum_name'] in self._camera_datum_names]

        visible_instance_ids = set()  # Instances that looks big enough in any camera.
        in_frustum_instance_ids_per_camera = {}  # In-frustum instances for each camera.
        id_to_bbox3d_per_camera, id_to_mask2d_per_camera, id_to_bbox2d_per_camera = {}, {}, {}
        for datum in cam_datums:
            datum_name = datum['datum_name']

            # What instances are (partially) within camera frustum?
            in_frustum_instance_ids_per_camera[datum_name] = [
                mask.instance_id for mask in datum['instance_segmentation_2d']
            ]

            # Map instance ID to annotations (e.g. bounding boxes, masks).
            id_to_bbox3d = {bbox3d.instance_id: bbox3d for bbox3d in datum['bounding_box_3d']}
            id_to_mask2d = {mask2d.instance_id: mask2d for mask2d in datum['instance_segmentation_2d']}
            id_to_bbox3d_per_camera[datum_name] = id_to_bbox3d
            id_to_mask2d_per_camera[datum_name] = id_to_mask2d

            if self._use_amodal_bbox2d_annotations:
                id_to_bbox2d = {bbox2d.instance_id: bbox2d for bbox2d in datum['bounding_box_2d']}
                id_to_bbox2d_per_camera[datum_name] = id_to_bbox2d

            # TODO: Remove this filtering, once the ontology is unified between 2d and 3d in the upcoming PD data.
            if self._use_amodal_bbox2d_annotations:
                in_frustum_instance_ids_per_camera[datum_name] = [
                    _id for _id in in_frustum_instance_ids_per_camera[datum_name]
                    if _id in id_to_bbox2d and _id in id_to_bbox3d
                ]
            else:
                in_frustum_instance_ids_per_camera[datum_name] = [
                    _id for _id in in_frustum_instance_ids_per_camera[datum_name] if _id in id_to_bbox3d
                ]
            ids_missing_in_bbox3d = list(set(in_frustum_instance_ids_per_camera[datum_name]) - set(id_to_bbox3d))
            if ids_missing_in_bbox3d:
                raise ValueError(
                    "Missing instances from `bounding_box_3d`: {:s}".format(', '.join(sorted(ids_missing_in_bbox3d)))
                )
            if self._use_amodal_bbox2d_annotations:
                ids_missing_in_bbox2d = list(set(in_frustum_instance_ids_per_camera[datum_name]) - set(id_to_bbox2d))
                if ids_missing_in_bbox2d:
                    raise ValueError(
                        "Missing instances from `bounding_box_2d`: {:s}".format(
                            ', '.join(sorted(ids_missing_in_bbox2d))
                        )
                    )

            for instance_mask in datum['instance_segmentation_2d']:
                if instance_mask.area >= self._min_mask_size:
                    visible_instance_ids.add(instance_mask.instance_id)

        # For each camera, create new annotation and replace the original one.
        new_sample = sample
        for datum in new_sample:
            datum_name = datum['datum_name']
            if datum_name not in self._camera_datum_names:
                continue

            # if instance is within frustum and it looks big enough from any (neigboring) camera, then include its annotations.
            new_boxlist_3d, new_boxlist_2d, new_masklist_2d = [], [], []
            for instance_id in in_frustum_instance_ids_per_camera[datum_name]:
                if instance_id in visible_instance_ids:
                    new_boxlist_3d.append(id_to_bbox3d_per_camera[datum_name][instance_id])

                    mask2d = id_to_mask2d_per_camera[datum_name][instance_id]
                    if self._use_amodal_bbox2d_annotations:
                        # use physical groundtruth.
                        bbox2d = id_to_bbox2d_per_camera[datum_name][instance_id]
                    else:
                        # use tight bounding box of instance mask.
                        yy, xx = mask2d.bitmask.nonzero()
                        y1, y2 = np.min(yy), np.max(yy)
                        x1, x2 = np.min(xx), np.max(xx)
                        bbox2d = BoundingBox2D(
                            box=np.float32([x1, y1, x2, y2]),
                            class_id=mask2d.class_id,
                            instance_id=mask2d.instance_id,
                            # color=mask2d.color, # TODO: Add color property to Mask2D, and uncomment this.
                            attributes=mask2d.attributes,
                            mode="ltrb",
                        )
                    new_boxlist_2d.append(bbox2d)
                    new_masklist_2d.append(mask2d)

            # Replace annotations in place.
            datum['bounding_box_3d'] = BoundingBox3DAnnotationList(datum['bounding_box_3d'].ontology, new_boxlist_3d)
            datum['bounding_box_2d'] = BoundingBox2DAnnotationList(datum['bounding_box_2d'].ontology, new_boxlist_2d)
            datum['instance_segmentation_2d'] = PanopticSegmentation2DAnnotation.from_masklist(
                new_masklist_2d,
                datum['instance_segmentation_2d'].ontology,
                mask_shape=(datum['rgb'].height, datum['rgb'].width)
            )

        return new_sample

    def transform_datum(self, datum):
        """Main entry point for filtering a single-modal datum using instance masks.

        Parameters
        ----------
        datum: OrderedDict
            Single-modal datum as returned by `__getitem__()` of `_FrameDataset`.

        Returns
        -------
        new_datum: OrderedDict
            Single-modal sample with all detection annotations are filtered.
        """
        return self.transform_sample([datum])[0]


class BoundingBox3DCoalescer(BaseTransform):
    """Coalesce 3D bounding box annotation from multiple datums and use it as an annotation of target datum.
    The bounding boxes are brought into the target datum frame.

    Parameters
    ----------
    src_datum_names: list[str]
        List of datum names used to create a list of coalesced bounding boxes.

    dst_datum_name: str
        Datum whose `bounding_box_3d` is replaced by the coelesced bounding boxes.

    drop_src_datums: bool, default: True
        If True, then remove the source datums in the transformed sample.
    """
    def __init__(self, src_datum_names, dst_datum_name, drop_src_datums=True):
        self._src_datum_names = src_datum_names
        self._dst_datum_name = dst_datum_name
        self._drop_src_datums = drop_src_datums

    def transform_sample(self, sample):
        """Main entry point for coalescing 3D bounding boxes.

        Parameters
        ----------
        sample: list[OrderedDict]
            Multimodal sample as returned by `__getitem__()` of `_SynchronizedDataset`.

        Returns
        -------
        new_sample: list[OrderedDict]
            Multimodal sample with updated 3D bounding box annotations.

        Raises
        ------
        ValueError
            Raised if there are multiple instances of the same kind of datum in a sample.
        """
        # Mapping index to datum. The order of datums is preserved in output.
        datums, src_datum_inds, dst_datum_ind = OrderedDict(), [], []
        for idx, datum in enumerate(sample):
            if datum['datum_name'] in self._src_datum_names:
                src_datum_inds.append(idx)
            elif datum['datum_name'] == self._dst_datum_name:
                dst_datum_ind.append(idx)
            datums[idx] = datum
        if len(dst_datum_ind) != 1:
            raise ValueError("There must be one {:s} datum.".format(self._dst_datum_name))
        dst_datum_ind = dst_datum_ind[0]

        # Merge 3D bounding boxes, bringing them into the destination frame.
        bbox_3d_V_merged, instance_ids_merged = [], []
        dst_datum = datums[dst_datum_ind]
        for idx in src_datum_inds:
            src_datum = datums[idx]
            p_src_dst = dst_datum['pose'].inverse() * src_datum['pose']
            for bbox_3d in src_datum['bounding_box_3d']:
                # Keep only the unique instance IDs
                if bbox_3d.instance_id not in instance_ids_merged:
                    instance_ids_merged.append(bbox_3d.instance_id)
                    bbox_3d_V_merged.append(p_src_dst * bbox_3d)
        ontology = dst_datum['bounding_box_3d'].ontology  # Assumption: ontology is shared.
        coalesced_bbox3d_annotation = BoundingBox3DAnnotationList(ontology, bbox_3d_V_merged)
        dst_datum['bounding_box_3d'] = coalesced_bbox3d_annotation

        transformed_sample = []
        for idx, datum in enumerate(sample):
            # Source datums
            if idx in src_datum_inds:
                if not self._drop_src_datums:
                    transformed_sample.append(datum)
            # destination datum
            elif idx == dst_datum_ind:
                transformed_sample.append(dst_datum)
            else:  # other datums
                transformed_sample.append(datum)
        return transformed_sample
