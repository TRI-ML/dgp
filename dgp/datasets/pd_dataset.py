# Copyright 2020 Toyota Research Institute.  All rights reserved.
import logging
from collections import OrderedDict

import numpy as np

from dgp.datasets.base_dataset import BaseSceneDataset, DatasetMetadata
from dgp.datasets.synchronized_dataset import _SynchronizedDataset
from dgp.utils.geometry import Pose

POINT_CLOUD_KEY = 'point_cloud'
COALESCED_LIDAR_DATUM_NAME = 'lidar'
LIDAR_DATUM_NAMES = ["lidar_02", "lidar_03", "lidar_04", "lidar_05", "lidar_11", "lidar_12", "lidar_13", "lidar_14"]


class _ParallelDomainDataset(_SynchronizedDataset):
    """Dataset for PD data. Works just like normal SynchronizedSceneDataset,
    with special keyword datum name "lidar". When this datum is requested,
    all lidars are coalesced into a single "lidar" datum.
    """
    def __init__(self, *args, **kwargs):
        self.coalesce_point_cloud = True if COALESCED_LIDAR_DATUM_NAME in kwargs["datum_names"] else False

        # Determine the index of 'lidar' so that we provide the OrderedDict in
        # the right place.
        if self.coalesce_point_cloud:
            self._datum_name_to_index = {
                datum_name: datum_idx
                for datum_idx, datum_name in enumerate(kwargs['datum_names'])
            }
            # Insert all other datum_names first before adding lidar datums
            new_datum_names = [
                datum_name for datum_name in kwargs['datum_names'] if datum_name != COALESCED_LIDAR_DATUM_NAME
            ]
            new_datum_names.extend(LIDAR_DATUM_NAMES)

            # Update datum_names with the full set of lidar datums, if requested.
            logging.info('Datum names with lidar datums={}'.format(new_datum_names))
            kwargs['datum_names'] = new_datum_names

        super().__init__(*args, **kwargs)

    def coalesce_pc_data(self, items):
        """Combine set of point cloud datums into a single point cloud.

        Parameters
        ----------
        items: list
            List of OrderedDict, containing parsed point cloud or image data.

        Returns
        -------
        coalesced_pc: OrderedDict
           OrderedDict containing coalesced point cloud and associated metadata.
        """
        pc_items = [item for item in items if POINT_CLOUD_KEY in item]
        assert self.coalesce_point_cloud
        assert len(pc_items) == len(LIDAR_DATUM_NAMES)

        # Only coalesce if there's more than 1 point cloud
        coalesced_pc = OrderedDict()
        X_V_merged, bbox_3d_V_merged, class_ids_merged, instance_ids_merged = [], [], [], []
        for item in pc_items:
            X_S = item[POINT_CLOUD_KEY]
            p_VS = item['extrinsics']
            X_V_merged.append(p_VS * X_S)
            if 'bounding_box_3d' in item:
                bbox_3d_V_merged.extend([p_VS * bbox_3d for bbox_3d in item['bounding_box_3d']])
                class_ids_merged.append(item['class_ids'])
                instance_ids_merged.append(item['instance_ids'])

        coalesced_pc['datum_name'] = COALESCED_LIDAR_DATUM_NAME
        coalesced_pc['timestamp'] = pc_items[0]['timestamp']
        coalesced_pc[POINT_CLOUD_KEY] = np.vstack(X_V_merged)
        coalesced_pc['extra_channels'] = np.vstack([item['extra_channels'] for item in pc_items])
        # Note: Pose for the coalesced point cloud is identical to pose_LV
        # Note: Extrinsics for this "virtual" datum is identity since the point cloud is defined in the (V)ehicle frame of reference.
        coalesced_pc['extrinsics'] = Pose()
        p_LS = pc_items[0]['pose']
        p_VS = pc_items[0]['extrinsics']
        p_LV = p_LS * p_VS.inverse()
        coalesced_pc['pose'] = p_LV

        if len(bbox_3d_V_merged):
            # Add bounding boxes and metadata in the vehicle frame
            coalesced_pc['bounding_box_3d'] = bbox_3d_V_merged
            coalesced_pc['class_ids'] = np.hstack(class_ids_merged)
            coalesced_pc['instance_ids'] = np.hstack(instance_ids_merged)
            # Keep only the unique instance IDs
            coalesced_pc['instance_ids'], unique_idx = np.unique(coalesced_pc['instance_ids'], return_index=True)
            coalesced_pc['bounding_box_3d'] = [
                bbox_3d for idx, bbox_3d in enumerate(coalesced_pc['bounding_box_3d']) if idx in unique_idx
            ]
            coalesced_pc['class_ids'] = np.array([
                bbox_3d for idx, bbox_3d in enumerate(coalesced_pc['class_ids']) if idx in unique_idx
            ])

        assert len(coalesced_pc['bounding_box_3d']) == len(coalesced_pc['class_ids'])
        assert len(coalesced_pc['bounding_box_3d']) == len(coalesced_pc['instance_ids'])

        return coalesced_pc

    def coalesce_sample(self, sample):
        """Coalesce a point cloud for a single sample"""
        # First coalesce the point cloud item and assign at the right index.
        items_dict = OrderedDict()
        items_dict[self._datum_name_to_index[COALESCED_LIDAR_DATUM_NAME]] = self.coalesce_pc_data(sample)
        # Fill in the rest of the items.
        items_dict.update({
            self._datum_name_to_index[item['datum_name']]: item
            for item in sample
            if POINT_CLOUD_KEY not in item
        })

        # Return the items list in the correct order.
        indices_and_items_sorted = sorted(list(items_dict.items()), key=lambda tup: tup[0])
        aligned_items = list(map(lambda tup: tup[1], indices_and_items_sorted))
        return aligned_items

    def __getitem__(self, idx):
        # Get all the items requested.
        sample_data = super().__getitem__(idx)

        if self.coalesce_point_cloud:
            if self.forward_context > 0 or self.backward_context > 0:
                # Coalesce for each timestamp
                sample_data = [self.coalesce_sample(t_item) for t_item in sample_data]
            else:
                sample_data = self.coalesce_sample(sample_data)
        return sample_data


class ParallelDomainSceneDataset(_ParallelDomainDataset):
    """
    Refer to SynchronizedSceneDataset for parameters.
    """
    def __init__(
        self,
        scene_dataset_json,
        split='train',
        datum_names=None,
        requested_annotations=None,
        requested_autolabels=None,
        backward_context=0,
        forward_context=0,
        generate_depth_from_datum=None,
        only_annotated_datums=False,
    ):
        # Extract all scenes from the scene dataset JSON for the appropriate split
        scenes, calibration_table = BaseSceneDataset._extract_scenes_from_scene_dataset_json(
            scene_dataset_json, split=split, requested_autolabels=requested_autolabels
        )

        # Return SynchronizedDataset with scenes built from dataset.json
        dataset_metadata = DatasetMetadata.from_scene_containers(scenes)
        super().__init__(
            dataset_metadata,
            scenes=scenes,
            calibration_table=calibration_table,
            datum_names=datum_names,
            requested_annotations=requested_annotations,
            requested_autolabels=requested_autolabels,
            backward_context=backward_context,
            forward_context=forward_context,
            generate_depth_from_datum=generate_depth_from_datum,
            is_scene_dataset=True,
            only_annotated_datums=only_annotated_datums
        )


class ParallelDomainScene(_ParallelDomainDataset):
    """
    Refer to SynchronizedScene for parameters.
    """
    def __init__(
        self,
        scene_json,
        datum_names=None,
        requested_annotations=None,
        requested_autolabels=None,
        backward_context=0,
        forward_context=0,
        generate_depth_from_datum=None,
        only_annotated_datums=False
    ):

        # Extract a single scene from the scene JSON
        scene, calibration_table = BaseSceneDataset._extract_scene_from_scene_json(
            scene_json, requested_autolabels=requested_autolabels
        )

        # Return SynchronizedDataset with scenes built from dataset.json
        dataset_metadata = DatasetMetadata.from_scene_containers([scene])
        super().__init__(
            dataset_metadata,
            scenes=[scene],
            calibration_table=calibration_table,
            datum_names=datum_names,
            requested_annotations=requested_annotations,
            requested_autolabels=requested_autolabels,
            backward_context=backward_context,
            forward_context=forward_context,
            generate_depth_from_datum=generate_depth_from_datum,
            is_scene_dataset=True,
            only_annotated_datums=only_annotated_datums
        )
