# Copyright 2020 Toyota Research Institute.  All rights reserved.
import logging
from collections import OrderedDict

import numpy as np

from dgp.annotations.bounding_box_3d_annotation import (
    BoundingBox3DAnnotationList,
)
# pylint: disable=W0611
from dgp.contribs.pd.metadata_pb2 import ParallelDomainSceneMetadata
# pylint: enable=W0611
from dgp.datasets.base_dataset import BaseDataset, DatasetMetadata
from dgp.datasets.synchronized_dataset import _SynchronizedDataset
from dgp.utils.pose import Pose

POINT_CLOUD_KEY = 'point_cloud'
COALESCED_LIDAR_DATUM_NAME = 'lidar'
LIDAR_DATUM_NAMES = ["lidar_02", "lidar_03", "lidar_04", "lidar_05", "lidar_11", "lidar_12", "lidar_13", "lidar_14"]
VIRTUAL_CAMERA_DATUM_NAMES = [
    "virtual_lidar_02_camera_0",
    "virtual_lidar_02_camera_1",
    "virtual_lidar_02_camera_2",
    "virtual_lidar_03_camera_0",
    "virtual_lidar_03_camera_1",
    "virtual_lidar_03_camera_2",
    "virtual_lidar_04_camera_0",
    "virtual_lidar_04_camera_1",
    "virtual_lidar_04_camera_2",
    "virtual_lidar_05_camera_0",
    "virtual_lidar_05_camera_1",
    "virtual_lidar_05_camera_2",
    "virtual_lidar_11_camera_0",
    "virtual_lidar_12_camera_0",
    "virtual_lidar_13_camera_0",
    "virtual_lidar_14_camera_0",
]


class _ParallelDomainDataset(_SynchronizedDataset):
    """Dataset for PD data. Works just like normal SynchronizedSceneDataset,
    with special keyword datum name "lidar". When this datum is requested,
    all lidars are coalesced into a single "lidar" datum.

    Parameters
    ----------
    dataset_metadata: DatasetMetadata
        Dataset metadata, populated from scene dataset JSON

    scenes: list[SceneContainer], default: None
        List of SceneContainers parsed from scene dataset JSON

    datum_names: list, default: None
        Select list of datum names for synchronization (see self.select_datums(datum_names)).

    requested_annotations: tuple, default: None
        Tuple of annotation types, i.e. ('bounding_box_2d', 'bounding_box_3d'). Should be equivalent
        to directory containing annotation from dataset root.

    requested_autolabels: tuple[str], default: None
        Tuple of annotation types similar to `requested_annotations`, but associated with a particular autolabeling model.
        Expected format is "<model_id>/<annotation_type>"

    forward_context: int, default: 0
        Forward context in frames [T+1, ..., T+forward]

    backward_context: int, default: 0
        Backward context in frames [T-backward, ..., T-1]

    generate_depth_from_datum: str, default: None
        Datum name of the point cloud. If is not None, then the depth map will be generated for the camera using
        the desired point cloud.

    only_annotated_datums: bool, default: False
        If True, only datums with annotations matching the requested annotation types are returned.

    use_virtual_camera_datums: bool, default: True
        If True, uses virtual camera datums. See dgp.datasets.pd_dataset.VIRTUAL_CAMERA_DATUM_NAMES for more details.

    accumulation_context: dict, default None
        Dictionary of datum names containing a tuple of (backward_context, forward_context) for sensor accumulation.
        For example, 'accumulation_context={'lidar':(3,1)} accumulates lidar points over the past three time steps and
        one forward step. Only valid for lidar and radar datums.

    transform_accumulated_box_points: bool, default: False
        Flag to use cuboid pose and instance id to warp points when using lidar accumulation.

    autolabel_root: str, default: None
        Path to autolabels.
    """
    def __init__(
        self,
        dataset_metadata,
        scenes=None,
        datum_names=None,
        requested_annotations=None,
        requested_autolabels=None,
        forward_context=0,
        backward_context=0,
        generate_depth_from_datum=None,
        only_annotated_datums=False,
        use_virtual_camera_datums=True,
        accumulation_context=None,
        transform_accumulated_box_points=False,
        autolabel_root=None,
    ):
        self.coalesce_point_cloud = datum_names is not None and \
                                    COALESCED_LIDAR_DATUM_NAME in datum_names
        self.use_virtual_camera_datums = use_virtual_camera_datums

        # Determine the index of 'lidar' so that we provide the OrderedDict in
        # the right place.
        if self.coalesce_point_cloud:
            self._datum_name_to_index = {datum_name: datum_idx for datum_idx, datum_name in enumerate(datum_names)}
            # Insert all other datum_names first before adding lidar datums
            new_datum_names = [datum_name for datum_name in datum_names if COALESCED_LIDAR_DATUM_NAME != datum_name]
            new_datum_names.extend(LIDAR_DATUM_NAMES)
            if use_virtual_camera_datums:
                new_datum_names.extend(VIRTUAL_CAMERA_DATUM_NAMES)

            # If we request the coalesced lidar with accumulation, update the accumulation for the
            # individual lidars
            if accumulation_context is not None and COALESCED_LIDAR_DATUM_NAME in accumulation_context:
                acc_context = accumulation_context.pop(COALESCED_LIDAR_DATUM_NAME)
                updated_acc = {datum_name: acc_context for datum_name in LIDAR_DATUM_NAMES}
                accumulation_context.update(updated_acc)

            # Update datum_names with the full set of lidar datums, if requested.
            logging.info('Datum names with lidar datums={}'.format(new_datum_names))
            datum_names = new_datum_names

        super().__init__(
            dataset_metadata=dataset_metadata,
            scenes=scenes,
            datum_names=datum_names,
            requested_annotations=requested_annotations,
            requested_autolabels=requested_autolabels,
            forward_context=forward_context,
            backward_context=backward_context,
            generate_depth_from_datum=generate_depth_from_datum,
            only_annotated_datums=only_annotated_datums,
            accumulation_context=accumulation_context,
            transform_accumulated_box_points=transform_accumulated_box_points,
            autolabel_root=autolabel_root,
        )

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

        # TODO: fix this
        if len(self.requested_autolabels) > 0:
            logging.warning(
                'autolabels were requested, however point cloud coalesce does not support coalescing autolabels'
            )

        # Only coalesce if there's more than 1 point cloud
        coalesced_pc = OrderedDict()
        X_V_merged, bbox_3d_V_merged, instance_ids_merged = [], [], []
        total_bounding_box_3d = 0
        for item in pc_items:
            X_S = item[POINT_CLOUD_KEY]
            p_VS = item['extrinsics']
            X_V_merged.append(p_VS * X_S)
            # Coalesce bounding_box_3d
            if 'bounding_box_3d' in item:
                total_bounding_box_3d += len(item['bounding_box_3d'])
                for bbox_3d in item['bounding_box_3d']:
                    # Keep only the unique instance IDs
                    if bbox_3d.instance_id not in instance_ids_merged:
                        instance_ids_merged.append(bbox_3d.instance_id)
                        bbox_3d_V_merged.append(p_VS * bbox_3d)
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
            ontology = pc_items[0]['bounding_box_3d'].ontology
            coalesced_pc['bounding_box_3d'] = BoundingBox3DAnnotationList(ontology, bbox_3d_V_merged)
        if 'bounding_box_3d' in coalesced_pc.keys():
            assert len(coalesced_pc['bounding_box_3d']) <= total_bounding_box_3d

        return coalesced_pc

    def coalesce_sample(self, sample):
        """Coalesce a point cloud for a single sample.

        Parameters
        ----------
        sample: list
            List of OrderedDict, containing parsed point cloud or image data.
        """
        # First coalesce the point cloud item and assign at the right index.
        items_dict = OrderedDict()
        items_dict[self._datum_name_to_index[COALESCED_LIDAR_DATUM_NAME]] = self.coalesce_pc_data(sample)
        # Fill in the rest of the items.
        items_dict.update({
            self._datum_name_to_index[item['datum_name']]: item
            for item in sample
            if POINT_CLOUD_KEY not in item and item['datum_name'] not in VIRTUAL_CAMERA_DATUM_NAMES
        })
        if self.use_virtual_camera_datums:
            # Append virtual camera datums.
            virtual_camera_datums = [item for item in sample if item['datum_name'] in VIRTUAL_CAMERA_DATUM_NAMES]
            virtual_camera_datums = {idx + len(items_dict): item for idx, item in enumerate(virtual_camera_datums)}
            items_dict.update(virtual_camera_datums)

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
                sample_data = [self.coalesce_sample(sample_data[0])]
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
        use_virtual_camera_datums=True,
        skip_missing_data=False,
        accumulation_context=None,
        dataset_root=None,
        transform_accumulated_box_points=False,
        use_diskcache=True,
        autolabel_root=None,
    ):
        if not use_diskcache:
            logging.warning('Instantiating a dataset with use_diskcache=False may exhaust memory with a large dataset.')

        # Extract all scenes from the scene dataset JSON for the appropriate split
        scenes = BaseDataset._extract_scenes_from_scene_dataset_json(
            scene_dataset_json,
            split,
            requested_autolabels,
            is_datums_synchronized=True,
            skip_missing_data=skip_missing_data,
            dataset_root=dataset_root,
            use_diskcache=use_diskcache,
            autolabel_root=autolabel_root,
        )

        # Return SynchronizedDataset with scenes built from dataset.json
        dataset_metadata = DatasetMetadata.from_scene_containers(
            scenes,
            requested_annotations,
            requested_autolabels,
            autolabel_root=autolabel_root,
        )
        super().__init__(
            dataset_metadata,
            scenes=scenes,
            datum_names=datum_names,
            requested_annotations=requested_annotations,
            requested_autolabels=requested_autolabels,
            backward_context=backward_context,
            forward_context=forward_context,
            generate_depth_from_datum=generate_depth_from_datum,
            only_annotated_datums=only_annotated_datums,
            use_virtual_camera_datums=use_virtual_camera_datums,
            accumulation_context=accumulation_context,
            transform_accumulated_box_points=transform_accumulated_box_points,
            autolabel_root=autolabel_root,
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
        only_annotated_datums=False,
        use_virtual_camera_datums=True,
        skip_missing_data=False,
        accumulation_context=None,
        transform_accumulated_box_points=False,
        use_diskcache=True,
        autolabel_root=None,
    ):
        if not use_diskcache:
            logging.warning('Instantiating a dataset with use_diskcache=False may exhaust memory with a large dataset.')

        # Extract a single scene from the scene JSON
        scene = BaseDataset._extract_scene_from_scene_json(
            scene_json,
            requested_autolabels,
            is_datums_synchronized=True,
            skip_missing_data=skip_missing_data,
            use_diskcache=use_diskcache,
            autolabel_root=autolabel_root,
        )

        # Return SynchronizedDataset with scenes built from dataset.json
        dataset_metadata = DatasetMetadata.from_scene_containers(
            [scene],
            requested_annotations,
            requested_autolabels,
            autolabel_root=autolabel_root,
        )
        super().__init__(
            dataset_metadata,
            scenes=[scene],
            datum_names=datum_names,
            requested_annotations=requested_annotations,
            requested_autolabels=requested_autolabels,
            backward_context=backward_context,
            forward_context=forward_context,
            generate_depth_from_datum=generate_depth_from_datum,
            only_annotated_datums=only_annotated_datums,
            use_virtual_camera_datums=use_virtual_camera_datums,
            accumulation_context=accumulation_context,
            transform_accumulated_box_points=transform_accumulated_box_points,
        )
