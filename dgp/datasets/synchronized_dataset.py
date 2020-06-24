# Copyright 2019-2020 Toyota Research Institute.  All rights reserved.
"""Dataset for handling synchronized multi-modal samples for unsupervised,
self-supervised and supervised tasks.
This dataset is compliant with the TRI-ML Dataset Governance Policy (DGP).

Please refer to `dgp/proto/dataset.proto` for the exact specifications of our DGP.
"""
import logging
import time
from collections import OrderedDict

from dgp.datasets import ANNOTATION_KEY_TO_TYPE_ID
from dgp.datasets.annotations import (get_depth_from_point_cloud, 
                                      load_aligned_bounding_box_annotations,
                                      load_bounding_box_2d_annotations,
                                      load_bounding_box_3d_annotations,
                                      load_panoptic_segmentation_2d_annotations,
                                      load_semantic_segmentation_2d_annotations)
from dgp.datasets.base_dataset import (BaseSceneDataset, DatasetMetadata,
                                       _BaseDataset)
from dgp.utils.geometry import Pose
from dgp.utils.ontology import (build_detection_lookup_tables,
                                build_instance_lookup_tables,
                                build_semseg_lookup_tables)


class _SynchronizedDataset(_BaseDataset):
    """Multi-modal dataset with sample-level synchronization.
    See BaseDataset for input parameters for the parent class.

    Parameters
    ----------
    dataset_json: str
        Full path to the dataset json holding dataset metadata, ontology, and image and
        annotation paths.

    split: str, default: "train"
        Split of dataset to read ("train" | "val" | "test" | "train_overfit")

    datum_names: list, default: None
        Select list of datum names for synchronization (see self.select_datums(datum_names)).

    requested_annotations: tuple, default: None
        Tuple of annotation types, i.e. ('bounding_box_2d', 'bounding_box_3d'). Should be equivalent
        to directory containing annotation from dataset root.

    requested_autolabels: tuple[str], default: None
        Tuple of annotation types similar to `requested_annotations`, but associated with a particular autolabeling model.
        Expected format is "<model_id>/<annotation_type>"

    backward_context: int, default: 0
        Backward context in frames [T-backward, ..., T-1]

    forward_context: int, default: 0
        Forward context in frames [T+1, ..., T+forward]

    is_scene_dataset: bool, default: False
        Whether or not the dataset is constructed from a scene_dataset.json

    only_annotated_datums: bool, default: False
        If True, only datums with annotations matching the requested annotation types are returned.
    """
    def __init__(
        self,
        dataset_metadata,
        scenes=None,
        calibration_table=None,
        datum_names=None,
        requested_annotations=None,
        requested_autolabels=None,
        forward_context=0,
        backward_context=0,
        generate_depth_from_datum=None,
        is_scene_dataset=False,
        only_annotated_datums=False
    ):
        self.set_context(backward=backward_context, forward=forward_context)
        self.generate_depth_from_datum = generate_depth_from_datum
        self.only_annotated_datums = only_annotated_datums if requested_annotations else False

        super().__init__(
            dataset_metadata,
            scenes=scenes,
            calibration_table=calibration_table,
            datum_names=datum_names,
            requested_annotations=requested_annotations,
            is_scene_dataset=is_scene_dataset
        )

        # Populate detection lookup tables
        if self.requested_annotations:
            if self.dataset_metadata.ontology_table:
                # TODO: Refactor ontology and _build_detection_lookup_tables to create a map from annotation type to is
                #       lookup table
                # For now we assume .BOUNDING_BOX_3D and BOUNDING_BOX_2D has identical ontology
                if "bounding_box_3d" in self.dataset_metadata.ontology_table: 
                    ontology_key = "bounding_box_3d"
                    ontology = self.dataset_metadata.ontology_table[ontology_key]
                    build_detection_lookup_tables(self, ontology)
                elif "bounding_box_2d" in self.dataset_metadata.ontology_table:
                    ontology_key = "bounding_box_2d"
                    ontology = self.dataset_metadata.ontology_table[ontology_key]
                    build_detection_lookup_tables(self, ontology)

                if "semantic_segmentation_2d" in self.dataset_metadata.ontology_table:
                    ontology_key = "semantic_segmentation_2d"
                    ontology = self.dataset_metadata.ontology_table[ontology_key]
                    build_semseg_lookup_tables(self, ontology)

                if "instance_segmentation_2d" in self.dataset_metadata.ontology_table:
                    ontology_key = "instance_segmentation_2d"
                    ontology = self.dataset_metadata.ontology_table[ontology_key]
                    build_instance_lookup_tables(self, ontology)
            else:
                ontology = self.dataset_metadata.metadata.ontology
                build_detection_lookup_tables(self, ontology)

    def _build_item_index(self):
        """Builds an index of dataset items that refer to the scene index,
        sample index and datum_within_scene index. This refers to a particular dataset
        split. __getitem__ indexes into this look up table.

        Synchronizes at the sample-level and only adds sample indices if context frames are available.
        This is enforced by adding sample indices that fall in (bacwkard_context, N-forward_context) range.

        Returns
        -------
        item_index: list
            List of dataset items that contain index into
            (scene_idx, sample_within_scene_idx, [datum_within_scene_idx, ...]).
        """
        logging.info('Building index for {}, num_scenes={}'.format(self.__class__.__name__, len(self.scenes)))
        st = time.time()

        item_index = []
        for scene_idx, scene in enumerate(self.scenes):
            # Get the available datum names and their datum key index for the
            # specified datum_type (i.e. image, point_cloud). This assumes that
            # all samples within a scene have the same datum index for the
            # associated datum name.
            datum_name_to_datum_index = self.get_lookup_from_datum_name_to_datum_index_in_sample(
                scene_idx, sample_idx_in_scene=0, datum_type=None
            )

            # Remove datum names that are not selected, if desired
            if self.selected_datums is not None:
                # If the selected datums are available, identify the subset and
                # their datum index within the scene.
                datum_name_to_datum_index = {
                    datum_name: datum_name_to_datum_index[datum_name]
                    for datum_name in datum_name_to_datum_index
                    if datum_name in self.selected_datums
                }

            # Only add to index if datum-name exists
            if not len(datum_name_to_datum_index):
                return

            item_index.extend(
                self._build_item_index_per_scene(
                    scene_idx,
                    scene,
                    datum_name_to_datum_index=datum_name_to_datum_index,
                    backward_context=self.backward_context,
                    forward_context=self.forward_context
                )
            )

        if self.only_annotated_datums:
            item_index = list(filter(self._has_annotations, item_index))

        item_lengths = [len(item_tup) for item_tup in item_index]
        assert all([l == item_lengths[0] for l in item_lengths]
                   ), ('All sample items are not of the same length, datum names might be missing.')
        logging.info('Index built in {:.2}s.'.format(time.time() - st))
        return item_index

    def _build_item_index_per_scene(
        self, scene_idx, scene, datum_name_to_datum_index, backward_context, forward_context
    ):
        """Build the index of selected datums for an individual scene"""
        item_index = [(scene_idx, sample_idx, list(datum_name_to_datum_index.values()))
                      for sample_idx in range(backward_context,
                                              len(scene.samples) - forward_context)]
        return item_index

    def _has_annotations(self, dataset_item):
        """Identifies if the specified dataset item has ALL requested annotations, 
        OR if the dataset item has ANY requested autolabels."""
        scene_idx, sample_idx, datum_indices = dataset_item

        # Verify presence of annotations
        requested_annotation_ids = set([
            ANNOTATION_KEY_TO_TYPE_ID[annotation] for annotation in self.requested_annotations
        ])
        valid_datums = all([
            requested_annotation_ids.issubset(
                set(self.get_annotations(self.get_datum(scene_idx, sample_idx, datum_idx)).keys())
            ) for datum_idx in datum_indices
        ])

        # Check for autolabels
        if self.requested_autolabels:
            valid_autolabels = all([
                self.get_autolabels_for_datum(scene_idx, sample_idx, datum_idx) for datum_idx in datum_indices
            ])
            return (valid_datums or valid_autolabels)
        return valid_datums

    def set_context(self, backward=1, forward=1):
        """Set the context size and strides.

        Parameters
        ----------
        backward: int, default: 1
            Backward context in frames [T-backward, ..., T-1]

        forward: int, default: 1
            Forward context in frames [T+1, ..., T+forward]
        """
        assert backward >= 0 and forward >= 0, 'Provide valid context'
        self.backward_context = backward
        self.forward_context = forward

    def get_context_indices(self, sample_idx):
        """Utility to get the context sample indices given the sample_idx.

        Parameters
        ----------
        sample_idx: int
            Sample index (T).

        Returns
        -------
        context_indices: list
            Sample context indices for T, i.e. [T-1, T, T+1, T+2] if
            backward_context=1, forward_context=2.
        """
        return list(range(sample_idx - self.backward_context, sample_idx + self.forward_context + 1))

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset_item_index)

    def get_image_from_datum(self, scene_idx, sample_idx_in_scene, datum_idx_in_sample):
        """Get the sample image data from image datum.

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        datum_idx_in_sample: int
            Index of datum within sample datum keys

        Returns
        -------
        data: OrderedDict

            "timestamp": int
                Timestamp of the image in microseconds.

            "datum_name": str
                Sensor name from which the data was collected

            "rgb": PIL.Image (mode=RGB)
                Image in RGB format.

            "intrinsics": np.ndarray
                Camera intrinsics.

            "extrinsics": Pose
                Camera extrinsics with respect to the vehicle frame.

            "pose": Pose
                Pose of sensor with respect to the world/global/local frame
                (reference frame that is initialized at start-time). (i.e. this
                provides the ego-pose in `pose_WC`).

            "bounding_box_2d": np.ndarray dtype=np.float32
                Tensor containing bounding boxes for this sample
                (x, y, w, h) in absolute pixel coordinates

            "bounding_box_3d": list of BoundingBox3D
                3D Bounding boxes for this sample specified in this camera's
                reference frame. (i.e. this provides the bounding box (B) in
                the camera's (C) reference frame `box_CB`).

            "class_ids": np.ndarray dtype=np.int64
                Tensor containing class ids (aligned with ``bounding_box_2d`` and ``bounding_box_3d``)

            "instance_ids": np.ndarray dtype=np.int64
                Tensor containing instance ids (aligned with ``bounding_box_2d`` and ``bounding_box_3d``)

        """
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_idx_in_sample)
        assert datum.datum.WhichOneof('datum_oneof') == 'image'

        # Get camera calibration and extrinsics for the datum name
        sample = self.get_sample(scene_idx, sample_idx_in_scene)
        camera = self.get_camera_calibration(sample.calibration_key, datum.id.name)
        pose_VC = self.get_sensor_extrinsics(sample.calibration_key, datum.id.name)

        # Get ego-pose for the image (at the corresponding image timestamp t=Tc)
        pose_WC_Tc = Pose.from_pose_proto(datum.datum.image.pose) \
                     if hasattr(datum.datum.image, 'pose') else Pose()

        # Populate data for image data
        image, annotations = self.load_datum_and_annotations(scene_idx, sample_idx_in_scene, datum_idx_in_sample)
        data = OrderedDict({
            "timestamp": datum.id.timestamp.ToMicroseconds(),
            "datum_name": datum.id.name,
            "rgb": image,
            "intrinsics": camera.K,
            "extrinsics": pose_VC,
            "pose": pose_WC_Tc
        })

        # Extract 2D/3D bounding box labels if requested
        # Also checks if BOUNDING_BOX_2D and BOUNDING_BOX_3D annotation exists because some datasets
        # have sparse annotations.
        if self.requested_annotations:
            ann_root_dir = self.get_scene_directory(scene_idx)

        # TODO: Load the datum based on the type, no need to hardcode these conditions. In particular,
        # figure out how to handle joint conditions like this:
        if "bounding_box_2d" in self.requested_annotations and "bounding_box_3d" in self.requested_annotations and "bounding_box_2d" in annotations and "bounding_box_3d" in annotations:
            annotation_data = load_aligned_bounding_box_annotations(
                annotations, ann_root_dir, self.json_category_id_to_contiguous_id
            )
            data.update(annotation_data)

        elif "bounding_box_2d" in self.requested_annotations and "bounding_box_2d" in annotations:
            annotation_data = load_bounding_box_2d_annotations(
                annotations, ann_root_dir, self.json_category_id_to_contiguous_id
            )
            data.update(annotation_data)

        elif "bounding_box_3d" in self.requested_annotations and "bounding_box_3d" in annotations:
            annotation_data = load_bounding_box_3d_annotations(
                annotations, ann_root_dir, self.json_category_id_to_contiguous_id
            )
            data.update(annotation_data)

        if "semantic_segmentation_2d" in self.requested_annotations and "semantic_segmentation_2d" in annotations:
            annotation_data = load_semantic_segmentation_2d_annotations(
                annotations, ann_root_dir, self.semseg_label_lookup, self.VOID_ID
            )
            data.update(annotation_data)

        if "instance_segmentation_2d" in self.requested_annotations and "instance_segmentation_2d" in annotations:
            annotation_data = load_panoptic_segmentation_2d_annotations(
                annotations, ann_root_dir, self.instance_name_to_contiguous_id
            )
            data.update(annotation_data)

        return data

    def get_point_cloud_from_datum(self, scene_idx, sample_idx_in_scene, datum_idx_in_sample):
        """Get the sample image data from point cloud datum.

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        datum_idx_in_sample: int
            Index of datum within sample datum keys

        Returns
        -------
        data: OrderedDict

            "timestamp": int
                Timestamp of the image in microseconds.

            "datum_name": str
                Sensor name from which the data was collected

            "extrinsics": Pose
                Sensor extrinsics with respect to the vehicle frame.

            "point_cloud": np.ndarray (N x 3)
                Point cloud in the local/world (L) frame returning X, Y and Z
                coordinates. The local frame is consistent across multiple
                timesteps in a scene.

            "extra_channels": np.ndarray (N x M)
                Remaining channels from point_cloud (i.e. lidar intensity I or pixel colors RGB)

            "pose": Pose
                Pose of sensor with respect to the world/global/local frame
                (reference frame that is initialized at start-time). (i.e. this
                provides the ego-pose in `pose_WS` where S refers to the point
                cloud sensor (S)).

            "bounding_box_3d": list of BoundingBox3D
                3D Bounding boxes for this sample specified in this point cloud
                sensor's reference frame. (i.e. this provides the bounding box
                (B) in the sensor's (S) reference frame `box_SB`).

            "class_ids": np.ndarray dtype=np.int64
                Tensor containing class ids (aligned with ``bounding_box_3d``)

            "instance_ids": np.ndarray dtype=np.int64
                Tensor containing instance ids (aligned with ``bounding_box_3d``)

        """
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_idx_in_sample)
        assert datum.datum.WhichOneof('datum_oneof') == 'point_cloud'

        # Determine the ego-pose of the lidar sensor (S) with respect to the world
        # (W) @ t=Ts
        pose_WS_Ts = Pose.from_pose_proto(datum.datum.point_cloud.pose) \
                     if hasattr(datum.datum.point_cloud, 'pose') else Pose()
        # Get sensor extrinsics for the datum name
        pose_VS = self.get_sensor_extrinsics(
            self.get_sample(scene_idx, sample_idx_in_scene).calibration_key, datum.id.name
        )

        # Points are described in the Lidar sensor (S) frame captured at the
        # corresponding lidar timestamp (Ts).
        # Points are in the lidar sensor's (S) frame.
        X_S, annotations = self.load_datum_and_annotations(scene_idx, sample_idx_in_scene, datum_idx_in_sample)
        data = OrderedDict({
            "timestamp": datum.id.timestamp.ToMicroseconds(),
            "datum_name": datum.id.name,
            "extrinsics": pose_VS,
            "pose": pose_WS_Ts,
            "point_cloud": X_S[:, :3],
            "extra_channels": X_S[:, 3:],
        })

        # Extract 3D bounding box labels, if requested.
        # Also checks if BOUNDING_BOX_3D annotation exists because some datasets have sparse annotations.
        if "bounding_box_3d" in self.requested_annotations and "bounding_box_3d" in annotations:
            annotation_data = load_bounding_box_3d_annotations(
                annotations, self.get_scene_directory(scene_idx), self.json_category_id_to_contiguous_id
            )
            data.update(annotation_data)
        return data

    def get_datum_data(self, scene_idx, sample_idx_in_scene, datum_idx_in_sample):
        """Get the datum at (scene_idx, sample_idx_in_scene, datum_idx_in_sample) with labels (optionally)

        Parameters
        ----------
        scene_idx: int
            Scene index.

        sample_idx_in_scene: int
            Sample index within the scene.

        datum_idx_in_sample: int
            Datum index within the sample.
        """
        # Get corresponding datum and load it
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_idx_in_sample)
        datum_type = datum.datum.WhichOneof('datum_oneof')
        if datum_type == 'image':
            datum_data = self.get_image_from_datum(scene_idx, sample_idx_in_scene, datum_idx_in_sample)
            if self.generate_depth_from_datum:
                # Get the datum index for the desired point cloud sensor
                pc_datum_idx_in_sample = self.get_datum_index_for_datum_name(
                    scene_idx, sample_idx_in_scene, self.generate_depth_from_datum
                )
                # Generate the depth map for the camera using the point cloud and cache it
                datum_data['depth'] = get_depth_from_point_cloud(
                    self, scene_idx, sample_idx_in_scene, datum_idx_in_sample, pc_datum_idx_in_sample
                )
            return datum_data
        elif datum_type == 'point_cloud':
            return self.get_point_cloud_from_datum(scene_idx, sample_idx_in_scene, datum_idx_in_sample)
        else:
            raise ValueError('Unknown datum type: {}'.format(datum_type))

    def __getitem__(self, index):
        """Get the dataset item at index.

        Parameters
        ----------
        index: int
            Index of item to get.

        Returns
        -------
        data: list of OrderedDict, or list of list of OrderedDict

            "index": int
                Index of item to get

            "timestamp": int
                Timestamp of the image in microseconds.

            "datum_name": str
                Sensor name from which the data was collected

            "rgb": PIL.Image (mode=RGB)
                Image in RGB format.

            "intrinsics": np.ndarray
                Camera intrinsics.

            "extrinsics": Pose
                Camera extrinsics with respect to the world frame.

            "pose": Pose
                Pose of sensor with respect to the world/global frame

        For datasets with a single sensor selected, `__getitem__` returns an
        `OrderedDict` with the aforementioned keys.

        For datasets with multiple sensor(s) selected, `__getitem__` returns a
        list of OrderedDict(s), one item for each datum.

        For datasets with multiple datum(s) selected and temporal contexts > 1,
        `__getitem__` returns with the first list corresponding to the the temporal
        context and the inner list corresponding to the sensor.
        """
        assert self.dataset_item_index is not None, ('Index is not built, select datums before getting elements.')
        # Get dataset item index
        scene_idx, sample_idx_in_scene, datum_indices = self.dataset_item_index[index]

        # All sensor data (including pose, point clouds and 3D annotations are
        # defined with respect to the sensor's reference frame captured at that
        # corresponding timestamp. In order to move to a locally consistent
        # reference frame, you will need to use the "pose" that specifies the
        # ego-pose of the sensor with respect to the local (L) frame (pose_LS).

        # If context is not required
        if self.backward_context == 0 and self.forward_context == 0:
            return [
                self.get_datum_data(scene_idx, sample_idx_in_scene, datum_idx_in_sample)
                for datum_idx_in_sample in datum_indices
            ]
        else:
            sample = []
            # Iterate through context samples
            for qsample_idx_in_scene in range(
                sample_idx_in_scene - self.backward_context, sample_idx_in_scene + self.forward_context + 1
            ):
                sample_data = [
                    self.get_datum_data(scene_idx, qsample_idx_in_scene, datum_idx_in_sample)
                    for datum_idx_in_sample in datum_indices
                ]
                sample.append(sample_data)
            return sample


class SynchronizedSceneDataset(_SynchronizedDataset):
    """Main entry-point for multi-modal dataset with sample-level
    synchronization using scene directories as input.

    Note: This class is primarily used for self-supervised learning tasks where
    the default mode of operation is learning from a collection of scene
    directories.

    Parameters
    ----------
    scene_dataset_json: str
        Full path to the scene dataset json holding collections of paths to scene json.

    split: str, default: 'train'
        Split of dataset to read ("train" | "val" | "test" | "train_overfit").

    datum_names: list, default: None
        Select list of datum names for synchronization (see self.select_datums(datum_names)).

    requested_annotations: tuple, default: None
        Tuple of annotation types, i.e. ('bounding_box_2d', 'bounding_box_3d'). Should be equivalent
        to directory containing annotation from dataset root.

    requested_autolabels: tuple[str], default: None
        Tuple of annotation types similar to `requested_annotations`, but associated with a particular autolabeling model.
        Expected format is "<model_id>/<annotation_type>"

    backward_context: int, default: 0
        Backward context in frames [T-backward, ..., T-1]

    forward_context: int, default: 0
        Forward context in frames [T+1, ..., T+forward]

    generate_depth_from_datum: str, default: None
        Datum name of the point cloud. If is not None, then the depth map will be generated for the camera using
        the desired point cloud.

    only_annotated_datums: bool, default: False
        If True, only datums with annotations matching the requested annotation types are returned.

    Refer to _SynchronizedDataset for remaining parameters.
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
        only_annotated_datums=False
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


class SynchronizedScene(_SynchronizedDataset):
    """Main entry-point for multi-modal dataset with sample-level
    synchronization using a single scene JSON as input.

    Note: This class can be used to introspect a single scene given a scene
    directory with its associated scene JSON.

    Parameters
    ----------
    scene_json: str
        Full path to the scene json.

    datum_names: list, default: None
        Select list of datum names for synchronization (see self.select_datums(datum_names)).

    requested_annotations: tuple, default: None
        Tuple of annotation types, i.e. ('bounding_box_2d', 'bounding_box_3d'). Should be equivalent
        to directory containing annotation from dataset root.

    requested_autolabels: tuple[str], default: None
        Tuple of annotation types similar to `requested_annotations`, but associated with a particular autolabeling model.
        Expected format is "<model_id>/<annotation_type>"

    backward_context: int, default: 0
        Backward context in frames [T-backward, ..., T-1]

    forward_context: int, default: 0
        Forward context in frames [T+1, ..., T+forward]

    generate_depth_from_datum: str, default: None
        Datum name of the point cloud. If is not None, then the depth map will be generated for the camera using
        the desired point cloud.

    only_annotated_datums: bool, default: False
        If True, only datums with annotations matching the requested annotation types are returned.

    Refer to _SynchronizedDataset for remaining parameters.
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
