# Copyright 2019-2020 Toyota Research Institute.  All rights reserved.
"""Dataset for handling synchronized multi-modal samples for unsupervised,
self-supervised and supervised tasks.
This dataset is compliant with the TRI-ML Dataset Governance Policy (DGP).

Please refer to `dgp/proto/dataset.proto` for the exact specifications of our dgp.
"""
import itertools
import logging
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np

from dgp.datasets import BaseDataset, DatasetMetadata
from dgp.utils.accumulate import accumulate_points
from dgp.utils.annotation import get_depth_from_point_cloud


class _SynchronizedDataset(BaseDataset):
    """Multi-modal dataset with sample-level synchronization.
    See BaseDataset for input parameters for the parent class.

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

    backward_context: int, default: 0
        Backward context in frames [T-backward, ..., T-1]

    forward_context: int, default: 0
        Forward context in frames [T+1, ..., T+forward]

    accumulation_context: dict, default None
        Dictionary of datum names containing a tuple of (backward_context, forward_context) for sensor accumulation. For example, 'accumulation_context={'lidar':(3,1)}
        accumulates lidar points over the past three time steps and one forward step. Only valid for lidar and radar datums.

    generate_depth_from_datum: str, default: None
        Datum name of the point cloud. If is not None, then the depth map will be generated for the camera using
        the desired point cloud.

    only_annotated_datums: bool, default: False
        If True, only datums with annotations matching the requested annotation types are returned.

    transform_accumulated_box_points: bool, default: False
        Flag to use cuboid pose and instance id to warp points when using lidar accumulation.

    autolabel_root: str, default: None
        Path to autolabels.

    ignore_raw_datum: Optional[list[str]], default: None
        Optionally pass a list of datum types to skip loading their raw data (but still load their annotations). For
        example, ignore_raw_datum=['image'] will skip loading the image rgb data. The rgb key will be set to None.
        This is useful when only annotations or extrinsics are needed. Allowed values are any combination of
        'image','point_cloud','radar_point_cloud'    
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
        accumulation_context=None,
        generate_depth_from_datum=None,
        only_annotated_datums=False,
        transform_accumulated_box_points=False,
        autolabel_root=None,
        ignore_raw_datum=None,
    ):
        self.set_context(backward=backward_context, forward=forward_context, accumulation_context=accumulation_context)
        self.generate_depth_from_datum = generate_depth_from_datum
        self.only_annotated_datums = only_annotated_datums if requested_annotations or requested_autolabels else False
        self.transform_accumulated_box_points = transform_accumulated_box_points

        super().__init__(
            dataset_metadata,
            scenes=scenes,
            datum_names=datum_names,
            requested_annotations=requested_annotations,
            requested_autolabels=requested_autolabels,
            autolabel_root=autolabel_root,
            ignore_raw_datum=ignore_raw_datum,
        )

    def _build_item_index(self):
        """
        Synchronizes at the sample-level and only adds sample indices if context frames are available.
        This is enforced by adding sample indices that fall in (bacwkard_context, N-forward_context) range.

        Returns
        -------
        item_index: list
            List of dataset items that contain index into
            [(scene_idx, sample_within_scene_idx, [datum_names]), ...].
        """
        logging.info(
            f'{self.__class__.__name__} :: Building item index for {len(self.scenes)} scenes, this will take a while.'
        )
        st = time.time()

        # Calculate the maximum context to be asked for when using accumulation
        acc_back, acc_forward = 0, 0
        if self.accumulation_context:
            acc_context = [(v[0], v[1]) for v in self.accumulation_context.values()]
            acc_back, acc_forward = np.max(acc_context, 0)

        # Fetch the item index per-scene based on the selected datums.
        with Pool(cpu_count()) as proc:
            item_index = proc.starmap(
                partial(
                    _SynchronizedDataset._item_index_for_scene,
                    backward_context=self.backward_context + acc_back,
                    forward_context=self.forward_context + acc_forward,
                    only_annotated_datums=self.only_annotated_datums
                ), [(scene_idx, scene) for scene_idx, scene in enumerate(self.scenes)]
            )
        logging.info(f'Index built in {time.time() - st:.2f}s.')
        assert len(item_index) > 0, 'Failed to index items in the dataset.'

        # Chain the index across all scenes.
        item_index = list(itertools.chain.from_iterable(item_index))
        # Filter out indices that failed to load.
        item_index = [item for item in item_index if item is not None]
        item_lengths = [len(item_tup) for item_tup in item_index]
        assert all([l == item_lengths[0] for l in item_lengths]
                   ), ('All sample items are not of the same length, datum names might be missing.')
        return item_index

    @staticmethod
    def _item_index_for_scene(scene_idx, scene, backward_context, forward_context, only_annotated_datums):
        st = time.time()
        logging.debug(f'Indexing scene items for {scene.scene_path}')
        if not only_annotated_datums:
            # Define a safe sample range given desired context
            sample_range = np.arange(backward_context, len(scene.datum_index) - forward_context)
            # Build the item-index of selected samples for an individual scene.
            scene_item_index = [(scene_idx, sample_idx, scene.selected_datums) for sample_idx in sample_range]
            logging.debug(f'No annotation filter--- Scene item index built in {time.time() - st:.2f}s.')
        else:
            # Filter out samples that do not have annotations.
            # A sample is considered annotated if ANY selected datum in the sample contains ANY requested annotation.
            sample_range = np.arange(0, len(scene.datum_index))
            annotated_samples = scene.annotation_index[scene.datum_index[sample_range]].any(axis=(1, 2))
            scene_item_index = []
            for idx in range(backward_context, len(scene.datum_index) - forward_context):
                if all(annotated_samples.data[idx - backward_context:idx + 1 + forward_context]):
                    scene_item_index.append((scene_idx, sample_range[idx], scene.selected_datums))
            logging.debug(f'Annotation filter -- Scene item index built in {time.time() - st:.2f}s.')
        return scene_item_index

    def set_context(self, backward=1, forward=1, accumulation_context=None):
        """Set the context size and strides.

        Parameters
        ----------
        backward: int, optional
            Backward context in frames [T-backward, ..., T-1]. Default: 1.

        forward: int, optional
            Forward context in frames [T+1, ..., T+forward]. Default: 1.

        accumulation_context: dict, optional
            Dictionary of accumulation context. Default: None
        """
        assert backward >= 0 and forward >= 0, 'Provide valid context'

        if accumulation_context:
            for k, v in accumulation_context.items():
                assert v[0] >= 0, f'Provide valid accumulation backward context for {k}'
                assert v[1] >= 0, f'Provide valid accumulation forward context for {k}'
                # Forward accumulation should almost never be used for inference
                if v[1] > 0:
                    logging.warning(
                        f'Forward accumulation context is enabled for {k}. Doing so at inference time is not suggested, is this intentional?'
                    )

        self.backward_context = backward
        self.forward_context = forward
        self.accumulation_context = accumulation_context

        # Use lower case datum names for accumulation context
        if self.accumulation_context:
            self.accumulation_context = {k.lower(): v for k, v in self.accumulation_context.items()}

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

    def get_datum_data(self, scene_idx, sample_idx_in_scene, datum_name):
        """Get the datum at (scene_idx, sample_idx_in_scene, datum_name) with labels (optionally)

        Parameters
        ----------
        scene_idx: int
            Scene index.

        sample_idx_in_scene: int
            Sample index within the scene.

        datum_name: str
            Datum within the sample.

        Raises
        ------
        ValueError
            Raised if the type of the requested datum is unsupported.
        """
        # Get corresponding datum and load it
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_name)
        datum_type = datum.datum.WhichOneof('datum_oneof')

        if datum_type == 'image':
            datum_data, annotations = self.get_image_from_datum(scene_idx, sample_idx_in_scene, datum_name)
            if self.generate_depth_from_datum:
                # Generate the depth map for the camera using the point cloud and cache it
                datum_data['depth'] = get_depth_from_point_cloud(
                    self, scene_idx, sample_idx_in_scene, datum_name, self.generate_depth_from_datum.lower()
                )
        elif datum_type == 'point_cloud':
            datum_data, annotations = self.get_point_cloud_from_datum(scene_idx, sample_idx_in_scene, datum_name)
        elif datum_type == 'file_datum':
            datum_data, annotations = self.get_file_meta_from_datum(scene_idx, sample_idx_in_scene, datum_name)
        elif datum_type == 'radar_point_cloud':
            datum_data, annotations = self.get_radar_point_cloud_from_datum(scene_idx, sample_idx_in_scene, datum_name)
        else:
            raise ValueError('Unknown datum type: {}'.format(datum_type))

        # TODO: Implement data/annotation load-time transforms, `torchvision` style.
        if annotations:
            datum_data.update(annotations)

        datum_data['datum_type'] = datum_type

        return datum_data

    def __getitem__(self, index):
        """Get the dataset item at index.

        Parameters
        ----------
        index: int
            Index of item to get.

        Returns
        -------
        data: list of list of OrderedDict

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

        Returns a list of list of OrderedDict(s).
        Outer list corresponds to temporal ordering of samples. Each element is
        a list of OrderedDict(s) corresponding to synchronized datums.

        In other words, __getitem__ returns a nested list with the ordering as
        follows: (C, D, I), where
            C = forward_context + backward_context + 1,
            D = len(datum_names)
            I = OrderedDict item
        """
        assert self.dataset_item_index is not None, ('Index is not built, select datums before getting elements.')
        # Get dataset item index
        scene_idx, sample_idx_in_scene, datum_names = self.dataset_item_index[index]

        # All sensor data (including pose, point clouds and 3D annotations are
        # defined with respect to the sensor's reference frame captured at that
        # corresponding timestamp. In order to move to a locally consistent
        # reference frame, you will need to use the "pose" that specifies the
        # ego-pose of the sensor with respect to the local (L) frame (pose_LS).

        datums_with_context = dict()
        for datum_name in datum_names:

            acc_back, acc_forward = 0, 0
            if self.accumulation_context:
                accumulation_context = self.accumulation_context.get(datum_name.lower(), (0, 0))
                acc_back, acc_forward = accumulation_context

            # We need to fetch our datum's data for all time steps we care about. If our datum's index is i,
            # then to support the requested backward/forward context, we need indexes i- backward ... i + forward.
            # However if we also have accumulation, our first sample index = (i - backward) also needs data starting from
            # index = i - backward - acc_back (it accumulates over a window of size (acc_back, acc_forward).
            # The final sample will need i + forward + acc_forward. Combined, we need samples from
            # [i-backward_context - acc_back, i+forward_context + acc_forward].
            datum_list = [
                self.get_datum_data(scene_idx, sample_idx_in_scene + offset, datum_name)
                for offset in range(-1 * (self.backward_context + acc_back), self.forward_context + acc_forward + 1)
            ]

            if acc_back != 0 or acc_forward != 0:
                # Make sure we have the right datum type
                assert 'point_cloud' in datum_list[0], "Accumulation is only defined for radar and lidar currently."
                # Our datum list now has samples ranging from [i-backward_context-acc_back to i+forward_context+acc_forward]
                # We instead need a list that ranges from [i-backward_context to i+forward_context] AFTER accumulation
                # This means the central sample in our datum list starts at index = acc_back.
                datum_list = [
                    accumulate_points(
                        datum_list[k - acc_back:k + acc_forward + 1], datum_list[k],
                        self.transform_accumulated_box_points
                    ) for k in range(acc_back,
                                     len(datum_list) - acc_forward)
                ]

            datums_with_context[datum_name] = datum_list

        # We now have a dictionary of lists, swap the order to build context windows
        context_window = []
        for t in range(self.backward_context + self.forward_context + 1):
            context_window.append([datums_with_context[datum_name][t] for datum_name in datum_names])

        return context_window


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

    accumulation_context: dict, default None
        Dictionary of datum names containing a tuple of (backward_context, forward_context) for sensor accumulation.
        For example, 'accumulation_context={'lidar':(3,1)} accumulates lidar points over the past three time steps and
        one forward step. Only valid for lidar and radar datums.

    generate_depth_from_datum: str, default: None
        Datum name of the point cloud. If is not None, then the depth map will be generated for the camera using
        the desired point cloud.

    only_annotated_datums: bool, default: False
        If True, only datums with annotations matching the requested annotation types are returned.

    skip_missing_data: bool, default: False
        If True, check for missing files and skip during datum index building.

    dataset_root: str
        Optional path to dataset root folder. Useful if dataset scene json is not in the same directory as the rest of the data.

    transform_accumulated_box_points: bool, default: False
        Flag to use cuboid pose and instance id to warp points when using lidar accumulation.

    use_diskcache: bool, default: True
        If True, cache ScenePb2 object using diskcache. If False, save the object in memory.
        NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of scenes.

    autolabel_root: str, default: None
        Path to autolabels if not stored inside scene root. Note this must still respect the scene structure, i.e,
        autolabel_root = '/some-autolabels' means the autolabel scene.json is found at
        /some-autolabels/<scene-dir>/autolabels/my-model/scene.json.

    ignore_raw_datum: Optional[list[str]], default: None
        Optionally pass a list of datum types to skip loading their raw data (but still load their annotations). For
        example, ignore_raw_datum=['image'] will skip loading the image rgb data. The rgb key will be set to None.
        This is useful when only annotations or extrinsics are needed. Allowed values are any combination of
        'image','point_cloud','radar_point_cloud'

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
        accumulation_context=None,
        generate_depth_from_datum=None,
        only_annotated_datums=False,
        skip_missing_data=False,
        dataset_root=None,
        transform_accumulated_box_points=False,
        use_diskcache=True,
        autolabel_root=None,
        ignore_raw_datum=None,
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
            scenes, requested_annotations, requested_autolabels, autolabel_root=autolabel_root
        )
        super().__init__(
            dataset_metadata,
            scenes=scenes,
            datum_names=datum_names,
            requested_annotations=requested_annotations,
            requested_autolabels=requested_autolabels,
            backward_context=backward_context,
            forward_context=forward_context,
            accumulation_context=accumulation_context,
            generate_depth_from_datum=generate_depth_from_datum,
            only_annotated_datums=only_annotated_datums,
            transform_accumulated_box_points=transform_accumulated_box_points,
            autolabel_root=autolabel_root,
            ignore_raw_datum=ignore_raw_datum,
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

    accumulation_context: dict, default None
        Dictionary of datum names containing a tuple of (backward_context, forward_context) for sensor accumulation.
        For example, 'accumulation_context={'lidar':(3,1)} accumulates lidar points over the past three time steps and
        one forward step. Only valid for lidar and radar datums.

    generate_depth_from_datum: str, default: None
        Datum name of the point cloud. If is not None, then the depth map will be generated for the camera using
        the desired point cloud.

    only_annotated_datums: bool, default: False
        If True, only datums with annotations matching the requested annotation types are returned.

    transform_accumulated_box_points: bool, default: False
        Flag to use cuboid pose and instance id to warp points when using lidar accumulation.

    use_diskcache: bool, default: True
        If True, cache ScenePb2 object using diskcache. If False, save the object in memory.
        NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of scenes.

    autolabel_root: str, default: None
        Path to autolabels if not stored inside scene root. Note this must still respect the scene structure, i.e,
        autolabel_root = '/some-autolabels' means the autolabel scene.json is found at
        /some-autolabels/<scene-dir>/autolabels/my-model/scene.json.

    ignore_raw_datum: Optional[list[str]], default: None
        Optionally pass a list of datum types to skip loading their raw data (but still load their annotations). For
        example, ignore_raw_datum=['image'] will skip loading the image rgb data. The rgb key will be set to None.
        This is useful when only annotations or extrinsics are needed. Allowed values are any combination of
        'image','point_cloud','radar_point_cloud'

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
        accumulation_context=None,
        generate_depth_from_datum=None,
        only_annotated_datums=False,
        transform_accumulated_box_points=False,
        use_diskcache=True,
        autolabel_root=None,
        ignore_raw_datum=None,
    ):
        if not use_diskcache:
            logging.warning('Instantiating a dataset with use_diskcache=False may exhaust memory with a large dataset.')

        # Extract a single scene from the scene JSON
        scene = BaseDataset._extract_scene_from_scene_json(
            scene_json,
            requested_autolabels,
            is_datums_synchronized=True,
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
            accumulation_context=accumulation_context,
            generate_depth_from_datum=generate_depth_from_datum,
            only_annotated_datums=only_annotated_datums,
            transform_accumulated_box_points=transform_accumulated_box_points,
            autolabel_root=autolabel_root,
            ignore_raw_datum=ignore_raw_datum,
        )
