# Copyright 2020 Toyota Research Institute.  All rights reserved.
import copy
import itertools
import logging
#import math
import os
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np

from dgp.annotations import ANNOTATION_REGISTRY
from dgp.constants import ANNOTATION_KEY_TO_TYPE_ID
from dgp.datasets.base_dataset import (BaseDataset, DatasetMetadata, SceneContainer)
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.proto.scene_pb2 import Scene as ScenePb2
from dgp.utils.protobuf import open_pbobject


class ResampledSceneContainer(SceneContainer):
    """Object-oriented container for assembling datasets from collections of scenes.
    Each scene is resampled from a scene described within a sub-directory with an associated
    scene.json file, by a given sampling rate.
    """
    def __init__(
        self,
        scene_path,
        directory=None,
        autolabeled_scenes=None,
        is_datums_synchronized=False,
        use_diskcache=True,
        sampling_rate=1.0
    ):
        """Initialize a scene with a scene object and optionally provide the
        directory containing the scene.json to gather additional information
        for directory-based dataset loading mode.

        Parameters
        ----------
        scene_path: str
            Path to the Scene object containing data samples.

        directory: str, default: None
            Optional directory containing scene_<sha1>.json.

        autolabeled_scenes: dict, default: None
            Dictionary mapping <autolabel_key> (defined as:`autolabel_model`/`annotation_key`) to autolabeled SceneContainer.

        is_datums_synchronized: bool, default: False
            If True, sample-level synchronization is required i.e. each sample must contain all datums specified in the requested
            `datum_names`, and all samples in this scene must contain the same number of datums.
            If False, sample-level synchronization is not required i.e. samples are allowed to have different sets of datums.

        use_diskcache: bool, default: True
            If True, cache ScenePb2 object using diskcache. If False, save the object in memory.
            NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of scenes.

        sampling_rate: float, default: 1.0

        """
        super().__init__(
            scene_path=scene_path,
            directory=directory,
            autolabeled_scenes=autolabeled_scenes,
            is_datums_synchronized=is_datums_synchronized,
            use_diskcache=use_diskcache
        )
        self.sampling_rate = sampling_rate

    @property
    def scene(self):
        """ Returns scene.
        - If self.use_diskcache is True: returns the cached `_scene` if available, otherwise load the 
          scene and cache it.
        - If self.use_diskcache is False: returns `_scene` in memory if the instance has attribute
          `_scene`, otherwise load the scene and save it in memory.
          NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of scenes.
        """
        if self.use_diskcache:
            cached_path = self.scene_path if self.sampling_rate == 1.0 \
                else os.path.join(self.scene_path, f'{self.sampling_rate:.3f}')
            if cached_path in SceneContainer.SCENE_CACHE:
                _scene = SceneContainer.SCENE_CACHE.get(cached_path)
                if _scene is not None:
                    return _scene
            _scene = self.resample_scene(open_pbobject(self.scene_path, ScenePb2))
            SceneContainer.SCENE_CACHE.add(cached_path, _scene)
            return _scene
        else:
            if self._scene is None:
                self._scene = self.resample_scene(open_pbobject(self.scene_path, ScenePb2))
            return self._scene

    def resample_scene(self, scene):
        """Resample the scene based on given sampling rate.

        Parameters
        ----------
        scene: scene_pb2.Scene
            scene protobuf data with original sampling rate.

        Returns
        -------
        resampled_scene: scene_pb2.Scene
            scene protobuf data with giving sampling rate.
        """
        resampled_indices = np.linspace(0, len(scene.samples) - 1, int(len(scene.samples) * self.sampling_rate))
        resampled_indices = resampled_indices.astype(np.int32).tolist()
        resampled_scene = copy.deepcopy(scene)
        resampled_scene.ClearField('samples')
        resampled_scene.ClearField('data')
        datum_per_sample = len(scene.data) // len(scene.samples)
        for r_idx in resampled_indices:
            resampled_scene.samples.append(scene.samples[r_idx])
            for d_idx in range(datum_per_sample):
                resampled_scene.data.append(scene.data[r_idx * datum_per_sample + d_idx])
        return resampled_scene


class PredictionAgentDataset(BaseDataset):
    """Dataset for agent-centric prediction use cases, works just like normal SynchronizedSceneDataset,
    but guaranteeing trajectory of main agent is present in any fetched sample.

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

    requested_main_agent_types: tuple, default: 'car'
        Tuple of main agent types, i.e. ('car', 'pedestrian').
        The string should be the same as dataset_metadata.ontology_table.

    requested_main_agent_attributes: tuple[str], default: None
        Tuple of main agent attributes, i.e. ('moving', 'stopped'). This is predefined per dataset.
        By default (None) will include all attributes.

    requested_autolabels: tuple[str], default: None
        Currently not supported.

    forward_context: int, default: 0
        Forward context in frames [T+1, ..., T+forward]

    backward_context: int, default: 0
        Backward context in frames [T-backward, ..., T-1]

    min_main_agent_forward: int, default: 0
        Minimum forward samples for main agent. The main-agent will be guaranteed to appear
        minimum samples in forward context; i.e., the main-agent will appear in number of
        [min_main_agent_forward, forward_context] samples in forward direction.

    min_main_agent_backward: int, default: 0
        Minimum backward samples for main agent. The main-agent will be guaranteed to appear
        minimum samples in backward context; i.e., the main-agent will appear in number of
        [min_main_agent_backward, backward_context] samples in backward direction.

    generate_depth_from_datum: str, default: None
        Datum name of the point cloud. If is not None, then the depth map will be generated for the camera using
        the desired point cloud.

    use_3d_trajectories: bool, default: True
        Use 3D trajectories (from bounding_box_3d) as main reference of agents.
        This requires camera datum with bounding_box_3d annotations.

    batch_per_agent: bool, default: False
        Include whole trajectory of an agent in each batch Fetch, this is designed to be used for inference.
        If True, backward_context = forward_context = 0 implicitly.

    fps: float, default: -1
        Frame per second during data fetch. -1 means use original fps. 
    """
    ATTRIBUTE_NAME = 'behavior'

    def __init__(
        self,
        dataset_json,
        split='train',
        datum_names=None,
        requested_annotations=('bounding_box_3d', ),
        requested_main_agent_types=('car', ),
        requested_main_agent_attributes=None,
        requested_autolabels=None,
        forward_context=0,
        backward_context=0,
        min_main_agent_forward=0,
        min_main_agent_backward=0,
        generate_depth_from_datum=None,
        use_3d_trajectories=True,
        batch_per_agent=False,
        fps=-1
    ):
        self.generate_depth_from_datum = generate_depth_from_datum
        self.use_3d_trajectories = use_3d_trajectories
        assert len(datum_names)
        self.trajectories_reference = 'lidar' if any(['lidar' in datum_name.lower() \
                                          for datum_name in datum_names]) else datum_names[0].lower()
        self.use_3d_trajectories = use_3d_trajectories or self.trajectories_reference == 'lidar'
        self.annotation_reference = 'bounding_box_3d' if self.use_3d_trajectories else 'bounding_box_2d'
        assert self.annotation_reference in requested_annotations
        assert min_main_agent_backward <= backward_context and \
               min_main_agent_forward <= forward_context, 'Provide valid minimum context for main agent.'
        if batch_per_agent:  # fetch frame-by-frame for agent
            backward_context = forward_context = 0
        SynchronizedSceneDataset.set_context(self, backward=backward_context, forward=forward_context)
        self.min_main_agent_forward = min_main_agent_forward if min_main_agent_forward else forward_context
        self.min_main_agent_backward = min_main_agent_backward if min_main_agent_forward else backward_context

        # Extract all scenes from the scene dataset JSON for the appropriate split
        scenes = BaseDataset._extract_scenes_from_scene_dataset_json(
            dataset_json, split=split, requested_autolabels=requested_autolabels, is_datums_synchronized=True
        )
        metadata = BaseDataset._extract_metadata_from_scene_dataset_json(dataset_json)

        # Return SynchronizedDataset with scenes built from dataset.json
        dataset_metadata = DatasetMetadata.from_scene_containers(scenes, requested_annotations, requested_autolabels)
        name_to_id = dataset_metadata.ontology_table[self.annotation_reference].name_to_id
        self.requested_main_agent_types = tuple([name_to_id[atype] + 1 for atype in requested_main_agent_types])
        self.requested_main_agent_attributes = requested_main_agent_attributes

        # Resample scenes based on given fps
        self.sampling_rate = fps / metadata.frame_per_second if fps != -1 and metadata.frame_per_second else 1.0
        assert self.sampling_rate <= 1, f"Support lower fps only (current is {metadata.frame_per_second:.2f} fps)."
        resampled_scenes = []
        for scene in scenes:
            resampled_scene = ResampledSceneContainer(
                scene_path=scene.scene_path,
                directory=scene.directory,
                autolabeled_scenes=scene.autolabeled_scenes,
                is_datums_synchronized=scene.is_datums_synchronized,
                use_diskcache=scene.use_diskcache,
                sampling_rate=self.sampling_rate
            )
            resampled_scenes.append(resampled_scene)

        super().__init__(
            dataset_metadata,
            scenes=resampled_scenes,
            datum_names=datum_names,
            requested_annotations=requested_annotations
        )

        # Record each agent's life time
        self.batch_per_agent = batch_per_agent
        if batch_per_agent:
            self.dataset_agent_index = defaultdict(list)
            for index in range(len(self.dataset_item_index)):
                scene_idx, sample_idx_in_scene, main_agent_info, datum_names = self.dataset_item_index[index]
                _, main_agent_id = main_agent_info
                # Find the range of agents' trajectories
                if main_agent_id not in self.dataset_agent_index:
                    self.dataset_agent_index[main_agent_id] = [-1, -1, float('inf'), -1, []]
                self.dataset_agent_index[main_agent_id] = [
                    main_agent_id,
                    scene_idx,
                    min(sample_idx_in_scene, self.dataset_agent_index[main_agent_id][2]),  # birth sample index
                    max(sample_idx_in_scene, self.dataset_agent_index[main_agent_id][3]),  # death sample item index
                    datum_names
                ]
            self.dataset_agent_index = [v for k, v in self.dataset_agent_index.items()]

    def _build_item_index(self):
        """Builds an index of dataset items that refer to the scene index, agent index,
        sample index and datum_within_scene index. This refers to a particular dataset
        split. __getitem__ indexes into this look up table.

        Synchronizes at the sample-level and only adds sample indices if context frames are available.
        This is enforced by adding sample indices that fall in (bacwkard_context, N-forward_context) range.

        Returns
        -------
        item_index: list
            List of dataset items that contain index into
            (scene_idx, sample_idx_in_scene, (main_agent_idx, main_agent_id), [datum_name ...]).
        """
        logging.info(f'Building index for {self.__class__.__name__}, this will take a while.')
        st = time.time()
        # Fetch the item index per scene based on the selected datums.
        with Pool(cpu_count()) as proc:
            item_index = proc.starmap(
                self._item_index_for_scene, [(scene_idx, ) for scene_idx in range(len(self.scenes))]
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

    def _item_index_for_scene(self, scene_idx):
        scene = self.scenes[scene_idx]
        instance_id_to_trajectory = defaultdict(list)
        instance_id_to_segment_idx = defaultdict(int)
        # Decide main trajectories reference.
        # There are 3 cases to decide referenced annotation for trajectories:
        #   1. LIDAR only: bounding_box_3d as reference.
        #   2. CAMERA only: bounding_box_3d if use_3d_trajectories else bounding_box_2d.
        #   3. LIDAR + CAMERA: bounding_box_3d (from LIDAR) as reference.
        reference_datums = [datum_name for datum_name in scene.selected_datums \
                            if self.trajectories_reference in datum_name]

        # Only add to index if datum-name exists.
        if len(reference_datums) == 0:
            logging.debug('Skipping scene {} due to missing datums'.format(scene))
            return []

        # Define a safe sample range given desired context
        sample_range = np.arange(0, len(scene.datum_index))
        annotated_samples = scene.annotation_index[scene.datum_index[sample_range]].any(axis=(1, 2))
        for sample_idx_in_scene, is_annotated in zip(sample_range, annotated_samples):
            if not is_annotated:
                continue
            else:
                for datum_name in reference_datums:
                    datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_name)
                    annotation_key = self.annotation_reference
                    annotations = self.get_annotations(datum)
                    annotation_file = os.path.join(
                        self.scenes[scene_idx].directory, annotations[ANNOTATION_KEY_TO_TYPE_ID[annotation_key]]
                    )
                    bboxes_list = ANNOTATION_REGISTRY[annotation_key].load(
                        annotation_file, self.dataset_metadata.ontology_table[annotation_key]
                    )
                    for agent_idx, bbox in enumerate(bboxes_list):
                        # Filter undesired agent types and attributes.
                        if bbox.class_id not in self.requested_main_agent_types or \
                           (self.ATTRIBUTE_NAME in bbox.attributes and \
                            self.requested_main_agent_attributes is not None and \
                            bbox.attributes[self.ATTRIBUTE_NAME] not in self.requested_main_agent_attributes):
                            continue
                        # Make sure the track is sample-continuous
                        instance_index_prefix = \
                            f'{datum.id.name}_{str(scene_idx)}_{str(bbox.instance_id)}'
                        segment_idx_start = instance_id_to_segment_idx[instance_index_prefix] \
                                            if instance_index_prefix in instance_id_to_segment_idx else 0
                        for segment_idx in range(segment_idx_start, len(self.scenes[scene_idx].samples)):
                            instance_index_id = f'{instance_index_prefix}_{segment_idx}'
                            if instance_index_id in instance_id_to_trajectory and \
                               instance_id_to_trajectory[instance_index_id][-1][1] + 1 != sample_idx_in_scene:
                                continue
                            instance_id_to_trajectory[instance_index_id].append(
                                (scene_idx, sample_idx_in_scene, (agent_idx, bbox.instance_id), scene.selected_datums)
                            )
                            instance_id_to_segment_idx[instance_index_prefix] = segment_idx
                            break
        # Fiter unavailable items according to forward_context/backward_context for each agent.
        item_index = []
        trajectory_min_length = self.min_main_agent_backward + self.min_main_agent_forward + 1
        for id_ in instance_id_to_trajectory:
            scene_idx = instance_id_to_trajectory[id_][0][0]
            num_samples = len(self.scenes[scene_idx].samples)
            trajectory_length = len(instance_id_to_trajectory[id_])
            # Make sure track length is sufficient
            if trajectory_length >= trajectory_min_length:
                first_sample_idx = instance_id_to_trajectory[id_][0][1]
                final_sample_idx = instance_id_to_trajectory[id_][-1][1]
                # Crop out valid samples as items
                beg = self.min_main_agent_backward \
                    if self.min_main_agent_backward + first_sample_idx > self.backward_context \
                    else self.backward_context
                end = trajectory_length - (self.min_main_agent_forward \
                    if self.min_main_agent_forward + final_sample_idx < num_samples \
                    else self.forward_context)
                if end > beg:
                    item_index.append(instance_id_to_trajectory[id_][beg:end])
        return list(itertools.chain.from_iterable(item_index))

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset_agent_index) if self.batch_per_agent else len(self.dataset_item_index)

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

            "main_agent_idx": int
                Index of main agent in agent list (bounding_box_2d/bounding_box_3d).

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

        if self.batch_per_agent:
            # Get dataset agent index
            main_agent_id, scene_idx, sample_idx_in_scene_start, sample_idx_in_scene_end, datum_names = \
                self.dataset_agent_index[index]
        else:
            # Get dataset item index
            scene_idx, sample_idx_in_scene, main_agent_info, datum_names = self.dataset_item_index[index]
            _, main_agent_id = main_agent_info
            sample_idx_in_scene_start = sample_idx_in_scene - self.backward_context
            sample_idx_in_scene_end = sample_idx_in_scene + self.forward_context

        # All sensor data (including pose, point clouds and 3D annotations are
        # defined with respect to the sensor's reference frame captured at that
        # corresponding timestamp. In order to move to a locally consistent
        # reference frame, you will need to use the "pose" that specifies the
        # ego-pose of the sensor with respect to the local (L) frame (pose_LS).

        context_window = []
        reference_annotation_key = self.annotation_reference
        # Iterate through context samples
        for qsample_idx_in_scene in range(sample_idx_in_scene_start, sample_idx_in_scene_end + 1):
            # Main agent index may be different along the samples.
            synchronized_sample = []
            for datum_name in datum_names:
                datum_data = SynchronizedSceneDataset.get_datum_data(self, scene_idx, qsample_idx_in_scene, datum_name)

                if reference_annotation_key in datum_data:
                    # Over the main agent's trajectory, set main_agent_idx as None.
                    # Notice: main agent index may be different along the samples.
                    instance_matched = [
                        bbox.instance_id == main_agent_id for bbox in datum_data[reference_annotation_key]
                    ]
                    main_agent_idx_in_sample = instance_matched.index(True) if any(instance_matched) else None
                    datum_data['main_agent_idx'] = main_agent_idx_in_sample

                synchronized_sample.append(datum_data)
            context_window.append(synchronized_sample)
        return context_window
