# Copyright 2021-2022 Toyota Research Institute.  All rights reserved.

import hashlib
import itertools
import logging
import os
import random
import time
from collections import OrderedDict, defaultdict
from functools import lru_cache, partial
from multiprocessing import Pool, cpu_count

import numpy as np
from diskcache import Cache

from dgp import DGP_CACHE_DIR, FEATURE_ONTOLOGY_FOLDER, ONTOLOGY_FOLDER
from dgp.agents import (
    AGENT_REGISTRY,
    AGENT_TYPE_TO_ANNOTATION_TYPE,
    ANNOTATION_TYPE_TO_AGENT_TYPE,
)
from dgp.annotations import ONTOLOGY_REGISTRY
from dgp.constants import (
    ALL_FEATURE_TYPES,
    ANNOTATION_TYPE_ID_TO_KEY,
    FEATURE_TYPE_ID_TO_KEY,
)
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.features import FEATURE_ONTOLOGY_REGISTRY
from dgp.proto import dataset_pb2
from dgp.proto.agent_pb2 import AgentGroup as AgentGroupPb2
from dgp.proto.agent_pb2 import AgentsSlices as AgentsSlicesPb2
from dgp.proto.agent_pb2 import AgentTracks as AgentTracksPb2
from dgp.proto.dataset_pb2 import Agents as AgentsPb2
from dgp.utils.protobuf import open_pbobject


class BaseAgentDataset:
    """A base class representing a Agent Dataset. Provides utilities for parsing and slicing
    DGP format agent datasets.

    Parameters
    ----------
    Agent_dataset_metadata: DatasetMetadata
        Dataset metadata object that encapsulates dataset-level agents metadata for
        both operating modes (scene or JSON).

    agent_groups: list[AgentContainer]
        List of AgentContainer objects to be included in the dataset.

    split: str, default: None
        Split of dataset to read ("train" | "val" | "test" | "train_overfit").
        If the split is None, the split type is not known and the dataset can
        be used for unsupervised / self-supervised learning.
    """
    def __init__(
        self,
        Agent_dataset_metadata,
        agent_groups,
        split=None,
    ):
        logging.info(f'Instantiating dataset with {len(agent_groups)} scenes.')
        # Dataset metadata
        self.Agent_dataset_metadata = Agent_dataset_metadata

        self.split = split

        # Scenes management
        self.agent_groups = agent_groups

        # Dataset item index
        # This is the main index into the pytorch Dataset, where the index is
        # used to retrieve the item in the dataset via __getitem__.
        self.dataset_item_index = self._build_item_index()

    @staticmethod
    def _load_agents_data(agent_group, ontology_table, feature_ontology_table):
        """Call loading method from agent_group to load agent slice data and agent track data.

        Parameters
        ----------
        agent_group: AgentContainer
            Group of agents from a scene.

        ontology_table: dict[str->dgp.annotations.Ontology]
            A dictionary mapping annotation type key(s) to Ontology(s)

        feature_ontology_table: dict, default: None
            A dictionary mapping feature type key(s) to Ontology(s).

        Returns
        -------
        agent_group: AgentContainer
           An AgentContainer objects with agents loaded.
        """
        agent_group.load_agent_data(ontology_table, feature_ontology_table)
        return agent_group

    @staticmethod
    def _extract_agent_groups_from_agent_dataset_json(
        agent_dataset_json,
        requested_agent_type,
        split='train',
        use_diskcache=True,
    ):
        """Extract agent container objects from the agent dataset JSON
        for the appropriate split.

        Parameters
        ----------
        agent_dataset_json: str
            Path of the dataset.json

        requested_agent_type: str, default: 'train'
            Split of dataset to read ("train" | "val" | "test" | "train_overfit").

        split: str, default: 'train'
            Split of dataset to read ("train" | "val" | "test" | "train_overfit").

        use_diskcache: bool, default: True
            If True, cache ScenePb2 object using diskcache. If False, save the object in memory.
            NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of scenes in this
            scene dataset.

        Returns
        -------
        scene_containers: list
            List of SceneContainer objects.
        """
        # Identify splits
        assert split in ("train", "val", "test", "train_overfit")
        split_enum = {
            "train": dataset_pb2.TRAIN,
            "val": dataset_pb2.VAL,
            "test": dataset_pb2.TEST,
            "train_overfit": dataset_pb2.TRAIN_OVERFIT
        }[split]

        # Load the dataset and dataset root
        if not agent_dataset_json.startswith("s3://"):
            assert os.path.exists(agent_dataset_json), 'Path {} does not exist'.format(agent_dataset_json)
        logging.info("Loading dataset from {}, split={}".format(agent_dataset_json, split))

        agent_dataset_root = os.path.dirname(agent_dataset_json)
        agent_dataset = open_pbobject(agent_dataset_json, AgentsPb2)

        logging.info("Generating agents for split={}".format(split))
        st = time.time()
        agent_jsons = [
            os.path.join(agent_dataset_root, _f) for _f in list(agent_dataset.agents_splits[split_enum].filenames)
        ]

        # Load all agent containers in parallel. Each agent container shall be 1-1 correspondence to a scene.
        with Pool(cpu_count()) as proc:
            agent_containers = list(
                proc.map(
                    partial(
                        AgentDataset._get_agent_container,
                        requested_agent_type=requested_agent_type,
                        use_diskcache=use_diskcache
                    ), agent_jsons
                )
            )

        logging.info("Scene generation completed in {:.2f}s".format(time.time() - st))
        return agent_containers

    @staticmethod
    def _get_agent_container(
        agent_json,
        requested_agent_type,
        use_diskcache=True,
    ):
        """Extract scene objects and calibration from the scene dataset JSON
        for the appropriate split.

        Parameters
        ----------
        agent_json: str
            Path of the agent_scene.json

        requested_agent_type: str, default: 'train'
            Split of dataset to read ("train" | "val" | "test" | "train_overfit").

        use_diskcache: bool, default: True
            If True, cache ScenePb2 object using diskcache. If False, save the object in memory.
            NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of scenes in this
            scene dataset.

        Returns
        -------
        scene_containers: list
            List of SceneContainer objects.
        """
        agent_dir = os.path.dirname(agent_json)

        logging.debug(f"Loading agents from {agent_json}")
        agent_container = AgentContainer(
            agent_json, requested_agent_type, directory=agent_dir, use_diskcache=use_diskcache
        )
        return agent_container

    def _build_item_index(self):
        """Builds an index of dataset items that refer to the scene index,
        sample index and selected datum names. __getitem__ indexes into this look up table.

        Returns
        -------
        item_index: list
            List of dataset items that contain index into
            (scene_idx, sample_idx_in_scene, (datum_name_1, datum_name_2, ...)).
        """
        raise NotImplementedError

    def __len__(self):
        """Return the length of the dataset."""
        raise NotImplementedError

    def __getitem__(self, index):
        """Get the dataset item at index."""
        raise NotImplementedError

    def __hash__(self):
        """Hashes the dataset instance that is consistent across Python instances."""
        logging.debug('Hashing dataset with dataset directory, split and datum-index')
        return int(
            hashlib.sha1(
                self.Agent_dataset_metadata.directory.encode() + str(self.dataset_item_index).encode() +
                str(self.split).encode()
            ).hexdigest(), 16
        )


class AgentContainer:
    """Object-oriented container for holding agent information from a scene.
    """
    RANDOM_STR = ''.join([str(random.randint(0, 9)) for _ in range(5)])
    cache_suffix = os.environ.get('DGP_SCENE_CACHE_SUFFIX', RANDOM_STR)
    cache_dir = os.path.join(DGP_CACHE_DIR, f'dgp_diskcache_{cache_suffix}')
    AGENT_GROUP_CACHE = Cache(cache_dir)

    def __init__(self, agent_file_path, requested_agent_type, directory=None, use_diskcache=True):
        """Initialize a scene with a agent group object and optionally provide the
        directory containing the agent.json to gather additional information
        for directory-based dataset loading mode.

        Parameters
        ----------
        agent_file_path: str
            Path to the agent object containing agent tracks and agent slices.

        requested_agent_type: str, default: 'train'
            Split of dataset to read ("train" | "val" | "test" | "train_overfit").

        directory: str, default: None
            Directory containing scene_<sha1>.json.

        use_diskcache: bool, default: True
            If True, cache AgentGroupPb2 object using diskcache. If False, save the object in memory.
            NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of scenes.
        """
        self.agent_file_path = agent_file_path
        self.directory = directory
        self.use_diskcache = use_diskcache
        self.requested_agent_type = requested_agent_type
        self._agent_group = None
        self.sample_id_to_agent_snapshots = {}
        self.instance_id_to_agent_snapshots = {}
        logging.debug(f"Loading agent-based dataset from {self.directory}")

    @property
    def agent_group(self):
        """ Returns agent group.
        - If self.use_diskcache is True: returns the cached `_agent_group` if available, otherwise load the
          agent group and cache it.
        - If self.use_diskcache is False: returns `_agent_group` in memory if the instance has attribute
          `_agent_group`, otherwise load the agent group and save it in memory.
          NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of agent groups.
        """
        # TODO: Profile disk loading to decide if use_diskcache option is necessary.
        if self.use_diskcache:
            if self.agent_file_path in AgentContainer.AGENT_GROUP_CACHE:
                _agent_group = AgentContainer.AGENT_GROUP_CACHE.get(self.agent_file_path)
                if _agent_group is not None:
                    return _agent_group
            _agent_group = open_pbobject(self.agent_file_path, AgentGroupPb2)
            AgentContainer.AGENT_GROUP_CACHE.add(self.agent_file_path, _agent_group)
            return _agent_group
        else:
            if self._agent_group is None:
                self._agent_group = open_pbobject(self.agent_file_path, AgentGroupPb2)
            return self._agent_group

    def __repr__(self):
        return "AgentContainer[{}][agents: {}]".format(self.directory, len(self.instance_id_to_agent_snapshots))

    @property
    def ontology_files(self):
        """Returns the ontology files for the agent group.

        Returns
        -------
        ontology_files: dict
            Maps annotation_key -> filename.

            For example:
            filename = agent.ontology_files['bounding_box_2d']
        """
        # Load ontology files.
        ontology_files = {
            ANNOTATION_TYPE_ID_TO_KEY[ann_id]: os.path.join(self.directory, ONTOLOGY_FOLDER, "{}.json".format(f))
            for ann_id, f in self.agent_group.agent_ontologies.items()
        }

        return ontology_files

    @property
    def feature_ontology_files(self):
        """Returns the feature ontology files for a agent group.

        Returns
        -------
        ontology_files: dict
            Maps annotation_key -> filename.

            For example:
            filename = agent.feature_ontology_files['agent_3d']
        """
        # Load ontology files.
        feature_ontology_files = {
            FEATURE_TYPE_ID_TO_KEY[feature_id]:
            os.path.join(self.directory, FEATURE_ONTOLOGY_FOLDER, "{}.json".format(f))
            for feature_id, f in self.agent_group.feature_ontologies.items()
        }
        return feature_ontology_files

    @property
    @lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    def metadata_index(self):
        """Helper for building metadata index.

        TODO: Need to verify that the hashes are unique, and these lru-cached
        properties are consistent across disk-cached reads.
        """
        logging.debug(f'Building metadata index for agent group {self.agent_file_path}')
        agent_group = self.agent_group
        return {
            'log_id': agent_group.log,
            'agent_group_name': agent_group.name,
            'agent_group_description': agent_group.description
        }

    def agent_slice(self, sample_id):  # pylint: disable=missing-param-doc,missing-type-doc
        """Return AgentSnapshotList in a frame."""
        return self.sample_id_to_agent_snapshots[sample_id]

    def agent_track(self, instance_id):  # pylint: disable=missing-param-doc,missing-type-doc
        """Return AgentSnapshotList in a track."""
        return self.instance_id_to_agent_snapshots[instance_id]

    def load_agent_data(self, ontology_table, feature_ontology_table):
        """Load agent slice data and agent track data.

        Parameters
        ----------
        ontology_table: dict[str->dgp.annotations.Ontology]
            Ontology object *per annotation type*.
            The original ontology table.
            {
                "bounding_box_2d": BoundingBoxOntology[<ontology_sha>],
                "autolabel_model_1/bounding_box_2d": BoundingBoxOntology[<ontology_sha>],
                "semantic_segmentation_2d": SemanticSegmentationOntology[<ontology_sha>]
                "bounding_box_3d": BoundingBoxOntology[<ontology_sha>],
            }

        feature_ontology_table: dict[str->dgp.features.FeatureOntology]
            Ontology object *per feature type*.
            The original feature ontology table.
            {
                "agent_2d": AgentFeatureOntology,
                "agent_3d": AgentFeatureOntology,
                "ego_intention": AgentFeatureOntology
            }

        """

        agent_slices_path = self.agent_group.agents_slices_file
        agent_tracks_path = self.agent_group.agent_tracks_file

        if agent_slices_path is None or agent_tracks_path is None:
            logging.debug('Skipping agent_group {} due to missing agents'.format(self.agent_group))
            return []

        agents_slices_file = os.path.join(self.directory, agent_slices_path)
        agent_tracks_file = os.path.join(self.directory, agent_tracks_path)

        agents_slices_pb2 = open_pbobject(agents_slices_file, AgentsSlicesPb2)
        agent_tracks_pb2 = open_pbobject(agent_tracks_file, AgentTracksPb2)

        agent_ontology = ontology_table[AGENT_TYPE_TO_ANNOTATION_TYPE[self.requested_agent_type]]
        for agent_slice in agents_slices_pb2.agents_slices:
            agents_list = AGENT_REGISTRY[self.requested_agent_type
                                         ].load(agent_slice.agent_snapshots, agent_ontology, feature_ontology_table)
            self.sample_id_to_agent_snapshots[agent_slice.slice_id.index] = agents_list

        for track in agent_tracks_pb2.agent_tracks:
            self.instance_id_to_agent_snapshots[track.instance_id] = AGENT_REGISTRY[
                self.requested_agent_type].load(track.agent_snapshots, agent_ontology, feature_ontology_table)


class AgentDataset(BaseAgentDataset):
    """Dataset for agent-centric prediction or planning use cases, works just like normal SynchronizedSceneDataset,
    but guaranteeing trajectory of main agent is present in any fetched sample.

    Parameters
    ----------
    scene_dataset_json: str
        Full path to the scene dataset json holding collections of paths to scene json.

    agents_dataset_json: str
        Full path to the agent dataset json holding collections of paths to scene json.

    split: str, default: 'train'
        Split of dataset to read ("train" | "val" | "test" | "train_overfit").

    datum_names: list, default: None
        Select list of datum names for synchronization (see self.select_datums(datum_names)).

    requested_agent_type: tuple, default: None
        Tuple of agent type, i.e. ('agent_2d', 'agent_3d'). Only one type of agent can be requested.

    requested_main_agent_classes: tuple, default: 'car'
        Tuple of main agent types, i.e. ('car', 'pedestrian').
        The string should be the same as dataset_metadata.ontology_table.

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

    batch_per_agent: bool, default: False
        Include whole trajectory of an agent in each batch fetch, this is designed to be used for inference.
        If True, backward_context = forward_context = 0 implicitly.

    use_diskcache: bool, default: True
        If True, cache ScenePb2 object using diskcache. If False, save the object in memory.
        NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of scenes.

    """
    def __init__(
        self,
        scene_dataset_json,
        agents_dataset_json,
        split='train',
        datum_names=None,
        requested_agent_type='agent_3d',
        requested_main_agent_classes=('car', ),
        requested_feature_types=None,
        requested_autolabels=None,
        forward_context=0,
        backward_context=0,
        min_main_agent_forward=0,
        min_main_agent_backward=0,
        generate_depth_from_datum=None,
        batch_per_agent=False,
        use_diskcache=True,
    ):

        # Make sure requested agent type keys match protos
        if requested_agent_type is not None:
            assert requested_agent_type in AGENT_REGISTRY, "Invalid agent type requested!"
            self.requested_agent_type = requested_agent_type
        else:
            self.requested_agent_type = ()

        # Make sure requested feature type keys match protos
        if requested_feature_types is not None:
            assert all(
                requested_feature_type in ALL_FEATURE_TYPES for requested_feature_type in requested_feature_types
            ), "Invalid feature type requested!"
            self.requested_feature_types = requested_feature_types
        else:
            self.requested_feature_types = ()

        self.batch_per_agent = batch_per_agent
        self.generate_depth_from_datum = generate_depth_from_datum
        datum_names = sorted(set(datum_names)) if datum_names else set([])
        self.selected_datums = [_d.lower() for _d in datum_names]
        # TODO: support for ParallelDomainDataset.
        if len(self.selected_datums) != 0:
            self.synchronized_dataset = SynchronizedSceneDataset(
                scene_dataset_json,
                split=split,
                backward_context=backward_context,
                requested_autolabels=requested_autolabels,
                forward_context=forward_context,
                datum_names=self.selected_datums,
                use_diskcache=use_diskcache
            )

        assert min_main_agent_backward <= backward_context and \
               min_main_agent_forward <= forward_context, 'Provide valid minimum context for main agent.'
        if batch_per_agent:  # fetch frame-by-frame for agent
            backward_context = forward_context = 0

        self.forward_context = forward_context
        self.backward_context = backward_context
        self.min_main_agent_forward = min_main_agent_forward if min_main_agent_forward else forward_context
        self.min_main_agent_backward = min_main_agent_backward if min_main_agent_forward else backward_context

        # Extract all agents from agent dataset JSON for the appropriate split
        agent_groups = AgentDataset._extract_agent_groups_from_agent_dataset_json(
            agents_dataset_json, requested_agent_type, split=split
        )

        agent_metadata = AgentMetadata.from_agent_containers(
            agent_groups, requested_agent_type, requested_feature_types
        )
        name_to_id = agent_metadata.ontology_table[AGENT_TYPE_TO_ANNOTATION_TYPE[requested_agent_type]].name_to_id
        self.requested_main_agent_classes = tuple([name_to_id[atype] + 1 for atype in requested_main_agent_classes])

        # load agents data into agent container
        with Pool(cpu_count()) as proc:
            agent_groups = list(
                proc.map(
                    partial(
                        self._load_agents_data,
                        ontology_table=agent_metadata.ontology_table,
                        feature_ontology_table=agent_metadata.feature_ontology_table
                    ), agent_groups
                )
            )

        super().__init__(agent_metadata, agent_groups=agent_groups)

        # Record each agent's life time
        if batch_per_agent:
            self._build_agent_index()

    def _build_agent_index(self):
        """Build an index of agents grouped by instance id. This index is used to index dataset by agent track.

        Returns
        -------
        agent_item_index: list
            List of dataset items that contain index into.
            (main_agent_id, scene_idx, sample_idx_in_scene_start, sample_idx_in_scene_end, [datum_name ...]).
        """
        self.dataset_agent_index = defaultdict(list)
        logging.info('Building agent index, this will take a while.')
        for index in range(len(self.dataset_item_index)):
            scene_idx, sample_idx_in_scene, main_agent_id, datum_names = self.dataset_item_index[index]
            # Find the range of agents' trajectories
            main_agent_id_and_scene_idx = f'{str(scene_idx)}_{str(main_agent_id)}'
            if main_agent_id_and_scene_idx not in self.dataset_agent_index:
                self.dataset_agent_index[main_agent_id_and_scene_idx] = [-1, -1, float('inf'), -1, []]
            self.dataset_agent_index[main_agent_id_and_scene_idx] = [
                main_agent_id,
                scene_idx,
                min(sample_idx_in_scene, self.dataset_agent_index[main_agent_id_and_scene_idx][2]),
                # birth sample index
                max(sample_idx_in_scene, self.dataset_agent_index[main_agent_id_and_scene_idx][3]),
                # death sample item index
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
                self._item_index_for_scene, [(scene_idx, ) for scene_idx in range(len(self.agent_groups))]
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

        # build item index
        agent_group = self.agent_groups[scene_idx]
        num_samples = len(agent_group.sample_id_to_agent_snapshots)
        instance_id_to_trajectory = defaultdict(list)
        instance_id_to_segment_idx = defaultdict(int)

        for sample_id, agents_slice in agent_group.sample_id_to_agent_snapshots.items():
            for agent_snapshot in agents_slice:
                # Filter undesired agent types and attributes.
                if agent_snapshot.class_id not in self.requested_main_agent_classes:
                    continue
                # Make sure the track is sample-continuous
                instance_index_prefix = \
                    f'{str(scene_idx)}_{str(agent_snapshot.instance_id)}'
                segment_idx_start = instance_id_to_segment_idx[instance_index_prefix] \
                    if instance_index_prefix in instance_id_to_segment_idx else 0
                for segment_idx in range(segment_idx_start, num_samples):
                    instance_index_id = f'{instance_index_prefix}_{segment_idx}'
                    if instance_index_id in instance_id_to_trajectory and \
                            instance_id_to_trajectory[instance_index_id][-1][1] + 1 != sample_id:
                        continue
                    instance_id_to_trajectory[instance_index_id].append(
                        (scene_idx, sample_id, agent_snapshot.instance_id, self.selected_datums)
                    )
                    instance_id_to_segment_idx[instance_index_prefix] = segment_idx
                    break

        # Fiter unavailable items according to forward_context/backward_context for each agent.
        item_index = []
        trajectory_min_length = self.min_main_agent_backward + self.min_main_agent_forward + 1
        for id_ in instance_id_to_trajectory:

            trajectory_length = len(instance_id_to_trajectory[id_])
            # Make sure track length is sufficient
            if trajectory_length >= trajectory_min_length:
                first_sample_idx = instance_id_to_trajectory[id_][0][1]
                final_sample_idx = instance_id_to_trajectory[id_][-1][1]
                # Crop out valid samples as items
                beg = self.min_main_agent_backward \
                    if self.min_main_agent_backward + first_sample_idx > self.backward_context \
                    else self.backward_context
                end = trajectory_length - (
                    self.min_main_agent_forward
                    if self.min_main_agent_forward + final_sample_idx < num_samples else self.forward_context
                )
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
        datums: list of OrderedDict
            List of datum_data at (scene_idx, sample_idx_in_scene, datum_name). 
            Datum_names can be image, point_cloud, radar_point_cloud, etc.

        agents: AgentSnapshotList
            AgentSnapshotList in a frame.

        main_agent_id: int
            Instance ID of main agent.

        main_agent_idx: int 
            Index of main agent in AgentSlice.

        Returns a list of OrderedDict(s).
        Outer list corresponds to temporal ordering of samples. Each element is
        a OrderedDict(s) corresponding to synchronized datums and agents.
        """
        # return list()
        assert self.dataset_item_index is not None, ('Index is not built, select datums before getting elements.')

        if self.batch_per_agent:
            # Get dataset agent index
            main_agent_id, scene_idx, sample_idx_in_scene_start, sample_idx_in_scene_end, datum_names = \
                self.dataset_agent_index[index]
        else:
            # Get dataset item index
            scene_idx, sample_idx_in_scene, main_agent_id, datum_names = self.dataset_item_index[index]
            sample_idx_in_scene_start = sample_idx_in_scene - self.backward_context
            sample_idx_in_scene_end = sample_idx_in_scene + self.forward_context

        # All sensor data (including pose, point clouds and 3D annotations are
        # defined with respect to the sensor's reference frame captured at that
        # corresponding timestamp. In order to move to a locally consistent
        # reference frame, you will need to use the "pose" that specifies the
        # ego-pose of the sensor with respect to the local (L) frame (pose_LS).

        context_window = []
        # Iterate through context samples
        for qsample_idx_in_scene in range(sample_idx_in_scene_start, sample_idx_in_scene_end + 1):
            # Main agent index may be different along the samples.
            datums = []
            if len(datum_names) > 0:
                for datum_name in datum_names:
                    datum_data = self.synchronized_dataset.get_datum_data(scene_idx, qsample_idx_in_scene, datum_name)
                    datums.append(datum_data)
            agents_in_sample = self.agent_groups[scene_idx].agent_slice(qsample_idx_in_scene)
            instance_matched = [agent.instance_id == main_agent_id for agent in agents_in_sample]
            main_agent_idx_in_agents_slice = instance_matched.index(True) if any(instance_matched) else None
            synchronized_sample = OrderedDict({
                'datums': datums,
                'agents': agents_in_sample,
                'main_agent_id': main_agent_id,
                'main_agent_idx': main_agent_idx_in_agents_slice
            })
            context_window.append(synchronized_sample)
        return context_window


class AgentDatasetLite(BaseAgentDataset):
    """Dataset for agent-centric prediction or planning use cases. It provide two mode of accessing agent, by track
    and by frame. If 'batch_per_agent' is set true, then the data iterate per track, note, the length of the track
    could vary. Otherwise, the data iterate per frame and each sample contains all agents in the frame.

    Parameters
    ----------
    scene_dataset_json: str
        Full path to the scene dataset json holding collections of paths to scene json.

    agents_dataset_json: str
        Full path to the agent dataset json holding collections of paths to scene json.

    split: str, default: 'train'
        Split of dataset to read ("train" | "val" | "test" | "train_overfit").

    datum_names: list, default: None
        Select list of datum names for synchronization (see self.select_datums(datum_names)).

    requested_agent_type: tuple, default: None
        Tuple of agent type, i.e. ('agent_2d', 'agent_3d'). Only one type of agent can be requested.

    requested_main_agent_classes: tuple, default: 'car'
        Tuple of main agent types, i.e. ('car', 'pedestrian').
        The string should be the same as dataset_metadata.ontology_table.

    forward_context: int, default: 0
        Forward context in frames [T+1, ..., T+forward]

    backward_context: int, default: 0
        Backward context in frames [T-backward, ..., T-1]

    batch_per_agent: bool, default: False
        Include whole trajectory of an agent in each batch fetch.
        If True, backward_context = forward_context = 0 implicitly.

    use_diskcache: bool, default: True
        If True, cache ScenePb2 object using diskcache. If False, save the object in memory.
        NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of scenes.

    """
    def __init__(
        self,
        scene_dataset_json,
        agents_dataset_json,
        split='train',
        datum_names=None,
        requested_agent_type='agent_3d',
        requested_main_agent_classes=('car', ),
        requested_feature_types=None,
        requested_autolabels=None,
        forward_context=0,
        backward_context=0,
        batch_per_agent=False,
        use_diskcache=True,
    ):

        # Make sure requested agent type keys match protos
        if requested_agent_type is not None:
            assert requested_agent_type in AGENT_REGISTRY, "Invalid agent type requested!"
            self.requested_agent_type = requested_agent_type
        else:
            self.requested_agent_type = ()

        # Make sure requested feature type keys match protos
        if requested_feature_types is not None:
            assert all(
                requested_feature_type in ALL_FEATURE_TYPES for requested_feature_type in requested_feature_types
            ), "Invalid feature type requested!"
            self.requested_feature_types = requested_feature_types
        else:
            self.requested_feature_types = ()

        self.batch_per_agent = batch_per_agent
        datum_names = sorted(set(datum_names)) if datum_names else set([])
        self.selected_datums = [_d.lower() for _d in datum_names]
        if len(self.selected_datums) != 0:
            self.synchronized_dataset = SynchronizedSceneDataset(
                scene_dataset_json,
                split=split,
                backward_context=backward_context,
                requested_autolabels=requested_autolabels,
                forward_context=forward_context,
                datum_names=self.selected_datums,
                use_diskcache=use_diskcache
            )

        if batch_per_agent:  # fetch frame-by-frame for agent
            backward_context = forward_context = 0

        self.forward_context = forward_context
        self.backward_context = backward_context

        # Extract all agents from agent dataset JSON for the appropriate split
        agent_groups = AgentDataset._extract_agent_groups_from_agent_dataset_json(
            agents_dataset_json, requested_agent_type, split=split
        )

        agent_metadata = AgentMetadata.from_agent_containers(
            agent_groups, requested_agent_type, requested_feature_types
        )
        name_to_id = agent_metadata.ontology_table[AGENT_TYPE_TO_ANNOTATION_TYPE[requested_agent_type]].name_to_id
        self.requested_main_agent_classes = tuple([name_to_id[atype] + 1 for atype in requested_main_agent_classes])

        # load agents data into agent container
        with Pool(cpu_count()) as proc:
            agent_groups = list(
                proc.map(
                    partial(
                        self._load_agents_data,
                        ontology_table=agent_metadata.ontology_table,
                        feature_ontology_table=agent_metadata.feature_ontology_table
                    ), agent_groups
                )
            )

        super().__init__(agent_metadata, agent_groups=agent_groups)

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
        if self.batch_per_agent:
            with Pool(cpu_count()) as proc:
                item_index = proc.starmap(
                    partial(self._agent_index_for_scene, selected_datums=self.selected_datums),
                    [(scene_idx, agent_group) for scene_idx, agent_group in enumerate(self.agent_groups)]
                )
        else:
            with Pool(cpu_count()) as proc:
                item_index = proc.starmap(
                    partial(
                        self._item_index_for_scene,
                        backward_context=self.backward_context,
                        forward_context=self.forward_context,
                        selected_datums=self.selected_datums
                    ), [(scene_idx, agent_group) for scene_idx, agent_group in enumerate(self.agent_groups)]
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
    def _item_index_for_scene(scene_idx, agent_group, backward_context, forward_context, selected_datums):
        # Define a safe sample range given desired context
        sample_range = np.arange(backward_context, len(agent_group.sample_id_to_agent_snapshots) - forward_context)
        scene_item_index = [(scene_idx, sample_idx, selected_datums) for sample_idx in sample_range]

        return scene_item_index

    @staticmethod
    def _agent_index_for_scene(scene_idx, agent_group, selected_datums):
        scene_item_index = [
            (scene_idx, instance_id, selected_datums) for instance_id in agent_group.instance_id_to_agent_snapshots
        ]

        return scene_item_index

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.dataset_item_index)

    def __getitem__(self, index):
        """Get the dataset item at index.

        Parameters
        ----------
        index: int
            Index of item to get.

        Returns
        -------
        datums: list of OrderedDict
            List of datum_data at (scene_idx, sample_idx_in_scene, datum_name).
            Datum_names can be image, point_cloud, radar_point_cloud, etc.

        Agents: AgentSnapshotList
            A list of agent snapshots.

        Returns a list of OrderedDict(s).
        Outer list corresponds to temporal ordering of samples. Each element is
        a OrderedDict(s) corresponding to synchronized datums and agents.
        """
        # return list()
        assert self.dataset_item_index is not None, ('Index is not built, select datums before getting elements.')

        context_window = []
        if self.batch_per_agent:
            # Get dataset agent index
            scene_idx, instance_id, datum_names = self.dataset_item_index[index]
            track = self.agent_groups[scene_idx].agent_track(instance_id)
            ontology = track.ontology
            type_of_track = type(track)
            for agent_snapshot in track:
                qsample_idx_in_scene = agent_snapshot.sample_idx
                datums = []
                if len(datum_names) > 0:
                    for datum_name in datum_names:
                        datum_data = self.synchronized_dataset.get_datum_data(
                            scene_idx, qsample_idx_in_scene, datum_name
                        )
                        datums.append(datum_data)
                synchronized_sample = OrderedDict({
                    'datums': datums,
                    'agents': type_of_track(ontology, [agent_snapshot]),
                })
                context_window.append(synchronized_sample)
        else:
            # Get dataset item index
            scene_idx, sample_idx_in_scene, datum_names = self.dataset_item_index[index]
            sample_idx_in_scene_start = sample_idx_in_scene - self.backward_context
            sample_idx_in_scene_end = sample_idx_in_scene + self.forward_context

            # Iterate through context samples
            for qsample_idx_in_scene in range(sample_idx_in_scene_start, sample_idx_in_scene_end + 1):
                # Main agent index may be different along the samples.
                datums = []
                if len(datum_names) > 0:
                    for datum_name in datum_names:
                        datum_data = self.synchronized_dataset.get_datum_data(
                            scene_idx, qsample_idx_in_scene, datum_name
                        )
                        datums.append(datum_data)
                agents_in_sample = self.agent_groups[scene_idx].agent_slice(qsample_idx_in_scene)
                synchronized_sample = OrderedDict({
                    'datums': datums,
                    'agents': agents_in_sample,
                })
                context_window.append(synchronized_sample)
        return context_window


class AgentMetadata:
    """A Wrapper agents metadata class to support two entrypoints for agents dataset
    (reading from agents.json).

    Parameters
    ----------
    agent_groups: list[AgentContainer]
        List of AgentContainer objects to be included in the dataset.

    directory: str
        Directory of agent_dataset.

    feature_ontology_table: dict, default: None
        A dictionary mapping feature type key(s) to Ontology(s), i.e.:
        {
            "agent_2d": AgentFeatureOntology[<ontology_sha>],
            "agent_3d": AgentFeatureOntology[<ontology_sha>]
        }
    ontology_table: dict, default: None
        A dictionary mapping annotation key(s) to Ontology(s), i.e.:
        {
            "bounding_box_2d": BoundingBoxOntology[<ontology_sha>],
            "autolabel_model_1/bounding_box_2d": BoundingBoxOntology[<ontology_sha>],
            "semantic_segmentation_2d": SemanticSegmentationOntology[<ontology_sha>]
        }
    """
    def __init__(self, agent_groups, directory, feature_ontology_table=None, ontology_table=None):
        assert directory is not None, 'Dataset directory is required, and cannot be None.'
        self.agent_groups = agent_groups
        self.directory = directory
        self.feature_ontology_table = feature_ontology_table
        self.ontology_table = ontology_table

    @classmethod
    def from_agent_containers(cls, agent_containers, requested_agent_types=None, requested_feature_types=None):
        """Load DatasetMetadata from Scene Dataset JSON.

        Parameters
        ----------
        agent_containers: list of AgentContainer
            List of AgentContainer objects.

        requested_agent_types: List(str)
            List of agent types, such as ['agent_3d', 'agent_2d']

        requested_feature_types: List(str)
            List of feature types, such as ['parked_car', 'ego_intention']

        Raises
        ------
        Exception
            Raised if an ontology from an agent container is not in our ontology registry.
        """
        assert len(agent_containers), 'SceneContainers is empty.'
        requested_agent_types = [] if requested_agent_types is None else requested_agent_types

        if not requested_agent_types or not requested_feature_types:
            # Return empty ontology table
            return cls(
                agent_containers,
                directory=os.path.dirname(agent_containers[0].directory),
                feature_ontology_table={},
                ontology_table={}
            )
        feature_ontology_table = {}
        dataset_ontology_table = {}
        logging.info('Building ontology table.')
        st = time.time()

        # Determine scenes with unique ontologies based on the ontology file basename.
        unique_scenes = {
            os.path.basename(f): agent_container
            for agent_container in agent_containers
            for _, _, filenames in os.walk(os.path.join(agent_container.directory, FEATURE_ONTOLOGY_FOLDER))
            for f in filenames
        }
        # Parse through relevant scenes that have unique ontology keys.
        for _, agent_container in unique_scenes.items():

            for feature_ontology_key, ontology_file in agent_container.feature_ontology_files.items():

                # Look up ontology for specific annotation type
                if feature_ontology_key in FEATURE_ONTOLOGY_REGISTRY:

                    if feature_ontology_key not in requested_feature_types:
                        continue

                    feature_ontology_spec = FEATURE_ONTOLOGY_REGISTRY[feature_ontology_key]

                    # No need to add ontology-less tasks to the ontology table.
                    if feature_ontology_spec is None:
                        continue
                    # If ontology and key have not been added to the table, add it.
                    if feature_ontology_key not in feature_ontology_table:
                        feature_ontology_table[feature_ontology_key] = feature_ontology_spec.load(ontology_file)

                    # If we've already loaded an ontology for this feature type, make sure other scenes have the same
                    # ontology
                    else:
                        assert feature_ontology_table[feature_ontology_key] == feature_ontology_spec.load(
                            ontology_file
                        ), "Inconsistent ontology for key {}.".format(feature_ontology_key)

                # In case an ontology type is not implemented yet
                else:
                    raise Exception(f"Ontology for key {feature_ontology_key} not found in registry!")

            for ontology_key, ontology_file in agent_container.ontology_files.items():

                # Look up ontology for specific annotation type
                if ontology_key in ONTOLOGY_REGISTRY:

                    # Skip if we don't require this annotation/autolabel

                    if ANNOTATION_TYPE_TO_AGENT_TYPE[ontology_key] not in requested_agent_types:
                        continue

                    ontology_spec = ONTOLOGY_REGISTRY[ontology_key]

                    # No need to add ontology-less tasks to the ontology table.
                    if ontology_spec is None:
                        continue

                    # If ontology and key have not been added to the table, add it.
                    if ontology_key not in dataset_ontology_table:
                        dataset_ontology_table[ontology_key] = ontology_spec.load(ontology_file)

                    # If we've already loaded an ontology for this annotation type, make sure other scenes have the
                    # same ontology
                    else:
                        assert dataset_ontology_table[ontology_key] == ontology_spec.load(
                            ontology_file
                        ), "Inconsistent ontology for key {}.".format(ontology_key)

                # In case an ontology type is not implemented yet
                else:
                    raise Exception(f"Ontology for key {ontology_key} not found in registry!")

        logging.info(f'Ontology table built in {time.time() - st:.2f}s.')
        return cls(
            agent_containers,
            directory=os.path.dirname(agent_containers[0].directory),
            feature_ontology_table=feature_ontology_table,
            ontology_table=dataset_ontology_table
        )

    @staticmethod
    def get_dataset_splits(agents_json):
        """Get a list of splits in the agent_json.json.

        Parameters
        ----------
        agents_json: str
            Full path to the agents json holding agent metadata, agent splits.

        Returns
        -------
        agents_splits: list of str
            List of agents splits (train | val | test | train_overfit).

        """
        assert agents_json.endswith('.json'), 'Please provide a dataset.json file.'
        agents = open_pbobject(agents_json, AgentsPb2)
        return [
            dataset_pb2.DatasetSplit.DESCRIPTOR.values_by_number[split_index].name.lower()
            for split_index in agents.agents_splits
        ]
