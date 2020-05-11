"""Base dataset class compliant with the TRI-ML Data Governance Policy (DGP), which standardizes
TRI's data formats.

Please refer to `dgp/proto/dataset.proto` for the exact specifications of our DGP
and to `dgp/proto/annotations.proto` for the expected structure of
2D and 3D bounding box annotations.
"""
import glob
import hashlib
import json
import logging
import os
import time
from collections import ChainMap, OrderedDict, defaultdict
from functools import lru_cache, partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from dgp import (AUTOLABEL_FOLDER, CALIBRATION_FOLDER, ONTOLOGY_FOLDER,
                 SCENE_JSON_FILENAME)
from dgp.datasets import ANNOTATION_KEY_TO_TYPE_ID, ANNOTATION_TYPE_ID_TO_KEY
from dgp.proto import dataset_pb2
from dgp.proto.dataset_pb2 import DatasetMetadata as DatasetMetadataPb2
from dgp.proto.dataset_pb2 import SceneDataset as SceneDatasetPb2
from dgp.proto.sample_pb2 import SampleCalibration
from dgp.proto.scene_pb2 import Scene as ScenePb2
from dgp.utils import tqdm
from dgp.utils.aws import prefetch_lustre_files
from dgp.utils.camera import Camera
from dgp.utils.geometry import Pose
from dgp.utils.protobuf import open_ontology_pbobject, open_pbobject


class SceneContainer:
    """Scene container to support two-modes of operation for datasets.

    One mode allows the downstream task to train from datasets defined by their
    corresponding JSONs, while the other mode allows training from collections
    of scenes (each scene is fully described within a sub-directory with an
    associated scene.json file).

    This class also provides functionality for reinjecting autolabeled scenes into
    other scenes.
    """
    def __init__(self, scene, directory=None, autolabeled_scenes=None):
        """Initialize a scene with a scene object and optionally provide the
        directory containing the scene.json to gather additional information
        for directory-based dataset loading mode.

        Parameters
        ----------
        scene: scene_pb2
            Scene object containing data samples.

        directory: str, default: None
            Optional directory containing scene_<sha1>.json. This is used only when
            the directory-based dataset loading mode is desired.

        autolabeled_scenes: dict, default: None
            Dictionary mapping <autolabel_key> (defined as:`autolabel_model`/`annotation_key`) to autolabeled SceneContainer.
        """
        self.scene = scene
        self.directory = directory
        self.autolabeled_scenes = autolabeled_scenes
        logging.debug("Loading Scene-based dataset from {}".format(self.directory))

    @property
    def data(self):
        """Returns the scene data."""
        return self.scene.data

    @property
    def samples(self):
        """Returns the scene samples."""
        return self.scene.samples

    @property
    def ontology_files(self):
        """Returns the ontology files for a scene.

        Returns
        -------
        ontology_files: dict
            Maps annotation_key -> filename

            For example:
            filename = scene.ontology_files['bounding_box_2d']
        """
        # Load ontology files.
        ontology_files = {
            ANNOTATION_TYPE_ID_TO_KEY[ann_id]: os.path.join(self.directory, ONTOLOGY_FOLDER, "{}.json".format(f))
            for ann_id, f in self.scene.ontologies.items()
        }

        # Load autolabeled items in the scene.
        if self.autolabeled_scenes is not None:
            # Merge autolabeled scene ontologies into base scene ontology index.
            # Per autolabeled scene, we should only have a single ontology file.
            ontology_files.update({
                autolabel_key: list(autolabeled_scene.ontology_files.values())[0]
                for autolabel_key, autolabeled_scene in self.autolabeled_scenes.items()
            })
        return ontology_files

    @property
    def autolabels(self):
        """"Associate autolabels to datums. Iterate through datums, and if that datum has a corresponding autolabel,
        add it to the `autolabel` object. Example resulting autolabel map is:
        {
            <datum_hash>: {
                <autolabel_key>: <autolabeled_annotation>
                ...
            }
            ...
        }
        """
        autolabels = defaultdict(dict)
        if self.autolabeled_scenes is None:
            return autolabels

        # Load autolabeled items in the scene.
        for datum_idx, datum in enumerate(self.data):
            datum_type = datum.datum.WhichOneof('datum_oneof')
            for autolabel_key, autolabeled_scene in self.autolabeled_scenes.items():
                (_autolabel_model, annotation_key) = autolabel_key.split("/")
                requested_annotation_id = ANNOTATION_KEY_TO_TYPE_ID[annotation_key]
                annotations = getattr(autolabeled_scene.data[datum_idx].datum, datum_type).annotations
                if requested_annotation_id in annotations:
                    autolabels[datum.key][autolabel_key] = annotations[requested_annotation_id]
        return autolabels

    @property
    def calibration_files(self):
        """Returns the calibration index for a scene.

        Returns
        -------
        calibration_table: dict
            Maps (calibration_key, datum_key) -> (p_WS, Camera)

            For example:
            (p_WS, Camera) = self.calibration_table[(calibration_key, datum_name)]
        """
        if self.directory is None:
            return None
        assert os.path.exists(self.directory), 'Path {} does not exist'.format(self.directory)
        logging.info('Loading all scene calibrations in {}'.format(self.directory))
        calibration_files = glob.glob(os.path.join(self.directory, CALIBRATION_FOLDER, "*.json"))
        return calibration_files

    def __repr__(self):
        return "SceneContainer[{}][Samples: {}]".format(self.directory, len(self.samples))

    def __getstate__(self):
        """Custom pickle state getter for SceneContainer.

        Returns
        --------
        dict: Pickle-able dictionary object
        """
        return dict(
            scene=self.scene.SerializeToString(), directory=self.directory, autolabeled_scenes=self.autolabeled_scenes
        )

    def __setstate__(self, scene_object):
        """Custom pickle state loader for SceneContainer.
        Loads the state of the pickled object in the container.

        Parameters
        ----------
        scene_object: object
            Pickled scene object containing data samples.
        """
        scene = ScenePb2()
        scene.ParseFromString(scene_object['scene'])
        self.__init__(scene, directory=scene_object['directory'], autolabeled_scenes=scene_object['autolabeled_scenes'])

    def _build_datum_index(self):
        """Builds an of datum keys to index for datums in a scene.

        Returns
        -------
        datum_index: dict
            Datum index (datum_key -> index into self.data[index])
        """
        return {datum.key: idx for (idx, datum) in enumerate(self.data)}


class DatasetMetadata:
    """A Wrapper Dataset metadata class to support two entrypoints for datasets
    (reading from dataset.json OR from a scene_dataset.json).
    Aggregates statistics and onotology_table when construct DatasetMetadata
    object for SceneDataset.

    Parameters
    ----------
    metadata: dataset_pb2.DatasetMetadata
        Dataset metadata.

    directory: str
        Directory of dataset.

    ontology_table: dict, default: None
        A Dictionary containing annotation_type to ontology.

    """
    def __init__(self, metadata, directory, ontology_table=None):
        assert metadata is not None, 'Dataset metadata is required, and cannot be None.'
        assert directory is not None, 'Dataset directory is required, and cannot be None.'
        self.metadata = metadata
        self.directory = directory
        self.ontology_table = ontology_table

    @classmethod
    def from_scene_containers(cls, scene_containers):
        """Load DatasetMetadata from Scene Dataset JSON.

        Parameters
        ----------
        scene_containers: list of SceneContainer
            List of SceneContainer objects.
        """
        assert len(scene_containers), 'SceneContainers is empty.'
        logging.info("Computing SceneDataset statistics on-the-fly.")
        counts = np.array(
            [[scene_container.scene.statistics.image_statistics.count] for scene_container in scene_containers]
        )
        means = np.array([
            scene_container.scene.statistics.image_statistics.mean for scene_container in scene_containers
        ])
        stddevs = np.array([
            scene_container.scene.statistics.image_statistics.stddev for scene_container in scene_containers
        ])

        metadata = DatasetMetadataPb2()
        metadata.statistics.image_statistics.count = np.sum(counts)
        metadata.statistics.image_statistics.mean.extend(np.sum(np.multiply(means, counts), axis=0) / np.sum(counts))
        metadata.statistics.image_statistics.stddev.extend(
            np.sum(np.multiply(stddevs, counts), axis=0) / np.sum(counts)
        )
        # TODO: Compute point cloud statistics

        # Create SceneDataset level ontology table (annotation key to ontology).
        dataset_ontology_files = {}
        for scene_container in scene_containers:
            for ann_key, ontology_file in scene_container.ontology_files.items():
                if ann_key in dataset_ontology_files:
                    assert os.path.basename(dataset_ontology_files[ann_key]) == os.path.basename(ontology_file), \
                        "Inconsistent ontology for annotation_key {}.".format(ann_key)
                else:
                    dataset_ontology_files[ann_key] = ontology_file

        # TODO: Remove support for OntologyV1 once all datasets are registered
        # with new ontology v2 spec.
        dataset_ontology_table = {
            ann_key: open_ontology_pbobject(ontology_file)
            for ann_key, ontology_file in dataset_ontology_files.items()
        }

        return cls(
            metadata=metadata,
            directory=os.path.dirname(scene_containers[0].directory),
            ontology_table=dataset_ontology_table
        )

    @staticmethod
    def get_dataset_splits(dataset_json):
        """Get a list of splits in the dataset.json.

        Parameters
        ----------
        dataset_json: str
            Full path to the dataset json holding dataset metadata, ontology, and image and annotation paths.

        Returns
        -------
        dataset_splits: list of str
            List of dataset splits (train | val | test | train_overfit).

        """
        assert dataset_json.endswith('.json'), 'Please provide a dataset.json file.'
        dataset = open_pbobject(dataset_json, SceneDatasetPb2)
        return [
            dataset_pb2.DatasetSplit.DESCRIPTOR.values_by_number[split_index].name.lower()
            for split_index in dataset.scene_splits
        ]


class _BaseDataset(Dataset):
    """A base class representing a Dataset. Provides utilities for parsing and slicing
    DGP format datasets.

    Parameters
    ----------
    dataset_metadata: DatasetMetadata
        Dataset metadata object that encapsulates dataset-level metadata for
        both operating modes (scene or JSON).

    scenes: list[SceneContainer], default: None
        List of SceneContainer objects to be included in the dataset.

    calibration_table: dict, default: None
        Calibration table used for looking up sample-level calibration.

    datum_names: list, default: None
        List of datum names (str) to be considered in the dataset.

    requested_annotations: tuple[str], default: None
        Tuple of desired annotation keys, i.e. ('bounding_box_2d', 'bounding_box_3d'). Should match directory
        name containing annotations from dataset root.

    requested_autolabels: tuple[str], default: None
        Tuple of annotation keys similar to `requested_annotations`, but associated with a particular autolabeling model.
        Expected format is "<autolabel_model>/<annotation_key>"

    split: str, default: None
        Split of dataset to read ("train" | "val" | "test" | "train_overfit").
        If the split is None, the split type is not known and the dataset can
        be used for unsupervised / self-supervised learning.

    is_scene_dataset: bool, default: False
        Whether or not the dataset is constructed from a scene_dataset.json
    """
    AVAILABLE_DATUM_TYPES = ("image", "point_cloud")

    def __init__(
        self,
        dataset_metadata,
        scenes=None,
        calibration_table=None,
        datum_names=None,
        requested_annotations=None,
        requested_autolabels=None,
        split=None,
        is_scene_dataset=False
    ):
        super().__init__()

        # Dataset metadata
        self.dataset_metadata = dataset_metadata

        # Make sure requested annotation keys match protos
        if requested_annotations is not None:
            assert all(
                annotation in ANNOTATION_KEY_TO_TYPE_ID for annotation in requested_annotations
            ), "Invalid annotation key requested!"
            self.requested_annotations = requested_annotations
        else:
            self.requested_annotations = ()

        if requested_autolabels is not None:
            assert all(
                os.path.basename(autolabel) in ANNOTATION_KEY_TO_TYPE_ID for autolabel in requested_autolabels
            ), "Invalid autolabel annotation key requested!"
            self.requested_autolabels = requested_autolabels
        else:
            self.requested_autolabels = ()

        self.split = split

        # Scenes management
        self.scenes = scenes
        self.is_scene_dataset = is_scene_dataset

        # Calibration index
        # Index of sample-level calibration items that refer to the scene index,
        # sample_idx_in_scene and datum_idx_in_sample index.
        # Note: The construction of this index is delayed until and only if
        # calibration is required.
        # For example:
        # (p_WS, Camera) = self.calibration_table[(calibration_key, datum_name)]
        self.calibration_table = calibration_table

        # Map from (scene_idx, sample_idx_in_scene, datum_idx_in_sample) -> datum_idx_in_scene
        self.datum_index = self._build_datum_index()

        # Select subset of datums after scenes have been initialized.
        # If datum_names is None, select all datum(s) as items.
        # Otherwise, select a subset of specified datum_names
        self.select_datums(datum_names, reindex=False)

        # Dataset item index
        # This is the main index into the pytorch Dataset, where the index is
        # used to retrieve the item in the dataset via __getitem__.
        # Note: The construction of this index is delayed until the required
        # datum names or datum types are requested.
        # For example:
        # (scene_idx, sample_idx_in_scene, datum_idx_in_sample) = self.dataset_item_index[index]
        self.dataset_item_index = self._build_item_index()

        # Metadata item index
        # This index is maintained to keep the traceability of samples within a
        # dataset, and to additionally allow downstream sampling techniques
        # from sample-level or scene-level metadata.
        # For example:
        # sample_metadata = self.metadata_index[(scene_idx, sample_idx_scene)]
        self.metadata_index = self._build_metadata_index()

    def __hash__(self):
        """Hashes the dataset instance that is consistent across Python instances."""
        logging.debug('Hashing dataset with dataset directory, split and datum-index')
        # TODO: create unique identifier for both scene and .json datasets. Esp. split info of Scene Dataset.
        return int(
            hashlib.md5(
                self.dataset_metadata.directory.encode() + str(self.dataset_item_index).encode() +
                str(self.split).encode()
            ).hexdigest(), 16
        )

    def select_datums(self, datum_names=None, reindex=True):
        """Select a set of datums by name to be used in the dataset
        and rebuild dataset index.

        Parameters
        ----------
        datum_names: list
            List of datum names to be used for instance of dataset
        """
        # Select all available datums if none specified
        available_datums = set(self.list_available_datum_names_in_dataset())
        if datum_names is not None:
            datum_names = [datum_name.lower() for datum_name in datum_names]
            assert len(set(datum_names)
                       ) == len(datum_names), ('Select datum names uniquely, you provided the same datum name twice!')
            for datum_name in datum_names:
                assert datum_name in available_datums, ('Invalid datum name {}'.format(datum_name))
        else:
            datum_names = available_datums
        logging.info('Filtering datums in the dataset {}.'.format(datum_names))

        # Re-build dataset item index
        self.selected_datums = list(datum_names)
        if reindex:
            self.dataset_item_index = self._build_item_index()

    @property
    def image_mean(self):
        return np.array(self.dataset_metadata.metadata.statistics.image_statistics.mean, dtype=np.float32)

    @property
    def image_stddev(self):
        return np.array(self.dataset_metadata.metadata.statistics.image_statistics.stddev, dtype=np.float32)

    def __len__(self):
        """Return the length of the dataset."""
        raise NotImplementedError

    def __getitem__(self, index):
        """Get the dataset item at index."""
        raise NotImplementedError

    def get_scene_metadata(self, index):
        """Get scene-level metadata for the dataset item at index.

        Parameters
        ----------
        index: int
            Index of dataset item to get.

        Returns
        -------
        scene_metadata: OrderedDict
            Additional scene-level metadata for the dataset item at index.
            Note: This is used for traceability and sampling purposes.
        """
        assert self.metadata_index is not None, ('Metadata index is not built, provide metadata before accessing this.')
        sample_idx_in_scene = 0
        scene_idx, _, _ = self.dataset_item_index[index]
        return self.metadata_index[(scene_idx, sample_idx_in_scene)]

    def _build_item_index(self):
        """Builds an index of dataset items that refer to the scene index,
        sample index and datum_key index. This refers to a particular dataset
        split. __getitem__ indexes into this look up table.

        Returns
        -------
        item_index: list
            List of dataset items that contain index into
            (scene_idx, sample_idx_in_scene, datum_key_within_scene_idx).
        """
        raise NotImplementedError

    def _build_metadata_index(self, metadata=None):
        """Builds an index of metadata items that refer to the scene index,
        sample index index.

        Parameters
        ----------
        metadata: pd.Dataframe
            Dataframe used to join data across

        Returns
        -------
        metadata_index: dict
            Dictionary of metadata for tuple key (scene_idx,
            sample_idx_in_scene) returning a dictionary of additional metadata
            information for the requested sample.
        """
        # Only indexing scene-level metadata at the moment.
        metadata_index = {}
        for scene_idx, scene_container in enumerate(self.scenes):
            scene = scene_container.scene
            # Note (sudeep.pillai): Fow now we assume only the first sample
            # (indexed at 0) has metadata (scene-level metadata).
            SAMPLE_IDX_IN_SCENE = 0
            metadata_index[(scene_idx, SAMPLE_IDX_IN_SCENE)] = {
                'scene_index': scene_idx,
                'sample_index_in_scene': SAMPLE_IDX_IN_SCENE,
                'log_id': scene.log,
                'timestamp': scene.samples[SAMPLE_IDX_IN_SCENE].id.timestamp.ToMicroseconds(),
                'scene_name': scene.name,
                'scene_description': scene.description
            }

        # Create dataset with additional metadata
        dataset_df = pd.DataFrame(list(metadata_index.values()))
        if metadata is not None:
            # Note (sudeep.pillai): For now, we're only merging/joining on
            # log-level metadata. We pick the first item in the metadata dataframe
            # grouped by the log_id.
            assert 'scene_name' in metadata.columns, 'scene_name not in provided metadata'

            # Drop log_id before joining tables (since this column is redundant across tables)
            orig_length = len(dataset_df)
            dataset_df = pd.merge(dataset_df.drop(columns=['log_id']), metadata, on='scene_name')
            logging.info('Reducing dataset from {} to {}'.format(orig_length, len(dataset_df)))

            # Update metadata_index
            metadata_index = {}
            for _, row in dataset_df.iterrows():
                metadata_index[(row['scene_index'], row['sample_index_in_scene'])] = row.to_dict()
        return metadata_index

    def _build_datum_key_index(self):
        """Datum key index. Can be used to identify datums that exist across multiple scenes/samples.

        Recovering a datum from the key can be done as follows:
        ```
        datum_locations = datum_lookup_by_key[datum_key]
        for (scene_idx, sample_idx_in_scene, datum_idx_in_sample) in datum_locations:
            datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_idx_in_sample)
        ```

        Returns
        -------
        datum_lookup_by_key: defaultdict[list[tuple]]
            Returns a list because a single datum may be present in multiple samples, within or across scenes.
            {
                datum_key: list((scene_idx, sample_idx_in_scene, datum_idx_in_sample), ...)
            }

        """
        datum_key_index = defaultdict(list)
        for scene_idx, scene in enumerate(self.scenes):
            for sample_idx_in_scene, sample in enumerate(scene.samples):
                for datum_idx_in_sample, datum_key in enumerate(sample.datum_keys):
                    datum_key_index[datum_key].append((scene_idx, sample_idx_in_scene, datum_idx_in_sample))
        return datum_key_index

    def _build_datum_index(self):
        """Build global indexes scene data.

        Usage:
        ```
        datum_idx_in_scene = self.datum_index[scene_idx][sample_idx_in_scene][datum_idx_in_sample]
        datum = self.scenes[scene_idx].data[datum_idx_in_scene]
        ```

        Returns
        -------

        datum_index: dict[defaultdict[OrderedDict]]
            Returns a unique mapping from scene, sample, and datum to datum key.
            {
                scene_idx -> sample_idx_in_scene -> datum_idx_in_sample -> datum_key
        }
        """
        logging.info('Building datum index.')
        st = time.time()
        datum_index = {}
        for scene_idx, scene in tqdm(enumerate(self.scenes), total=len(self.scenes), desc="Building datum index."):
            datum_index[scene_idx] = self._build_datum_index_per_scene(scene)
        logging.info('Building datum index completed in {:.2f} s'.format(time.time() - st))
        return datum_index

    def _build_datum_index_per_scene(self, scene):
        """Helper for building parallel datum indices"""
        datum_index_per_scene = defaultdict(OrderedDict)
        datum_key_to_idx_in_scene = {datum.key: i for i, datum in enumerate(scene.data)}
        for sample_idx_in_scene, sample in enumerate(scene.samples):
            for datum_idx_in_sample, datum_key in enumerate(sample.datum_keys):
                datum_index_per_scene[sample_idx_in_scene][datum_idx_in_sample] = datum_key_to_idx_in_scene[datum_key]
        return datum_index_per_scene

    def get_scene(self, scene_idx):
        """Get scene given its scene index.

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        Returns
        -------
        scene: Scene
            Scene indexed at scene_idx.
        """
        return self.scenes[scene_idx].scene

    def get_sample(self, scene_idx, sample_idx_in_scene):
        """Get sample given its scene index and sample_idx_in_scene.

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        Returns
        -------
        sample: Sample
            Sample indexed at scene_idx and sample_idx_in_scene.
        """
        scene = self.get_scene(scene_idx)
        assert sample_idx_in_scene >= 0 and sample_idx_in_scene < len(scene.samples)
        return scene.samples[sample_idx_in_scene]

    def get_datum(self, scene_idx, sample_idx_in_scene, datum_idx_in_sample):
        """Get datum given its scene index, sample_idx_in_scene, and datum_idx_in_sample

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
        datum: Datum
            Datum indexed at scene_idx, sample_idx_in_scene and datum_idx_in_sample.
        """
        datum_idx_in_scene = self.datum_index[scene_idx][sample_idx_in_scene][datum_idx_in_sample]
        return self.scenes[scene_idx].data[datum_idx_in_scene]

    def get_lookup_from_datum_name_to_datum_index_in_sample(self, scene_idx, sample_idx_in_scene=0, datum_type=None):
        """"Build an index of available datums names within a sample to the index of that datum
        in the sample's datum keys

        Optionally, the datum names returned can be filtered by datum type.

        Parameters
        ----------
        scene_idx: int
            Index of scene of interest

        sample_idx_in_scene: int, default: 0
            Sample index within the scene from which we derive datum name <-> datum index. By
            default we use the first sample in the scene.

        datum_type: str, default: None
            One of ("image", "point_cloud"). If None, return all datum types.

        Returns
        -------
        datum_name_to_datum_index: dict
            (DatumId.name, datum_idx_in_sample) that are available for the scene.
        """
        datum_name_to_datum_idx = {}
        for datum_idx_in_sample, datum_idx_in_scene in self.datum_index[scene_idx][sample_idx_in_scene].items():
            datum = self.scenes[scene_idx].data[datum_idx_in_scene]
            assert datum.datum.WhichOneof("datum_oneof") in self.AVAILABLE_DATUM_TYPES
            if datum_type is None or datum.datum.WhichOneof("datum_oneof") in datum_type:
                datum_name_to_datum_idx[datum.id.name.lower()] = datum_idx_in_sample
        return datum_name_to_datum_idx

    def list_available_datum_names_in_scene(self, scene_idx, datum_type=None):
        """"Gets the list of available datums names within a scene.
        We assume that all samples in a scene have the same datums
        available.

        Optionally, the datum names returned can be filtered by datum type.

        Parameters
        ----------
        datum_type: str, default: None
            One of ("image", "point_cloud"). If None, return all datum types.

        Returns
        -------
        available_datum_names_in_scene: list
            List of DatumId.name that are available for the scene.
        """
        return list(
            self.get_lookup_from_datum_name_to_datum_index_in_sample(
                scene_idx, sample_idx_in_scene=0, datum_type=datum_type
            ).keys()
        )

    def list_datum_names_available_in_all_scenes(self, datum_type=None):
        """"Gets the set intersection of available datums names across all scenes.
        We assume that all samples in a scene have the same datums
        available. This should a subset of all available datums.

        Optionally, the datum names returned can be filtered by datum type.

        Parameters
        ----------
        datum_type: str, default: None
            Datum type, i.e. ("image", "point_cloud"). If None, return all datum types.

        Returns
        -------
        available_datum_names: list
            DatumId.name which are available across all scenes.
        """
        available_datum_names = set()
        for scene_idx, _scene in enumerate(self.scenes):
            available_datums_in_scene = self.list_available_datum_names_in_scene(scene_idx, datum_type=datum_type)

            if not available_datum_names:
                available_datum_names.update(available_datums_in_scene)
            else:
                available_datum_names.intersection_update(available_datums_in_scene)
        return list(available_datum_names)

    def list_available_datum_names_in_dataset(self, datum_type=None):
        """"Gets the set union of available datums names across all scenes.
        We assume that all samples in a scene have the same datums
        available. This should a subset of all available datums.

        Optionally, the datum names returned can be filtered by datum type.

        Parameters
        ----------
        datum_type: str, default: None
            Datum type, i.e. ("image", "point_cloud"). If None, return all datum types.

        Returns
        -------
        available_datum_names: list
            DatumId.name which are available across all scenes.
        """
        available_datum_names = set()
        for scene_idx, _scene in enumerate(self.scenes):
            available_datums_in_scene = self.list_available_datum_names_in_scene(scene_idx, datum_type=datum_type)
            available_datum_names.update(available_datums_in_scene)
        return list(available_datum_names)

    @lru_cache(maxsize=None)
    def get_datum_index_for_datum_name(self, scene_idx, sample_idx_in_scene, datum_name):
        """Get the datum_index for a sample given its datum name.

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        datum_name: str
            Datum name for desired datum.

        Returns
        -------
        datum_index: int
            Datum index within the sample.
        """
        # Get corresponding sample and datum_idx_in_sample
        for datum_idx_in_sample, datum_idx_in_scene in enumerate(self.datum_index[scene_idx][sample_idx_in_scene]):
            datum = self.scenes[scene_idx].data[datum_idx_in_scene]
            if datum.id.name.lower() == datum_name.lower():
                return datum_idx_in_sample

        assert 'Could not find datum {}'.format(datum_name)

    def get_scene_directory(self, scene_idx):
        """Get the directory in which data resides.

        Parameters
        ----------
        datum: Datum
            Datum container encapsulating some data.

        Returns
        -------
        scene_dir: str
            Directory of the corresponding datum.
        """
        # SceneDataset(s) will have an addition scene directory
        if self.is_scene_dataset:
            scene_dir = os.path.join(self.dataset_metadata.directory, self.scenes[scene_idx].directory)
        else:
            scene_dir = self.dataset_metadata.directory
        return scene_dir

    def load_datum(self, scene_idx, sample_idx_in_scene, datum_idx_in_sample):
        """Load a datum given a sample and a datum name

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
        datum: parsed datum type
            For different datums, we return different types.
            For image types, we return a PIL.Image
            For point cloud types, we return a numpy float64 array
        """
        # Get which scene datum comes from, otherwise use dataset directory
        scene_dir = self.get_scene_directory(scene_idx)
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_idx_in_sample)

        # Images are either stored in png/jpg and converted to RGB format.
        if datum.datum.HasField('image'):
            return Image.open(os.path.join(scene_dir, datum.datum.image.filename)).convert('RGB')
        # Point clouds are read from compressed numpy arrays from the 'data' key.
        elif datum.datum.HasField('point_cloud'):
            X = np.load(os.path.join(scene_dir, datum.datum.point_cloud.filename))['data']
            # If structured array, extract relevant fields.
            # Otherwise, load numpy array as-is, XYZI in (N, 4) format
            if X.dtype.fields is not None:
                # TODO: Check if relevant point_format fields are available.
                # TODO: Expose all point fields.
                # fields = datum.datum.point_cloud.point_format
                # PointCloudPb2.ChannelType.DESCRIPTOR.values_by_name['X'].number
                X = np.hstack([X['X'], X['Y'], X['Z'], X['INTENSITY']])
            return X
        else:
            raise TypeError("Datum has unknown type {}".format(datum))

    def load_datum_and_annotations(self, scene_idx, sample_idx_in_scene, datum_idx_in_sample):
        """Load a datum and return its annotations if available

        Parameters
        ----------
        datum: Datum
            Datum of type image, point cloud, etc..

        Returns
        -------
        datum: parsed datum type
            For different datums, we return different types.
            For image types, we return a PIL.Image
            For point cloud types, we return a numpy float64 array

        annotations: dict
            Dictionary mapping annotation key to annotation file for given datum.
        """
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_idx_in_sample)
        datum_type = datum.datum.WhichOneof('datum_oneof')
        datum_value = getattr(datum.datum, datum_type)
        annotations = dict(datum_value.annotations)
        return self.load_datum(scene_idx, sample_idx_in_scene, datum_idx_in_sample), {
            ANNOTATION_TYPE_ID_TO_KEY[ann_type]: v
            for ann_type, v in annotations.items()
        }

    def get_annotations(self, datum):
        """
        Parameters
        ----------
        datum: Datum
            Datum of type image, point cloud, etc..

        Returns
        -------
        annotations: annotations_pb2
            Annotation proto object corresponding to the datum.
        """
        datum_type = datum.datum.WhichOneof('datum_oneof')
        datum_value = getattr(datum.datum, datum_type)
        return datum_value.annotations

    def get_autolabels_for_datum(self, scene_idx, sample_idx_in_scene, datum_idx_in_sample):
        """Get autolabels associated with a datum if available

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
        autolabels: dict
            Map of <autolabel_model>/<annotation_key> : <annotation_path>. Returns empty dictionary
            if no autolabels exist for that datum.
        """
        datum_key = self.get_sample(scene_idx, sample_idx_in_scene).datum_keys[datum_idx_in_sample]
        return self.scenes[scene_idx].autolabels.get(datum_key, {})

    def get_camera_calibration(self, calibration_key, datum_name):
        """Get camera calibration given its calibration key and datum name.

        Parameters
        ----------
        calibration_key: str
            Calibration key.

        datum_name: str
            Datum name whose calibration is requested.

        Returns
        -------
        camera: Camera
            Calibrated camera with extrinsics/intrinsics set.
        """
        _p_WS, camera = self.calibration_table[(calibration_key, datum_name.lower())]
        return camera

    def get_sensor_extrinsics(self, calibration_key, datum_name):
        """Get sensor extrinsics given its calibration key and datum name.

        Parameters
        ----------
        calibration_key: str
            Calibration key.

        datum_name: str
            Datum name whose calibration is requested.

        Returns
        -------
        p_WS: Pose
            Extrinsics of sensor (S) with respect to the world (W)
        """
        p_WS, _ = self.calibration_table[(calibration_key, datum_name.lower())]
        return p_WS

    def get_datum_pose(self, datum):
        """Get the ego-pose associated with datum

        Parameters
        ----------
        datum: Datum
            Datum of type image, point cloud, etc..

        Returns
        -------
        datum_pose: Pose
            Pose object of datum's ego pose
        """

        if datum.datum.HasField('image'):
            datum_pose = Pose.from_pose_proto(datum.datum.image.pose)
        elif datum.datum.HasField('point_cloud'):
            datum_pose = Pose.from_pose_proto(datum.datum.point_cloud.pose)
        else:
            raise TypeError("Datum has unknown type {}".format(datum))

        # Empty pose -> identity
        if datum_pose.quat.norm == 0 and datum_pose.quat.magnitude == 0:
            datum_pose = Pose()
        return datum_pose

    def prefetch(self, datum_names=None):
        """Prefetch the dataset. Used for bootstrapping Lustre FSX datasets.

        Parameters
        ----------
        datums: list, default: None
            If None, prefetch raw datums files in the dataset split.
            If specific datum names are listed, prefetch only those datums.

        Notes
        -----
        Does not prefetch annotations. This is should be handled at the dataset specific
        level.
        """
        logging.info('Pre-fetching dataset.')
        files = []
        datum_names = [_d.lower() for _d in datum_names] if datum_names is not None else None

        for scene_idx in self.datum_index:
            scene_dir = self.get_scene_directory(scene_idx)
            for sample_idx_in_scene in self.datum_index[scene_idx]:
                for datum_idx_in_sample in self.datum_index[scene_idx][sample_idx_in_scene]:
                    datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_idx_in_sample)
                    if datum_names is None or datum.id.name in datum_names:
                        datum_type = datum.datum.WhichOneof('datum_oneof')
                        datum_value = getattr(datum.datum, datum_type)
                        files.append(os.path.join(scene_dir, datum_value.filename))
                        for annotation_file in datum_value.annotations.values():
                            files.append(os.path.join(scene_dir, annotation_file))
        prefetch_lustre_files(files)
        logging.info('Finished pre-fetching dataset.')


class BaseSceneDataset(_BaseDataset):
    """Main entry-point for dataset handling logic using scene dataset JSON as
    input. This class provides additional utilities for parsing directories of
    scenes, metadata and other relevant information.

    Note: Inherit from this class for self-supervised learning tasks where the
    default mode of operation is learning from a collection of scene
    directories.

    Parameters
    ----------
    scene_dataset_json: str
        Full path to the scene dataset json holding collections of paths to scene json.

    split: str, default: 'train'
        Split of dataset to read ("train" | "val" | "test" | "train_overfit").

    datum_names: list, default: None
        List of datum names (str) to be considered in the dataset.
    """
    def __init__(self, scene_dataset_json, split='train', datum_names=None, requested_annotations=None):
        logging.info('Loading scene dataset json in {}'.format(scene_dataset_json))

        # Extract scenes and build calibration table from scene_dataset_json
        scenes, calibration_table = self._extract_scenes_from_scene_dataset_json(scene_dataset_json, split=split)

        # Return BaseDataset with scenes built from <directory>/*/scene.json
        dataset_metadata = DatasetMetadata.from_scene_containers(scenes)
        super().__init__(
            dataset_metadata,
            scenes=scenes,
            calibration_table=calibration_table,
            datum_names=datum_names,
            requested_annotations=requested_annotations,
            is_scene_dataset=True
        )

    @staticmethod
    def _extract_scenes_from_scene_dataset_json(dataset_json, split='train', requested_autolabels=None):
        """Extract scene objects and calibration from the scene dataset JSON
        for the appropriate split.

        Parameters
        ----------
        dataset_json: str
            Path of the dataset.json

        split: str, default: 'train'
            Split of dataset to read ("train" | "val" | "test" | "train_overfit").

        requested_autolabels: tuple[str], default: None
            Tuple of strings of format "<autolabel_model>/<annotation_key>"

        Returns
        -------
        scene_containers: list
            List of SceneContainer objects.

        calibration_table: dict
            Calibration table used for looking up sample-level calibration.
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
        assert os.path.exists(dataset_json), 'Path {} does not exist'.format(dataset_json)
        logging.info("Loading dataset from {}, split={}".format(dataset_json, split))
        scene_dataset_root = os.path.dirname(dataset_json)
        scene_dataset = open_pbobject(dataset_json, SceneDatasetPb2)

        logging.info("Generating scenes for split={}".format(split))
        st = time.time()
        scene_jsons = [
            os.path.join(scene_dataset_root, _f) for _f in list(scene_dataset.scene_splits[split_enum].filenames)
        ]

        # Load all scene containers in parallel.
        with Pool(cpu_count()) as proc:
            scene_containers = list(
                tqdm(
                    proc.map(
                        partial(_get_scene_container, requested_autolabels=requested_autolabels), scene_jsons
                    ),
                    total=len(scene_jsons),
                    desc='Loading all scenes'
                )
            )
        logging.info("Scene generation completed in {:.2f}s".format(time.time() - st))

        # First get all calibration tables, then reduce to a single calibration table.
        logging.info("Build calibration table for all scenes.")
        calibration_files = [scene_container.calibration_files for scene_container in scene_containers]
        with Pool(cpu_count()) as proc:
            all_calibration_tables = list(
                tqdm(
                    proc.map(_get_scene_calibration_table, calibration_files),
                    total=len(calibration_files),
                    desc='Building calibration tables for all scenes'
                )
            )

        logging.info("Reduce calibration tables to unique keys.")
        calibration_table = dict(ChainMap(*all_calibration_tables))
        return scene_containers, calibration_table

    @staticmethod
    def _extract_scene_from_scene_json(scene_json, requested_autolabels=None):
        """Extract scene object and calibration from the scene JSON.

        Parameters
        ----------
        scene_json: str
            Path of the scene_<sha1>.json

        requested_autolabels: tuple[str], default: None
            Tuple of strings of format "<autolabel_model>/<annotation_key>"

        Returns
        -------
        scene_container: SceneContainer
            SceneContainer object.

        calibration_table: dict
            Calibration table used for looking up sample-level calibration.
        """
        # Load the scene and scene root
        assert os.path.exists(scene_json), 'Path {} does not exist'.format(scene_json)
        scene_container = _get_scene_container(scene_json, requested_autolabels)

        # Return calibration table for the scene
        calibration_table = _get_scene_calibration_table(scene_container.calibration_files)
        return scene_container, calibration_table


def _get_scene_container(scene_json, requested_autolabels=None):
    """Get a SceneContainer from the scene json. If autolabels are requested, inject them into the SceneContainer
    and merge ontologies

    Parameters
    ---------
    scene_json: str
        Path to scene JSON file

    requested_autolabels: tuple[str], default: None
        Tuple of strings of format "<autolabel_model>/<annotation_key>"

    Returns
    -------
    scene_container: SceneContainer
        SceneContainer, optionally with associated autolabels.
    """

    scene_dir = os.path.dirname(scene_json)
    if requested_autolabels is not None:
        logging.info("Loading autolabeled annotations.")
        autolabeled_scenes = _parse_autolabeled_scenes(scene_dir, requested_autolabels)
    else:
        autolabeled_scenes = None

    logging.info("Loading scene from {}".format(scene_json))
    scene_container = SceneContainer(
        open_pbobject(scene_json, ScenePb2), directory=scene_dir, autolabeled_scenes=autolabeled_scenes
    )
    return scene_container


def _get_scene_calibration_table(calibration_files):
    """Return a calibration object from filepaths"""
    calibration_table = {}
    for f in calibration_files:
        calibration = open_pbobject(f, SampleCalibration)
        calibration_key, _ = os.path.splitext(os.path.basename(f))
        for (name, intrinsic, extrinsic) in zip(calibration.names, calibration.intrinsics, calibration.extrinsics):
            p_WS = Pose.from_pose_proto(extrinsic)
            # If the intrinsics are invalid, i.e. fx = fy = 0, then it is
            # assumed to be a LIDAR sensor.
            cam = Camera.from_params(
                intrinsic.fx, intrinsic.fy, intrinsic.cx, intrinsic.cy, p_WS
            ) if intrinsic.fx > 0 and intrinsic.fy > 0 else None
            calibration_table[(calibration_key, name.lower())] = (p_WS, cam)
    return calibration_table


def _parse_autolabeled_scenes(scene_dir, requested_autolabels):
    """Parse autolabeled scene JSONs

    Parameters
    ----------
    scene_dir: str
        Path to root of scene directory

    requested_autolabels: tuple[str]
        Tuple of strings of format "<autolabel_model>/<annotation_key>"

    Returns
    -------
    autolabeled_scenes: dict
        Mapping from requested_autolabel key "<autolabel_model>/<annotation_key>" to SceneContainer
    """
    autolabeled_scenes = {}
    for autolabel in requested_autolabels:
        try:
            (autolabel_model, autolabel_type) = autolabel.split("/")
        except Exception:
            raise ValueError("Expected autolabel format <autolabel_model>/<annotation_key>, got {}".format(autolabel))
        autolabel_dir = os.path.join(scene_dir, AUTOLABEL_FOLDER, autolabel_model)
        autolabel_scene = os.path.join(autolabel_dir, SCENE_JSON_FILENAME)

        assert autolabel_type in ANNOTATION_KEY_TO_TYPE_ID, 'Autolabel type {} not valid'.format(autolabel_type)
        assert os.path.exists(autolabel_dir), 'Path to autolabels {} does not exist'.format(autolabel_dir)
        assert os.path.exists(autolabel_scene), 'Scene JSON expected but not found at {}'.format(autolabel_scene)
        autolabeled_scenes[autolabel] = SceneContainer(
            open_pbobject(autolabel_scene, ScenePb2), directory=autolabel_dir
        )
    return autolabeled_scenes
