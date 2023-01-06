# Copyright 2021-2022 Toyota Research Institute.  All rights reserved.
"""Base dataset class compliant with the TRI-ML Data Governance Policy (DGP), which standardizes
TRI's data formats.

Please refer to `dgp/proto/dataset.proto` for the exact specifications of our DGP
and to `dgp/proto/annotations.proto` for the expected structure for annotations.
"""
import glob
import hashlib
import logging
import os
import random
import time
from collections import ChainMap, OrderedDict, defaultdict
from functools import lru_cache, partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import xarray as xr
from diskcache import Cache
from PIL import Image

from dgp import (
    AUTOLABEL_FOLDER,
    CALIBRATION_FOLDER,
    DGP_CACHE_DIR,
    ONTOLOGY_FOLDER,
    SCENE_JSON_FILENAME,
)
from dgp.annotations import ANNOTATION_REGISTRY, ONTOLOGY_REGISTRY
from dgp.constants import ANNOTATION_KEY_TO_TYPE_ID, ANNOTATION_TYPE_ID_TO_KEY
from dgp.proto import dataset_pb2, radar_point_cloud_pb2
from dgp.proto.dataset_pb2 import DatasetMetadata as DatasetMetadataPb2
from dgp.proto.dataset_pb2 import SceneDataset as SceneDatasetPb2
from dgp.proto.sample_pb2 import SampleCalibration
from dgp.proto.scene_pb2 import Scene as ScenePb2
from dgp.utils.camera import Camera
from dgp.utils.pose import Pose
from dgp.utils.protobuf import open_pbobject

AVAILABLE_DATUM_TYPES = ("image", "point_cloud", "radar_point_cloud")
AVAILABLE_DISTORTION_PARAMS = (
    'k1',
    'k2',
    'k3',
    'k4',
    'k5',
    'k6',
    'p1',
    'p2',
    'alpha',
    'beta',
    'xi',
    's1',
    's2',
    's3',
    's4',
    'taux',
    'tauy',
    'fov',
    'fisheye',
    'w',
)


class SceneContainer:
    """Object-oriented container for assembling datasets from collections of scenes.
    Each scene is fully described within a sub-directory with an associated scene.json file.

    This class also provides functionality for reinjecting autolabeled scenes into other scenes.
    """
    random_str = ''.join([str(random.randint(0, 9)) for _ in range(5)])
    cache_suffix = os.environ.get('DGP_SCENE_CACHE_SUFFIX', random_str)
    cache_dir = os.path.join(DGP_CACHE_DIR, f'dgp_diskcache_{cache_suffix}')
    logging.debug(f'using {cache_dir} for dgp scene container disk cache')
    SCENE_CACHE = Cache(cache_dir)

    def __init__(
        self,
        scene_path,
        directory=None,
        autolabeled_scenes=None,
        is_datums_synchronized=False,
        use_diskcache=True,
        skip_missing_data=False
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

        skip_missing_data: bool, default: False
            If True, check for missing files and skip during datum index building.

        """
        self.scene_path = scene_path
        self.directory = directory
        self.autolabeled_scenes = autolabeled_scenes
        self.is_datums_synchronized = is_datums_synchronized
        self.use_diskcache = use_diskcache
        self._scene = None
        self.selected_datums = None
        self.requested_annotations = None
        self.requested_autolabels = None
        self.skip_missing_data = skip_missing_data
        logging.debug(f"Loading Scene-based dataset from {self.directory}")

    def select_datums(self, datum_names, requested_annotations=None, requested_autolabels=None):
        """Select a set of datums by name to be used in the scene.

        Parameters
        ----------
        datum_names: list
            List of datum names to be used for instance of dataset

        requested_annotations: tuple, optional
            Tuple of annotation types, i.e. ('bounding_box_2d', 'bounding_box_3d'). Should be equivalent
            to directory containing annotation from dataset root.
            Default: None.

        requested_autolabels: tuple[str], optional
            Tuple of annotation types similar to `requested_annotations`, but associated with a particular autolabeling model.
            Expected format is "<model_id>/<annotation_type>"
            Default: None.

        Raises
        ------
        ValueError
            Raised if datum_names is not a list or tuple or if it is a sequence with no elements.
        """
        if not isinstance(datum_names, (list, tuple)) or len(datum_names) == 0:
            raise ValueError('Provide a set of datum names as a list.')
        assert len(set(datum_names)
                   ) == len(datum_names), ('Select datum names uniquely, you provided the same datum name twice!')
        self.selected_datums = sorted(set(datum_names))

        self.requested_annotations = set(requested_annotations) if requested_annotations else ()
        assert all([annotation in ANNOTATION_KEY_TO_TYPE_ID for annotation in self.requested_annotations])

        self.requested_autolabels = set(requested_autolabels) if requested_autolabels is not None else ()
        logging.debug(f'Selected datums: {", ".join(datum_names)}')

    def get_datum(self, sample_idx_in_scene, datum_name):
        """Get datum given its sample_idx_in_scene and the datum name.

        Parameters
        ----------
        sample_idx_in_scene: int
            Index of the sample within the scene.

        datum_name: str
            Name of the datum within sample

        Returns
        -------
        datum: Datum
            Datum at sample_idx_in_scene and datum_name for the scene.
        """
        datum_idx_in_scene = self.datum_index[sample_idx_in_scene].loc[datum_name].data
        return self.data[datum_idx_in_scene]

    def get_sample(self, sample_idx_in_scene):
        """Get sample given its sample_idx_in_scene.

        NOTE: Some samples may be removed during indexing. These samples will
        NOT be returned by this function. An unmodified list of samples
        can be accessed via the `samples` property on each SceneContainer.

        Parameters
        ----------
        sample_idx_in_scene: int
            Index of the sample within the scene.

        Returns
        -------
        sample: Sample
            Sample indexed at sample_idx_in_scene for the scene.
        """
        assert sample_idx_in_scene >= 0 and sample_idx_in_scene < len(self.datum_index)
        return self.samples[self.datum_index.coords["samples"][sample_idx_in_scene].data]

    @lru_cache(maxsize=1024)
    def get_datum_type(self, datum_name):
        """Get datum type based on the datum name

        Parameters
        ----------
        datum_name: str
            The name of the datum to find a type for.
        """
        for datum in self.data:
            if datum.id.name.lower() == datum_name.lower():
                return datum.datum.WhichOneof('datum_oneof')
        return None

    @property
    @lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    def datum_names(self):
        """"Gets the list of datums names available within a scene."""
        logging.debug(f'Listing all available datum names in scene={self}.')
        return list(set([datum.id.name.lower() for datum in self.data]))

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
            if self.scene_path in SceneContainer.SCENE_CACHE:
                _scene = SceneContainer.SCENE_CACHE.get(self.scene_path)
                if _scene is not None:
                    return _scene
            _scene = open_pbobject(self.scene_path, ScenePb2)
            SceneContainer.SCENE_CACHE.add(self.scene_path, _scene)
            return _scene
        else:
            if self._scene is None:
                self._scene = open_pbobject(self.scene_path, ScenePb2)
            return self._scene

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

        # Note: When loading some experimental datasets from other sources such as Parallel Domain
        # It is possible to have annotation types that are not supported. For now we will throw
        # a warning and move on.
        # TODO: work with PD to have consistent protos
        for ann_id in self.scene.ontologies:
            if ann_id not in ANNOTATION_TYPE_ID_TO_KEY:
                logging.warning(
                    f'Found annotation type id {ann_id} however only the following ids are allowed {set(ANNOTATION_TYPE_ID_TO_KEY.keys())} are defined. Skipping...'
                )

        ontology_files = {
            ANNOTATION_TYPE_ID_TO_KEY[ann_id]: os.path.join(self.directory, ONTOLOGY_FOLDER, "{}.json".format(f))
            for ann_id, f in self.scene.ontologies.items()
            if ann_id in ANNOTATION_TYPE_ID_TO_KEY
        }

        # Load autolabeled items in the scene.
        if self.autolabeled_scenes is not None:
            # Merge autolabeled scene ontologies into base scene ontology index.
            # Per autolabeled scene, we should only have a single ontology file.

            # TODO: maybe remove this single ontology file constraint
            # no reason to have an autolabel be 1-1, the same scene can/should
            # be used for multiple annotations types because a single model can/should/will
            # output multiple predictions.

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
                (_, annotation_key) = os.path.split(autolabel_key)
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
        if not self.directory.startswith("s3://"):
            assert os.path.exists(self.directory), 'Path {} does not exist'.format(self.directory)
        logging.debug('Loading all scene calibrations in {}'.format(self.directory))
        calibration_files = glob.glob(os.path.join(self.directory, CALIBRATION_FOLDER, "*.json"))
        return calibration_files

    @property
    @lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    def annotation_index(self):
        """Build 2D boolean DataArray for annotations. Rows correspond to the `datum_idx_in_scene`
        and columns correspond to requested annotation types.

        For example:
        ```
        +----+-------------------+-------------------+-----+
        |    | "bounding_box_2d" | "bounding_box_3d" | ... |
        +----+-------------------+-------------------+-----+
        |  0 | False             | True              |     |
        |  1 | True              | True              |     |
        |  2 | False             | False             |     |
        | .. | ..                | ..                |     |
        +----+-------------------+-------------------+-----+
        ```
        Returns
        -------
        scene_annotation_index: xr.DataArray
            Boolean index of annotations for this scene
        """
        logging.debug(f"Building annotation index for scene {self.scene_path}")

        total_annotations = list(self.requested_annotations)
        if self.autolabeled_scenes is not None:
            total_annotations += list(self.autolabeled_scenes.keys())

        scene_annotation_index = xr.DataArray(
            np.zeros((len(self.data), len(total_annotations)), dtype=bool),
            dims=["datums", "annotations"],
            coords={"annotations": total_annotations}
        )

        requested_annotation_ids = [ANNOTATION_KEY_TO_TYPE_ID[ann] for ann in self.requested_annotations]

        autolabel_ids = {}
        if self.autolabeled_scenes is not None:
            autolabel_ids = {ann: ANNOTATION_KEY_TO_TYPE_ID[ann.split('/')[1]] for ann in self.autolabeled_scenes}

        for datum_idx_in_scene, datum in enumerate(self.data):
            datum_annotations = BaseDataset.get_annotations(datum).keys()
            has_annotation = [ann_id in datum_annotations for ann_id in requested_annotation_ids]

            # Find out which autolabels are present
            has_autolabel = []
            for key, ann_id in autolabel_ids.items():
                # TODO: make this robust to missing data
                datum = self.autolabeled_scenes[key].data[datum_idx_in_scene]
                datum_annotations = BaseDataset.get_annotations(datum).keys()
                if ann_id in datum_annotations:
                    has_autolabel.append(True)
                else:
                    has_autolabel.append(False)

            scene_annotation_index[datum_idx_in_scene] = has_annotation + has_autolabel

        logging.debug(f'Done building annotation index for scene {self.scene_path}')
        return scene_annotation_index

    @property
    @lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    def datum_index(self):
        """Build a multidimensional DataArray to represent a scene.
        Rows correspond to samples, and columns correspond to datums. The value at each location
        is the `datum_idx_in_scene`, which can be used to directly fetch the desired datum
        given a sample index and datum name.

        For example:
        ```
        +----+-------------+-------------+---------+-----+
        |    | "camera_01" | "camera_02" | "lidar" | ... |
        +----+-------------+-------------+---------+-----+
        |  0 |           0 |           1 |       2 |     |
        |  1 |           9 |          10 |      11 |     |
        |  2 |          18 |          19 |      20 |     |
        | .. |          .. |          .. |         |     |
        +----+-------------+-------------+---------+-----+
        ```

        Returns
        -------
        scene_datum_index: xr.DataArray
            2D index describing samples and datums
        """
        logging.debug(f'Building datum index for scene {self.scene_path}')
        scene_datum_index = xr.DataArray(
            -np.ones((len(self.samples), len(self.selected_datums)), dtype=np.int32),
            dims=['samples', 'datums'],
            coords={
                "samples": list(range(len(self.samples))),
                "datums": list(self.selected_datums)
            }
        )
        datum_key_to_idx_in_scene = {
            datum.key: (datum.id.name.lower(), idx)
            for idx, datum in enumerate(self.data)
            if datum.id.name.lower() in self.selected_datums
        }

        num_datums = 0
        bad_datums = 0
        for sample_idx_in_scene, sample in enumerate(self.scene.samples):
            for datum_key in sample.datum_keys:
                # If key is not available, the datum name is not among the requested datum_names.
                if datum_key not in datum_key_to_idx_in_scene:
                    continue
                datum_name, datum_idx_in_scene = datum_key_to_idx_in_scene[datum_key]

                # Skip missing data only if desired
                if self.skip_missing_data and not self.check_datum_file(datum_idx_in_scene):
                    bad_datums += 1
                    continue

                scene_datum_index[sample_idx_in_scene].loc[datum_name] = datum_idx_in_scene
                num_datums += 1

        assert len(datum_key_to_idx_in_scene) == bad_datums + num_datums, "Duplicated datum_key"

        if self.is_datums_synchronized:
            # Remove incomplete samples
            scene_datum_index = scene_datum_index[(scene_datum_index >= 0).all(axis=1)]
        logging.debug(f'Done building datum index for scene {self.scene_path}')
        return scene_datum_index

    @property
    @lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    def metadata_index(self):
        """Helper for building metadata index.

        TODO: Need to verify that the hashes are unique, and these lru-cached
        properties are consistent across disk-cached reads.
        """
        logging.debug(f'Building metadata index for scene {self.scene_path}')
        SAMPLE_IDX_IN_SCENE = 0
        scene = self.scene
        return {
            'log_id': scene.log,
            'timestamp': scene.samples[SAMPLE_IDX_IN_SCENE].id.timestamp.ToMicroseconds(),
            'scene_name': scene.name,
            'scene_description': scene.description
        }

    def __repr__(self):
        return "SceneContainer[{}][Samples: {}]".format(self.directory, len(self.samples))

    def check_files(self):
        """
        Checks if scene and calibration files exist
        Returns
        -------
        bool: True if scene and calibration files exist
        """

        if not os.path.exists(self.scene_path):
            logging.debug(f'Missing {self.scene_path}')
            return False

        for f in self.calibration_files:
            if not os.path.exists(f):
                logging.debug(f'Missing {f}')
                return False

        return True

    def check_datum_file(self, datum_idx_in_scene):
        """
        Checks if datum file exists
        Parameters
        ---------
        datum_idx_in_scene: int
            Index of datum in this scene

        Returns
        -------
        bool: True if datum file exists

        Raises
        ------
        TypeError
            Raised if the referenced datum has an unuspported type.
        """
        datum = self.data[datum_idx_in_scene]
        if datum.datum.HasField('image'):
            filename = datum.datum.image.filename
        elif datum.datum.HasField('point_cloud'):
            filename = datum.datum.point_cloud.filename
        elif datum.datum.HasField('file_datum'):
            filename = datum.datum.file_datum.datum.filename
        elif datum.datum.HasField('radar_point_cloud'):
            filename = datum.datum.radar_point_cloud.filename
        else:
            raise TypeError("Datum has unknown type {}".format(datum))

        filename = os.path.join(self.directory, filename)
        if os.path.exists(filename):
            return True
        else:
            logging.debug(f'Missing {filename}')
            return False

    def get_autolabels(self, sample_idx_in_scene, datum_name):
        """Get autolabels associated with a datum if available

        Parameters
        ----------
        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        datum_name: str
            Name of the datum within sample

        Returns
        -------
        autolabels: dict
            Map of <autolabel_model>/<annotation_key> : <annotation_path>. Returns empty dictionary
            if no autolabels exist for that datum.
        """
        autolabels = dict()
        if self.autolabeled_scenes is None:
            return autolabels

        datum_idx_in_scene = self.datum_index[sample_idx_in_scene].loc[datum_name].data
        datum = self.data[datum_idx_in_scene]

        datum_type = datum.datum.WhichOneof('datum_oneof')
        for autolabel_key, autolabeled_scene in self.autolabeled_scenes.items():
            (_, annotation_key) = os.path.split(autolabel_key)
            requested_annotation_id = ANNOTATION_KEY_TO_TYPE_ID[annotation_key]
            # TODO: Autolabels have to be valid on construction, we do not have a good way to make sure that the
            # same datum_idx is valid here.
            annotations = getattr(autolabeled_scene.data[datum_idx_in_scene].datum, datum_type).annotations
            if requested_annotation_id in annotations:
                autolabels[autolabel_key] = annotations[requested_annotation_id]

        return autolabels


class DatasetMetadata:
    """A Wrapper Dataset metadata class to support two entrypoints for datasets
    (reading from dataset.json OR from a scene_dataset.json).
    Aggregates statistics and onotology_table when construct DatasetMetadata
    object for SceneDataset.

    Parameters
    ----------
    scenes: list[SceneContainer]
        List of SceneContainer objects to be included in the dataset.

    directory: str
        Directory of dataset.

    ontology_table: dict, default: None
        A dictionary mapping annotation key(s) to Ontology(s), i.e.:
        {
            "bounding_box_2d": BoundingBoxOntology[<ontology_sha>],
            "autolabel_model_1/bounding_box_2d": BoundingBoxOntology[<ontology_sha>],
            "semantic_segmentation_2d": SemanticSegmentationOntology[<ontology_sha>]
        }

    """
    def __init__(self, scenes, directory, ontology_table=None):
        assert directory is not None, 'Dataset directory is required, and cannot be None.'
        self.scenes = scenes
        self.directory = directory
        self.ontology_table = ontology_table

    @property
    @lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    def metadata(self):
        st = time.time()
        logging.info('Computing SceneDataset statistics on-the-fly.')
        with Pool(cpu_count()) as proc:
            stats = np.array(list(proc.map(DatasetMetadata._get_scene_container_count_mean_stddev, self.scenes)))
            counts, means, stddevs = stats[:, 0].reshape(-1, 1), stats[:, 1:4], stats[:, 4:]

        # Populate dataset metadata.
        metadata = DatasetMetadataPb2()
        metadata.statistics.image_statistics.count = int(np.sum(counts))
        metadata.statistics.image_statistics.mean.extend(np.sum(np.multiply(means, counts), axis=0) / np.sum(counts))
        metadata.statistics.image_statistics.stddev.extend(
            np.sum(np.multiply(stddevs, counts), axis=0) / np.sum(counts)
        )
        logging.info(f'SceneDataset statistics computed in {time.time() - st:.2f}s.')
        return metadata

    @staticmethod
    def _get_scene_container_count_mean_stddev(scene_container):
        stats = np.hstack([
            scene_container.scene.statistics.image_statistics.count,
            scene_container.scene.statistics.image_statistics.mean,
            scene_container.scene.statistics.image_statistics.stddev
        ])
        if len(stats) != 7:
            return np.zeros(7)
        return stats

    @classmethod
    def from_scene_containers(
        cls,
        scene_containers,
        requested_annotations=None,
        requested_autolabels=None,
        autolabel_root=None,
    ):
        """Load DatasetMetadata from Scene Dataset JSON.

        Parameters
        ----------
        scene_containers: list of SceneContainer
            List of SceneContainer objects.

        requested_annotations: List(str)
            List of annotations, such as ['bounding_box_3d', 'bounding_box_2d']

        requested_autolabels: List(str)
            List of autolabels, such as['model_a/bounding_box_3d', 'model_a/bounding_box_2d']

        autolabel_root: str, optional
            Optional path to autolabel root directory. Default: None.

        Raises
        ------
        Exception
            Raised if an ontology in a scene has no corresponding implementation yet.
        """
        assert len(scene_containers), 'SceneContainers is empty.'
        requested_annotations = [] if requested_annotations is None else requested_annotations
        requested_autolabels = [] if requested_autolabels is None else requested_autolabels

        if not requested_annotations and not requested_autolabels:
            # Return empty ontology table
            return cls(scene_containers, directory=os.path.dirname(scene_containers[0].directory), ontology_table={})
        # For each annotation type, we enforce a consistent ontology across the
        # dataset (i.e. 2 different `bounding_box_3d` ontologies are not
        # permitted). However, an autolabel may support a different ontology
        # for the same annotation type. For example, the following
        # ontology_table is valid:
        # {
        #   "bounding_box_3d": BoundingBoxOntology,
        #   "bounding_box_2d": BoundingBoxOntology,
        #   "my_autolabel_model/bounding_box_3d": BoundingBoxOntology
        # }
        dataset_ontology_table = {}
        logging.info('Building ontology table.')
        st = time.time()

        # Determine scenes with unique ontologies based on the ontology file basename.
        # NOTE: We walk the directories instead of taking the ontology files directly from the scene proto purely for performance reasons,
        # a valid but slow method would be to call scene.ontology_files on every scene.
        unique_scenes = {
            os.path.basename(f): scene_container
            for scene_container in scene_containers
            for _, _, filenames in os.walk(os.path.join(scene_container.directory, ONTOLOGY_FOLDER)) for f in filenames
        }

        # Do the same as above but for the autolabel scenes
        # autolabels are in autolabel_root/<scene_dir>/autolabels/model_name/...>
        if requested_autolabels is not None and len(scene_containers) > 0:
            if autolabel_root is None:
                autolabel_root = os.path.dirname(scene_containers[0].directory)

            all_ontology_files = glob.glob(os.path.join(autolabel_root, '**', ONTOLOGY_FOLDER, '*'), recursive=True)

            # Extract just the scene directory from the files found above
            onotology_file_scene_dirs = [
                os.path.dirname(os.path.relpath(f, autolabel_root)).split('/')[0] for f in all_ontology_files
            ]

            # The contract with autolabels is that the scene directory name for the autolabel must match that of the non autolabel scene
            # therefore we can find the original scene container by the directory
            scene_dir_to_scene = {
                os.path.basename(scene_container.directory): scene_container
                for scene_container in scene_containers
            }
            unique_autolabel_scenes = {
                os.path.basename(f): scene_dir_to_scene[k]
                for f, k in zip(all_ontology_files, onotology_file_scene_dirs)
                if k in scene_dir_to_scene
            }
            unique_scenes.update(unique_autolabel_scenes)

        # Parse through relevant scenes that have unique ontology keys.
        for _, scene_container in unique_scenes.items():
            for ontology_key, ontology_file in scene_container.ontology_files.items():
                # Keys in `ontology_files` may correspond to autolabels,
                # so we strip those prefixes when instantiating `Ontology` objects
                _autolabel_model, annotation_key = os.path.split(ontology_key)

                # Look up ontology for specific annotation type
                if annotation_key in ONTOLOGY_REGISTRY:

                    # Skip if we don't require this annotation/autolabel
                    if _autolabel_model:
                        if ontology_key not in requested_autolabels:
                            continue
                    else:
                        if annotation_key not in requested_annotations:
                            continue

                    ontology_spec = ONTOLOGY_REGISTRY[annotation_key]

                    # No need to add ontology-less tasks to the ontology table.
                    if ontology_spec is None:
                        continue

                    # If ontology and key have not been added to the table, add it.
                    if ontology_key not in dataset_ontology_table:
                        dataset_ontology_table[ontology_key] = ontology_spec.load(ontology_file)

                    # If we've already loaded an ontology for this annotation type, make sure other scenes have the same ontology
                    else:
                        assert dataset_ontology_table[ontology_key] == ontology_spec.load(
                            ontology_file
                        ), "Inconsistent ontology for key {}.".format(ontology_key)

                # In case an ontology type is not implemented yet
                else:
                    raise Exception(f"Ontology for key {ontology_key} not found in registry!")

        logging.info(f'Ontology table built in {time.time() - st:.2f}s.')
        return cls(
            scene_containers,
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


class BaseDataset:
    """A base class representing a Dataset. Provides utilities for parsing and slicing
    DGP format datasets.

    Parameters
    ----------
    dataset_metadata: DatasetMetadata
        Dataset metadata object that encapsulates dataset-level metadata for
        both operating modes (scene or JSON).

    scenes: list[SceneContainer]
        List of SceneContainer objects to be included in the dataset.

    datum_names: list, default: None
        List of datum names (str) to be considered in the dataset.

    requested_annotations: tuple[str], default: None
        Tuple of desired annotation keys, i.e. ('bounding_box_2d', 'bounding_box_3d'). Should match directory
        name containing annotations from dataset root.

    requested_autolabels: tuple[str], default: None
        Tuple of annotation keys similar to `requested_annotations`, but associated with a particular autolabeling model.
        Expected format is "<autolabel_model>/<annotation_key>".

    split: str, default: None
        Split of dataset to read ("train" | "val" | "test" | "train_overfit").
        If the split is None, the split type is not known and the dataset can
        be used for unsupervised / self-supervised learning.

    autolabel_root: str, default: None
       Optional path to autolabel root directory.

    ignore_raw_datum: Optional[list[str]], default: None
        Optionally pass a list of datum types to skip loading their raw data (but still load their annotations). For
        example, ignore_raw_datum=['image'] will skip loading the image rgb data. The rgb key will be set to None.
        This is useful when only annotations or extrinsics are needed. Allowed values are any combination of
        'image','point_cloud','radar_point_cloud'
    """
    def __init__(
        self,
        dataset_metadata,
        scenes,
        datum_names,
        requested_annotations=None,
        requested_autolabels=None,
        split=None,
        autolabel_root=None,
        ignore_raw_datum=None,
    ):
        logging.info(f'Instantiating dataset with {len(scenes)} scenes.')
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
        # Check datum names, and normalize them.
        if not isinstance(datum_names, (tuple, list)):
            raise ValueError('Invalid datum_names provided, provide a list.')
        datum_names = [_d.lower() for _d in datum_names]

        # Select subset of datums after scenes have been initialized.
        # If datum_names is None, select all datum(s) as items.
        # Otherwise, select a subset of specified datum_names
        self._select_datums(
            datum_names, requested_annotations=requested_annotations, requested_autolabels=requested_autolabels
        )

        # Calibration index
        # >>> (p_WS, Camera) = self.calibration_table[(calibration_key, datum_name)]
        self.calibration_table = self._build_calibration_table(scenes)

        # Build index for each scene. See `SceneContainer.datum_index` for more details
        self.datum_index = self._build_datum_index()

        # Dataset item index
        # This is the main index into the pytorch Dataset, where the index is
        # used to retrieve the item in the dataset via __getitem__.
        self.dataset_item_index = self._build_item_index()

        # Metadata item index (now a cached property).
        # This index is maintained to keep the traceability of samples within a
        # dataset, and to additionally allow downstream sampling techniques
        # from sample-level or scene-level metadata.
        # For example:
        # >> sample_metadata = self.metadata_index[(scene_idx, sample_idx_scene)]
        self.additional_metadata = None
        self.autolabel_root = autolabel_root
        if self.autolabel_root is not None:
            self.autolabel_root = os.path.abspath(self.autolabel_root)

        # Ignore loading of raw rgb or point cloud data
        if ignore_raw_datum is None:
            ignore_raw_datum = []
        for datum_name in ignore_raw_datum:
            assert datum_name in AVAILABLE_DATUM_TYPES

        self.ignore_raw_datum = ignore_raw_datum

    @staticmethod
    def _extract_scenes_from_scene_dataset_json(
        dataset_json,
        split='train',
        requested_autolabels=None,
        is_datums_synchronized=False,
        use_diskcache=True,
        skip_missing_data=False,
        dataset_root=None,
        autolabel_root=None,
    ):
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

        is_datums_synchronized: bool, default: False
            If True, sample-level synchronization is required i.e. each sample must contain all datums specified in the requested
            `datum_names`, and all samples in this scene must contain the same number of datums.
            If False, sample-level synchronization is not required i.e. samples are allowed to have different sets of datums.

        use_diskcache: bool, default: True
            If True, cache ScenePb2 object using diskcache. If False, save the object in memory.
            NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of scenes in this scene dataset.

        skip_missing_data: bool, default: False
            If True, check for missing scene and datum files. These will be skipped during datum index building.

        dataset_root: str
            Optional path to dataset root folder. Useful if dataset scene json is not in the same directory as the rest of the data.

        autolabel_root: str, default: None
            Optional path to autolabel root directory.

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
        if not dataset_json.startswith("s3://"):
            assert os.path.exists(dataset_json), 'Path {} does not exist'.format(dataset_json)
        logging.info("Loading dataset from {}, split={}".format(dataset_json, split))

        scene_dataset_root = dataset_root
        if dataset_root is None:
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
                proc.map(
                    partial(
                        BaseDataset._get_scene_container,
                        requested_autolabels=requested_autolabels,
                        is_datums_synchronized=is_datums_synchronized,
                        use_diskcache=use_diskcache,
                        skip_missing_data=skip_missing_data,
                        autolabel_root=autolabel_root,
                    ), scene_jsons
                )
            )
        # Filter out scenes with missing data.
        full_length = len(scene_containers)
        if skip_missing_data:
            scene_containers = [scene for scene in scene_containers if scene.check_files()]
            num_bad_scenes = full_length - len(scene_containers)
            if num_bad_scenes > 0:
                logging.info(f'Skipping {num_bad_scenes}/{full_length} scenes with missing files.')

        logging.info("Scene generation completed in {:.2f}s".format(time.time() - st))
        return scene_containers

    @staticmethod
    def _extract_metadata_from_scene_dataset_json(dataset_json):
        """Extract dataset's existing metadata from the scene dataset JSON.

        Parameters
        ----------
        dataset_json: str
            Full path to the dataset json holding dataset metadata, ontology, and image and annotation paths.

        Returns
        -------
        dataset.metadata: dataset_pb2.DatasetMetadata
            Metadata existing in the dataset.

        """
        assert dataset_json.endswith('.json'), 'Please provide a dataset.json file.'
        dataset = open_pbobject(dataset_json, SceneDatasetPb2)
        return dataset.metadata

    @staticmethod
    def _extract_scene_from_scene_json(
        scene_json,
        requested_autolabels=None,
        is_datums_synchronized=False,
        use_diskcache=True,
        skip_missing_data=False,
        autolabel_root=None,
    ):
        """Extract scene object and calibration from a single scene JSON.
        If autolabels are requested, inject them into the SceneContainer and merge ontologies

        Parameters
        ----------
        scene_json: str
            Path of the scene_<sha1>.json

        See `_extract_scenes_from_scene_dataset_json` for other parameters.

        Returns
        -------
        scene_container: SceneContainer
            SceneContainer, optionally with associated autolabels.

        """
        # Load the scene and scene root
        assert os.path.exists(scene_json), 'Path {} does not exist'.format(scene_json)
        scene_container = BaseDataset._get_scene_container(
            scene_json,
            requested_autolabels,
            is_datums_synchronized,
            use_diskcache=use_diskcache,
            skip_missing_data=skip_missing_data,
            autolabel_root=autolabel_root,
        )
        return scene_container

    @staticmethod
    def _get_scene_container(
        scene_json,
        requested_autolabels=None,
        is_datums_synchronized=False,
        use_diskcache=True,
        skip_missing_data=False,
        autolabel_root=None,
    ):

        scene_dir = os.path.dirname(scene_json)

        if requested_autolabels is not None:
            logging.debug(f"Loading autolabeled annotations from {scene_dir}.")
            autolabeled_scenes = _parse_autolabeled_scenes(
                scene_dir,
                requested_autolabels,
                autolabel_root=autolabel_root,
                skip_missing_data=skip_missing_data,
                use_diskcache=use_diskcache,
            )
        else:
            autolabeled_scenes = None

        logging.debug(f"Loading scene from {scene_json}")
        scene_container = SceneContainer(
            scene_json,
            directory=scene_dir,
            autolabeled_scenes=autolabeled_scenes,
            is_datums_synchronized=is_datums_synchronized,
            use_diskcache=use_diskcache,
            skip_missing_data=skip_missing_data,
        )
        return scene_container

    @staticmethod
    def _build_calibration_table(scene_containers):
        """Build calibration table from scenes.

        Parameters
        ----------
        scene_containers: List[SceneContainer]
            List of scene containers to extract calibration tables.

        Returns
        -------
        calibration_table: dict, default: None
            Calibration table used for looking up sample-level calibration.
        """
        # First get all calibration tables, then reduce to a single calibration table.
        st = time.time()
        logging.info("Build calibration table for all scenes.")
        calibration_files = [scene_container.calibration_files for scene_container in scene_containers]
        with Pool(cpu_count()) as proc:
            all_calibration_tables = list(proc.map(BaseDataset._get_scene_calibration_table, calibration_files))
        logging.info("Reduce calibration tables to unique keys.")
        calibration_table = dict(ChainMap(*all_calibration_tables))
        logging.info(f"Calibration table built in {time.time() - st:.2f}s.")
        return calibration_table

    @staticmethod
    def _get_scene_calibration_table(calibration_files):
        """Return a calibration object from filepaths"""
        calibration_table = {}
        for f in calibration_files:
            calibration = open_pbobject(f, SampleCalibration)
            calibration_key, _ = os.path.splitext(os.path.basename(f))
            for (name, intrinsic, extrinsic) in zip(calibration.names, calibration.intrinsics, calibration.extrinsics):
                p_WS = Pose.load(extrinsic)
                # If the intrinsics are invalid, i.e. fx = fy = 0, then it is
                # assumed to be a LIDAR sensor.

                # TODO: this needs a refactor for two reasons,
                # 1. This uses a hardcoded list of distortion parameters, it should instead use the proto defintion
                # 2. We probably want the camera class to calculate and cache the remaps for undistortion

                # Get a dictionary of distortion parameters
                distortion = {}
                for k in AVAILABLE_DISTORTION_PARAMS:
                    if hasattr(intrinsic, k):
                        distortion[k] = getattr(intrinsic, k)

                cam = Camera.from_params(
                    intrinsic.fx, intrinsic.fy, intrinsic.cx, intrinsic.cy, p_WS, distortion=distortion
                ) if intrinsic.fx > 0 and intrinsic.fy > 0 else None
                calibration_table[(calibration_key, name.lower())] = (p_WS, cam)
        return calibration_table

    def _select_datums(self, datum_names=None, requested_annotations=None, requested_autolabels=None):
        """Select a set of datums by name to be used in the dataset
        and rebuild dataset index.

        Parameters
        ----------
        datum_names: list
            List of datum names to be used for instance of dataset

        requested_annotations: tuple, default: None
            Tuple of annotation types, i.e. ('bounding_box_2d', 'bounding_box_3d'). Should be equivalent
            to directory containing annotation from dataset root.

        requested_autolabels: tuple[str], default: None
            Tuple of annotation types similar to `requested_annotations`, but associated with a particular autolabeling model.
            Expected format is "<model_id>/<annotation_type>"
        """
        for scene in self.scenes:
            scene.select_datums(
                datum_names, requested_annotations=requested_annotations, requested_autolabels=requested_autolabels
            )

    @property
    def image_mean(self):
        return np.array(self.dataset_metadata.metadata.statistics.image_statistics.mean, dtype=np.float32)

    @property
    def image_stddev(self):
        return np.array(self.dataset_metadata.metadata.statistics.image_statistics.stddev, dtype=np.float32)

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
        # TODO: create unique identifier for both scene and .json datasets. Esp. split info of Scene Dataset.
        return int(
            hashlib.md5(
                self.dataset_metadata.directory.encode() + str(self.dataset_item_index).encode() +
                str(self.split).encode()
            ).hexdigest(), 16
        )

    def get_scene_metadata(self, scene_idx):
        """Get scene-level metadata for the scene index.

        Parameters
        ----------
        scene_idx: int
            Index of scene.

        Returns
        -------
        scene_metadata: OrderedDict
            Additional scene-level metadata for the dataset item at index.
            Note: This is used for traceability and sampling purposes.
        """
        sample_idx_in_scene = 0
        metadata = self.scenes[scene_idx].metadata_index
        metadata.update({'scene_index': scene_idx, 'sample_index_in_scene': sample_idx_in_scene})
        return metadata

    @property
    @lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
    def metadata_index(self):
        """Builds an index of metadata items that refer to the scene index,
        sample index index.

        Returns
        -------
        metadata_index: dict
            Dictionary of metadata for tuple key (scene_idx,
            sample_idx_in_scene) returning a dictionary of additional metadata
            information for the requested sample.
        """
        logging.info(f'Building metadata index for {len(self.scenes)} scenes, this will take a while.')
        st = time.time()

        # Build index on scene-level metadata.
        metadata_index = {}

        def _add_metadata_index_callback(scene_metadata_index):
            metadata_index[(scene_metadata_index['scene_index'], scene_metadata_index['sample_index_in_scene'])
                           ] = scene_metadata_index

        with Pool(cpu_count()) as proc:
            for scene_idx, scene in enumerate(self.scenes):  #pylint: disable=unused-variable
                proc.apply_async(
                    BaseDataset.get_scene_metadata, args=(scene_idx), callback=_add_metadata_index_callback
                )
            proc.close()
            proc.join()

        # Create dataset with additional metadata.
        dataset_df = pd.DataFrame(list(metadata_index.values()))
        if self.additional_metadata is not None:
            # Note: For now, we're only merging/joining on
            # log-level metadata. We pick the first item in the metadata dataframe
            # grouped by the log_id.
            assert 'scene_name' in dataset_df.columns, 'scene_name not in provided metadata'

            # Drop log_id before joining tables (since this column is redundant across tables)
            orig_length = len(dataset_df)
            dataset_df = pd.merge(dataset_df.drop(columns=['log_id']), self.additional_metadata, on='scene_name')
            logging.info('Reducing dataset from {} to {}'.format(orig_length, len(dataset_df)))

            # Update metadata_index
            metadata_index = {}
            for _, row in dataset_df.iterrows():
                metadata_index[(row['scene_index'], row['sample_index_in_scene'])] = row.to_dict()
        logging.info(f'Metadata index built in {time.time() - st:.2f}s.')
        return metadata_index

    @staticmethod
    def _datum_index_for_scene(scene):
        return scene.datum_index

    def _build_datum_index(self):
        """Build index of datums for each scene. See `SceneContainer.datum_index`
        for details.
        """
        logging.info('Building datum index, this will take a while..')
        st = time.time()
        with Pool(cpu_count()) as proc:
            datum_index = list(proc.map(BaseDataset._datum_index_for_scene, self.scenes))
        logging.info('Building datum index completed in {:.2f} s'.format(time.time() - st))
        return datum_index

    @staticmethod
    def _annotation_index_for_scene(scene):
        return scene.annotation_index

    def _build_annotation_index(self):
        """Build index of annotations for each scene. See `SceneContainer.annotation_index`
        for details.
        """
        logging.info('Building annotation index')
        st = time.time()
        with Pool(cpu_count()) as proc:
            annotation_index = list(proc.map(BaseDataset._datum_index_for_scene, self.scenes))
        logging.info('Building annotation index completed in {:.2f} s'.format(time.time() - st))
        return annotation_index

    @staticmethod
    def _datum_names_for_scene(scene):
        return scene.datum_names

    def list_datum_names_available_in_all_scenes(self):
        """"Gets the set union of available datums names across all scenes.
        We assume that all samples in a scene have the same datums
        available.

        Returns
        -------
        available_datum_names: list
            DatumId.name which are available across all scenes.
        """
        with Pool(cpu_count()) as proc:
            available_datum_names = set(proc.map(BaseDataset._datum_names_for_scene, self.scenes))
        return list(available_datum_names)

    def get_sample(self, scene_idx, sample_idx_in_scene):
        """Get sample given its scene index and sample_idx_in_scene.

        NOTE: Some samples may be removed during indexing. These samples will
        NOT be returned by this function. An unmodified list of samples
        can be accessed via the `samples` property on each SceneContainer.

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
        return self.scenes[scene_idx].get_sample(sample_idx_in_scene)

    def get_datum(self, scene_idx, sample_idx_in_scene, datum_name):
        """Get datum given its scene index, sample_idx_in_scene, and datum_name

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        datum_name: str
            Name of datum within simple

        Returns
        -------
        datum: Datum
            Datum indexed at scene_idx, sample_idx_in_scene with the given datum_name.
        """
        return self.scenes[scene_idx].get_datum(sample_idx_in_scene, datum_name)

    def load_datum(self, scene_idx, sample_idx_in_scene, datum_name):
        """Load a datum given a sample and a datum name

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        datum_name: str
            Name of the datum within sample

        Returns
        -------
        datum: parsed datum type
            For different datums, we return different types.
            For image types, we return a PIL.Image
            For point cloud types, we return a numpy float64 array

        Raises
        ------
        TypeError
            Raised if the datum type is unsupported.
        """
        # Get which scene datum comes from, otherwise use dataset directory
        scene_dir = self.scenes[scene_idx].directory
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_name)

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
        elif datum.datum.HasField('file_datum'):
            return datum.datum.file_datum.datum.filename
        elif datum.datum.HasField('radar_point_cloud'):
            X = np.load(os.path.join(scene_dir, datum.datum.radar_point_cloud.filename))['data']
            return X
        else:
            raise TypeError("Datum has unknown type {}".format(datum))

    def load_annotations(self, scene_idx, sample_idx_in_scene, datum_name):
        """Get annotations for a specified datum

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        datum_name: str
            Name of the datum within sample

        Returns
        -------
        annotations: dict
            Dictionary mapping annotation key to Annotation object for given annotation type.

        Raises
        ------
        Exception
            Raised if we cannot load an annotation type due to not finding an ontology for a requested annotation.
        """
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_name)
        annotations = self.get_annotations(datum)
        # Load annotations into useful Python objects
        annotation_objects = {}
        for annotation_key in self.requested_annotations:
            # Some datums on a sample may not have associated annotations. Return "None" for those datums
            annotation_path = annotations.get(ANNOTATION_KEY_TO_TYPE_ID[annotation_key], None)
            if annotation_path is None:
                annotation_objects[annotation_key] = None
                continue

            annotation_file = os.path.join(self.scenes[scene_idx].directory, annotation_path)
            if annotation_key in self.dataset_metadata.ontology_table:
                # Load annotation object with ontology
                annotation_objects[annotation_key] = ANNOTATION_REGISTRY[annotation_key].load(
                    annotation_file, self.dataset_metadata.ontology_table[annotation_key]
                )
            elif ONTOLOGY_REGISTRY[annotation_key] is None:
                # Some tasks have no associated ontology
                annotation_objects[annotation_key] = ANNOTATION_REGISTRY[annotation_key].load(annotation_file, None)
            else:
                raise Exception(f"Cannot load annotation type {annotation_key}, no ontology found!")

        # Now do the same but for autolabels
        autolabel_annotations = self.get_autolabels_for_datum(scene_idx, sample_idx_in_scene, datum_name)
        for autolabel_key in self.requested_autolabels:
            # Some datums in a sample may not have associated annotations. Return "None" for those datums
            model_name, annotation_key = autolabel_key.split('/')
            # NOTE: model_name should typically not be included in the annotation_path stored inside the scene.json
            # if for some reason it is, then it needs to be removed.

            annotation_path = autolabel_annotations.get(autolabel_key, None)

            if annotation_path is None:
                autolabel_annotations[autolabel_key] = None
                continue
            if self.autolabel_root is not None:
                annotation_file = os.path.join(
                    self.autolabel_root, os.path.basename(self.scenes[scene_idx].directory), AUTOLABEL_FOLDER,
                    model_name, annotation_path
                )
            else:
                annotation_file = os.path.join(
                    self.scenes[scene_idx].directory, AUTOLABEL_FOLDER, model_name, annotation_path
                )

            if not os.path.exists(annotation_file):
                logging.warning(f'missing {annotation_file}')
                autolabel_annotations[autolabel_key] = None
                continue

            if autolabel_key in self.dataset_metadata.ontology_table:
                # Load annotation object with ontology
                autolabel_annotations[autolabel_key] = ANNOTATION_REGISTRY[annotation_key].load(
                    annotation_file, self.dataset_metadata.ontology_table[autolabel_key]
                )

            elif ONTOLOGY_REGISTRY[annotation_key] is None:
                # Some tasks have no associated ontology
                autolabel_annotations[autolabel_key] = ANNOTATION_REGISTRY[annotation_key].load(annotation_file, None)
            else:
                raise Exception(f"Cannot load annotation type {autolabel_key}, no ontology found!")

        annotation_objects.update(autolabel_annotations)

        return annotation_objects

    @staticmethod
    def get_annotations(datum):
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

    def get_autolabels_for_datum(self, scene_idx, sample_idx_in_scene, datum_name):
        """Get autolabels associated with a datum if available

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        datum_name: str
            Name of the datum within sample

        Returns
        -------
        autolabels: dict
            Map of <autolabel_model>/<annotation_key> : <annotation_path>. Returns empty dictionary
            if no autolabels exist for that datum.
        """
        return self.scenes[scene_idx].get_autolabels(sample_idx_in_scene, datum_name)

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
        _, camera = self.calibration_table[(calibration_key, datum_name.lower())]
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

        Raises
        ------
        TypeError
            Raised if datum type is unsupported.
        """

        if datum.datum.HasField('image'):
            datum_pose = Pose.load(datum.datum.image.pose)
        elif datum.datum.HasField('point_cloud'):
            datum_pose = Pose.load(datum.datum.point_cloud.pose)
        else:
            raise TypeError("Datum has unknown type {}".format(datum))

        # Empty pose -> identity
        if datum_pose.quat.norm == 0 and datum_pose.quat.magnitude == 0:
            datum_pose = Pose()
        return datum_pose

    def get_image_from_datum(self, scene_idx, sample_idx_in_scene, datum_name):
        """Get the sample image data from image datum.

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        datum_name: str
            Name of the datum within sample

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
                Camera intrinsics if available.

            "extrinsics": Pose
                Camera extrinsics with respect to the vehicle frame, if available.

            "pose": Pose
                Pose of sensor with respect to the world/global/local frame
                (reference frame that is initialized at start-time). (i.e. this
                provides the ego-pose in `pose_WC`).

        annotations: dict
            Map from annotation key to annotation file for datum

        """
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_name)
        assert datum.datum.WhichOneof('datum_oneof') == 'image'

        # Get camera calibration and extrinsics for the datum name
        sample = self.get_sample(scene_idx, sample_idx_in_scene)

        if self.calibration_table:
            camera = self.get_camera_calibration(sample.calibration_key, datum.id.name)
            camera_intrinsics = camera.K
            camera_distortion = camera.D
            pose_VC = self.get_sensor_extrinsics(sample.calibration_key, datum.id.name)
            # Get ego-pose for the image (at the corresponding image timestamp t=Tc)
            pose_WC_Tc = Pose.load(datum.datum.image.pose)
        else:
            camera_intrinsics = None
            camera_distortion = None
            pose_VC = None
            pose_WC_Tc = Pose()

        # Populate data for image data
        image = None
        if 'image' not in self.ignore_raw_datum:
            image = self.load_datum(scene_idx, sample_idx_in_scene, datum_name)

        annotations = self.load_annotations(scene_idx, sample_idx_in_scene, datum_name)
        data = OrderedDict({
            "timestamp": datum.id.timestamp.ToMicroseconds(),
            "datum_name": datum.id.name,
            "rgb": image,
            "intrinsics": camera_intrinsics,
            "distortion": camera_distortion,
            "extrinsics": pose_VC,
            "pose": pose_WC_Tc
        })
        return data, annotations

    def get_point_cloud_from_datum(self, scene_idx, sample_idx_in_scene, datum_name):
        """Get the sample lidar data from point cloud datum.

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        datum_name: str
            Name of the datum within sample

        Returns
        -------
        data: OrderedDict

            "timestamp": int
                Timestamp of the lidar in microseconds.

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

        annotations: dict
            Map from annotation key to annotation file for datum
        """
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_name)
        assert datum.datum.WhichOneof('datum_oneof') == 'point_cloud'

        # Get sensor extrinsics for the datum name
        if self.calibration_table:
            pose_VS = self.get_sensor_extrinsics(
                self.get_sample(scene_idx, sample_idx_in_scene).calibration_key, datum.id.name
            )
            # Determine the ego-pose of the lidar sensor (S) with respect to the world (W) @ t=Ts
            pose_WS_Ts = Pose.load(datum.datum.point_cloud.pose)
        else:
            pose_VS = None
            pose_WS_Ts = Pose()

        # Points are described in the Lidar sensor (S) frame captured at the
        # corresponding lidar timestamp (Ts).
        # Points are in the lidar sensor's (S) frame.
        X_S = None
        if 'point_cloud' not in self.ignore_raw_datum:
            X_S = self.load_datum(scene_idx, sample_idx_in_scene, datum_name)

        annotations = self.load_annotations(scene_idx, sample_idx_in_scene, datum_name)
        data = OrderedDict({
            "timestamp": datum.id.timestamp.ToMicroseconds(),
            "datum_name": datum.id.name,
            "extrinsics": pose_VS,
            "pose": pose_WS_Ts,
            "point_cloud": X_S[:, :3] if X_S is not None else None,
            "extra_channels": X_S[:, 3:] if X_S is not None else None,
        })
        return data, annotations

    def get_radar_point_cloud_from_datum(self, scene_idx, sample_idx_in_scene, datum_name):
        """Get the sample radar data from radar point cloud datum.

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        datum_name: str
            Name of the datum within sample

        Returns
        -------
        data: OrderedDict

            "timestamp": int
                Timestamp of the radar point cloud in microseconds.

            "datum_name": str
                Sensor name from which the data was collected

            "extrinsics": Pose
                Sensor extrinsics with respect to the vehicle frame.

            "point_cloud": np.ndarray (N x 3)
                Point cloud in the local/world (L) frame returning X, Y and Z
                coordinates. The local frame is consistent across multiple
                timesteps in a scene.

            "velocity": np.ndarray(N x 3)
                Velocity vectors in sensor frame.

            "covariance": np.ndarray(N x 3 x 3)
                Covariance matrix of point positions in sensor frame.

            "extra_channels": np.ndarray (N x M)
                Remaining channels from radar, rcs_dbm, probability, sensor_id etc

            "pose": Pose
                Pose of sensor with respect to the world/global/local frame
                (reference frame that is initialized at start-time). (i.e. this
                provides the ego-pose in `pose_WS` where S refers to the point
                cloud sensor (S)).

        annotations: dict
            Map from annotation key to annotation file for datum
        """
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_name)
        assert datum.datum.WhichOneof('datum_oneof') == 'radar_point_cloud'

        # Get sensor extrinsics for the datum name
        if self.calibration_table:
            pose_VS = self.get_sensor_extrinsics(
                self.get_sample(scene_idx, sample_idx_in_scene).calibration_key, datum.id.name
            )
            # Determine the ego-pose of the lidar sensor (S) with respect to the world (W) @ t=Ts
            pose_WS_Ts = Pose.load(datum.datum.radar_point_cloud.pose)
        else:
            pose_VS = None
            pose_WS_Ts = Pose()

        # Points are described in the Radar sensor (S) frame captured at the
        # corresponding radar timestamp (Ts).
        # Points are in the radar sensor's (S) frame.
        X_S = None
        if 'radar_point_cloud' not in self.ignore_raw_datum:
            X_S = self.load_datum(scene_idx, sample_idx_in_scene, datum_name)

        data = OrderedDict({
            "timestamp": datum.id.timestamp.ToMicroseconds(),
            "datum_name": datum.id.name,
            "extrinsics": pose_VS,
            "pose": pose_WS_Ts,
        })

        # This is the channel format that we saved the proto with.
        # We need to use this to lookup which columns we want, and find their position
        # to index into the numpy array with.
        channels = list(datum.datum.radar_point_cloud.point_format)

        def fetch_channel_index_if_available(channel_ids, channel_format):  # pylint: disable=missing-any-param-doc
            """ Helper function, returns index into channel_format that map to the requested chanels_ids.
               If not all of the channel ids are available in the channel_format, then return None.
            """
            try:
                idx = [channel_format.index(i) for i in channel_ids]
                return idx
            except ValueError:
                return None

        # Keep track of the columns we have explicitly asked for, what is left over will become "extra_channels"
        fetched_columns = []

        xyz_ids = [
            radar_point_cloud_pb2.RadarPointCloud.X, radar_point_cloud_pb2.RadarPointCloud.Y,
            radar_point_cloud_pb2.RadarPointCloud.Z
        ]
        xyz_idx = fetch_channel_index_if_available(xyz_ids, channels)
        if xyz_idx:
            data['point_cloud'] = X_S[:, xyz_idx] if X_S is not None else None
            fetched_columns.extend(xyz_idx)

        vel_ids = [
            radar_point_cloud_pb2.RadarPointCloud.V_X, radar_point_cloud_pb2.RadarPointCloud.V_Y,
            radar_point_cloud_pb2.RadarPointCloud.V_Z
        ]
        vel_idx = fetch_channel_index_if_available(vel_ids, channels)
        if vel_idx:
            data['velocity'] = X_S[:, vel_idx] if X_S is not None else None
            fetched_columns.extend(vel_idx)

        cov_ids = [ radar_point_cloud_pb2.RadarPointCloud.COV_XX, radar_point_cloud_pb2.RadarPointCloud.COV_XY, radar_point_cloud_pb2.RadarPointCloud.COV_XZ, \
                    radar_point_cloud_pb2.RadarPointCloud.COV_YX, radar_point_cloud_pb2.RadarPointCloud.COV_YY, radar_point_cloud_pb2.RadarPointCloud.COV_YZ, \
                    radar_point_cloud_pb2.RadarPointCloud.COV_ZX, radar_point_cloud_pb2.RadarPointCloud.COV_ZY, radar_point_cloud_pb2.RadarPointCloud.COV_ZZ]
        cov_idx = fetch_channel_index_if_available(cov_ids, channels)
        if cov_idx:
            data['covariance'] = X_S[:, cov_idx].reshape(-1, 3, 3) if X_S is not None else None
            fetched_columns.extend(cov_idx)

        # Dump whatever we did not explictly ask for into "extra_channels"
        data['extra_channels'] = np.delete(X_S, fetched_columns, axis=1) if X_S is not None else None

        annotations = self.load_annotations(scene_idx, sample_idx_in_scene, datum_name)

        return data, annotations

    def get_file_meta_from_datum(self, scene_idx, sample_idx_in_scene, datum_name):
        """Get the sample file info from file datum.

        Parameters
        ----------
        scene_idx: int
            Index of the scene.

        sample_idx_in_scene: int
            Index of the sample within the scene at scene_idx.

        datum_name: str
            Name of the datum within sample

        Returns
        -------
        data: OrderedDict

            "timestamp": int
                Timestamp of the image in microseconds.

            "datum_name": str
                Sensor name from which the data was collected

            "filename": str
                File name associate to the file datum.

        annotations: dict
            Map from annotation key to annotation file for datum
        """
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_name)
        assert datum.datum.WhichOneof('datum_oneof') == 'file_datum'

        annotations = {key: datum.datum.file_datum.annotations[key].filename \
                       for key in datum.datum.file_datum.annotations}
        data = OrderedDict({
            "timestamp": datum.id.timestamp.ToMicroseconds(),
            "datum_name": datum.id.name,
            "filename": self.load_datum(scene_idx, sample_idx_in_scene, datum_name),
        })
        return data, annotations


def _parse_autolabeled_scenes(
    scene_dir,
    requested_autolabels,
    autolabel_root=None,
    skip_missing_data=False,
    use_diskcache=False,
):
    """Parse autolabeled scene JSONs

    Parameters
    ----------
    scene_dir: str
        Path to root of scene directory

    requested_autolabels: tuple[str]
        Tuple of strings of format "<autolabel_model>/<annotation_key>"

    autolabel_root: str, default: None
        Path to autolabel root folder

    skip_missing_data: bool, defaul: False
        If true, skip over missing autolabel scenes

    use_diskcache: bool, default: False
        If diskcache should be used for autolabels

    Returns
    -------
    autolabeled_scenes: dict
        Mapping from requested_autolabel key "<autolabel_model>/<annotation_key>" to SceneContainer

    Raises
    ------
    ValueError
        Raised if we encounter an invalid autolabel format in requested_autolabels.
    """
    autolabeled_scenes = {}
    for autolabel in requested_autolabels:
        try:
            (autolabel_model, autolabel_type) = autolabel.split("/")
        except Exception as e:
            raise ValueError(
                "Expected autolabel format <autolabel_model>/<annotation_key>, got {}".format(autolabel)
            ) from e
        if autolabel_root is not None:
            autolabel_dir = os.path.join(
                os.path.abspath(autolabel_root), os.path.basename(scene_dir), AUTOLABEL_FOLDER, autolabel_model
            )
        else:
            autolabel_dir = os.path.join(scene_dir, AUTOLABEL_FOLDER, autolabel_model)
        autolabel_scene = os.path.join(autolabel_dir, SCENE_JSON_FILENAME)

        assert autolabel_type in ANNOTATION_KEY_TO_TYPE_ID, 'Autolabel type {} not valid'.format(autolabel_type)

        if skip_missing_data:
            if not (os.path.exists(autolabel_dir) and os.path.exists(autolabel_scene)):
                logging.debug(f'skipping autolabel {autolabel_dir}')
                continue
        else:
            assert os.path.exists(autolabel_dir), 'Path to autolabels {} does not exist'.format(autolabel_dir)
            assert os.path.exists(autolabel_scene), 'Scene JSON expected but not found at {}'.format(autolabel_scene)

        autolabeled_scenes[autolabel] = SceneContainer(
            autolabel_scene, directory=autolabel_dir, use_diskcache=use_diskcache
        )
    return autolabeled_scenes
