# Copyright 2021-2022 Toyota Research Institute.  All rights reserved.
"""Dataset for handling frame-level (unordered) for unsupervised, self-supervised and supervised tasks.
This dataset is compliant with the TRI-ML Dataset Governance Policy (DGP).

Please refer to `dgp/proto/dataset.proto` for the exact specifications of our dgp.
"""
import itertools
import logging
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import xarray as xr

from dgp.constants import (
    ALL_ANNOTATION_TYPES,
    DATUM_TYPE_TO_SUPPORTED_ANNOTATION_TYPE,
)
from dgp.datasets import BaseDataset, DatasetMetadata

SUPPORTED_ANNOTATIONS_TABLE = xr.DataArray(
    np.zeros((len(DATUM_TYPE_TO_SUPPORTED_ANNOTATION_TYPE), len(ALL_ANNOTATION_TYPES)), dtype=bool),
    dims=["datum_types", "annotations"],
    coords={
        "datum_types": list(DATUM_TYPE_TO_SUPPORTED_ANNOTATION_TYPE),
        "annotations": list(ALL_ANNOTATION_TYPES)
    }
)
for datum_type_, annotations_ in DATUM_TYPE_TO_SUPPORTED_ANNOTATION_TYPE.items():
    for annotation_ in annotations_:
        SUPPORTED_ANNOTATIONS_TABLE.loc[datum_type_, annotations_] = True


class _FrameDataset(BaseDataset):
    """Single frame dataset.
    See BaseDataset for input parameters for the parent class.

    Parameters
    ----------
    dataset_metadata: DatasetMetadata
        Dataset metadata, populated from scene dataset JSON

    scenes: list[SceneContainer], default: None
        List of SceneContainers parsed from scene dataset JSON

    datum_names: list, default: None
        Select list of datum names for index (see self.select_datums(datum_names)).

    requested_annotations: tuple, default: None
        Tuple of annotation types, i.e. ('bounding_box_2d', 'bounding_box_3d'). Should be equivalent
        to directory containing annotation from dataset root.

    requested_autolabels: tuple[str], default: None
        Tuple of annotation types similar to `requested_annotations`, but associated with a particular autolabeling model.
        Expected format is "<model_id>/<annotation_type>"

    only_annotated_datums: bool, default: False
        If True, only datums with annotations matching the requested annotation types are returned.
    """
    def __init__(
        self,
        dataset_metadata,
        scenes=None,
        datum_names=None,
        requested_annotations=None,
        requested_autolabels=None,
        only_annotated_datums=False
    ):
        self.only_annotated_datums = only_annotated_datums if requested_annotations else False
        super().__init__(
            dataset_metadata,
            scenes=scenes,
            datum_names=datum_names,
            requested_annotations=requested_annotations,
            requested_autolabels=requested_autolabels
        )

    def _build_item_index(self):
        """Builds an index of dataset items that refer to the scene index,
        sample index and datum_within_scene index. This refers to a particular dataset
        split. __getitem__ indexes into this look up table.

        Returns
        -------
        item_index: list
            List of dataset items that contain index into
            [(scene_idx, sample_within_scene_idx, datum_idx_in_sample), ...].
        """
        logging.info(
            f'{self.__class__.__name__} :: Building item index for {len(self.scenes)} scenes, this will take a while.'
        )
        st = time.time()
        # Fetch the item index per-scene based on the selected datums.
        with Pool(cpu_count()) as proc:
            item_index = proc.starmap(
                partial(_FrameDataset._item_index_for_scene, only_annotated_datums=self.only_annotated_datums),
                [(scene_idx, scene) for scene_idx, scene in enumerate(self.scenes)]
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
    def _item_index_for_scene(scene_idx, scene, only_annotated_datums):
        st = time.time()
        logging.debug(f'Indexing scene items for {scene.scene_path}')
        scene_item_index = []
        for datum_name in scene.selected_datums:
            samples_for_datum_name = scene.datum_index.loc[:, datum_name]
            # Ignore samples where the selected datum is missing.
            valid_sample_indices = samples_for_datum_name[samples_for_datum_name >= 0].coords["samples"]
            # Skip if there's no sample in the scene contain `datum_name`.
            if valid_sample_indices.size == 0:
                continue
            if not only_annotated_datums:
                # Build the item-index of selected datums for an individual scene.
                scene_item_index.extend([(scene_idx, int(sample_idx), datum_name) for sample_idx in valid_sample_indices
                                         ])
                logging.debug(f'No annotation filter --- Scene item index built in {time.time() - st:.2f}s.')
            else:
                annotated_samples = scene.annotation_index[samples_for_datum_name[valid_sample_indices], :].any(axis=1)
                scene_item_index.extend([
                    (scene_idx, int(sample_idx), datum_name) for sample_idx in valid_sample_indices[annotated_samples]
                ])
                logging.debug(f'Annotation filter -- Scene item index built in {time.time() - st:.2f}s.')
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
        data: OrderedDict
            See `get_point_cloud_from_datum` and `get_image_from_datum` for details.

        Raises
        ------
        ValueError
            Raised if the datum type of item at index is unsupported.
        """
        assert self.dataset_item_index is not None, ('Index is not built, select datums before getting elements.')

        # Get dataset item
        scene_idx, sample_idx_in_scene, datum_name = self.dataset_item_index[index]
        datum = self.get_datum(scene_idx, sample_idx_in_scene, datum_name)
        datum_type = datum.datum.WhichOneof('datum_oneof')

        if datum_type == 'image':
            datum_data, annotations = self.get_image_from_datum(scene_idx, sample_idx_in_scene, datum_name)
        elif datum_type == 'point_cloud':
            datum_data, annotations = self.get_point_cloud_from_datum(scene_idx, sample_idx_in_scene, datum_name)
        else:
            raise ValueError('Unknown datum type: {}'.format(datum_type))

        # TODO: Implement data/annotation load-time transforms, `torchvision` style.
        if annotations:
            datum_data.update(annotations)
        return datum_data


class FrameSceneDataset(_FrameDataset):
    """Main entry-point for single-modality dataset. Used for tasks with unordered data,
    i.e. 2D detection.

    Parameters
    ----------
    scene_dataset_json: str
        Full path to the scene dataset json holding collections of paths to scene json.

    split: str, default: 'train'
        Split of dataset to read ("train" | "val" | "test" | "train_overfit").

    datum_names: list, default: None
        Select datums for which to build index (see self.select_datums(datum_names)).
        NOTE: All selected datums must be of a the same datum type!

    requested_annotations: tuple, default: None
        Tuple of annotation types, i.e. ('bounding_box_2d', 'bounding_box_3d'). Should be equivalent
        to directory containing annotation from dataset root.

    requested_autolabels: tuple[str], default: None
        Tuple of annotation types similar to `requested_annotations`, but associated with a particular autolabeling model.
        Expected format is "<model_id>/<annotation_type>"

    only_annotated_datums: bool, default: False
        If True, only datums with annotations matching the requested annotation types are returned.

    use_diskcache: bool, default: True
        If True, cache ScenePb2 object using diskcache. If False, save the object in memory.
        NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of scenes.

    skip_missing_data: bool, default: False
        If True, check for missing files and skip during datum index building.
    """
    def __init__(
        self,
        scene_dataset_json,
        split='train',
        datum_names=None,
        requested_annotations=None,
        requested_autolabels=None,
        only_annotated_datums=False,
        use_diskcache=True,
        skip_missing_data=False,
    ):
        if not use_diskcache:
            logging.warning('Instantiating a dataset with use_diskcache=False may exhaust memory with a large dataset.')

        # Extract all scenes from the scene dataset JSON for the appropriate split
        scenes = BaseDataset._extract_scenes_from_scene_dataset_json(
            scene_dataset_json,
            split,
            requested_autolabels,
            is_datums_synchronized=False,
            use_diskcache=use_diskcache,
            skip_missing_data=skip_missing_data,
        )

        # Return SynchronizedDataset with scenes built from dataset.json
        dataset_metadata = DatasetMetadata.from_scene_containers(scenes, requested_annotations, requested_autolabels)
        super().__init__(
            dataset_metadata,
            scenes=scenes,
            datum_names=datum_names,
            requested_annotations=requested_annotations,
            requested_autolabels=requested_autolabels,
            only_annotated_datums=only_annotated_datums
        )


class FrameScene(_FrameDataset):
    """Main entry-point for single-modality dataset using a single scene JSON as input.

    NOTE: This class can be used to introspect a single scene given a scene
    directory with its associated scene JSON.

    Parameters
    ----------
    scene_json: str
        Full path to the scene json.

    datum_names: list, default: None
        Select datums for which to build index (see self.select_datums(datum_names)).
        NOTE: All selected datums must be of a the same datum type!

    requested_annotations: tuple, default: None
        Tuple of annotation types, i.e. ('bounding_box_2d', 'bounding_box_3d'). Should be equivalent
        to directory containing annotation from dataset root.

    requested_autolabels: tuple[str], default: None
        Tuple of annotation types similar to `requested_annotations`, but associated with a particular autolabeling model.
        Expected format is "<model_id>/<annotation_type>"

    only_annotated_datums: bool, default: False
        If True, only datums with annotations matching the requested annotation types are returned.

    use_diskcache: bool, default: True
        If True, cache ScenePb2 object using diskcache. If False, save the object in memory.
        NOTE: Setting use_diskcache to False would exhaust the memory if have a large number of scenes.

    skip_missing_data: bool, default: False
        If True, check for missing files and skip during datum index building.
    """
    def __init__(
        self,
        scene_json,
        datum_names=None,
        requested_annotations=None,
        requested_autolabels=None,
        only_annotated_datums=False,
        use_diskcache=True,
        skip_missing_data=False,
    ):
        if not use_diskcache:
            logging.warning('Instantiating a dataset with use_diskcache=False may exhaust memory with a large dataset.')

        # Extract a single scene from the scene JSON
        scene = BaseDataset._extract_scene_from_scene_json(
            scene_json,
            requested_autolabels,
            is_datums_synchronized=False,
            use_diskcache=use_diskcache,
            skip_missing_data=skip_missing_data,
        )

        # Return SynchronizedDataset with scenes built from dataset.json
        dataset_metadata = DatasetMetadata.from_scene_containers([scene], requested_annotations, requested_autolabels)
        super().__init__(
            dataset_metadata,
            scenes=[scene],
            datum_names=datum_names,
            requested_annotations=requested_annotations,
            requested_autolabels=requested_autolabels,
            only_annotated_datums=only_annotated_datums
        )
