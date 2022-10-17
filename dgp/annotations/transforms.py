# Copyright 2021-2022 Woven Planet. All rights reserved.
from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from dgp.annotations import ONTOLOGY_REGISTRY
from dgp.annotations.transform_utils import (
    construct_remapped_ontology,
    remap_bounding_box_annotations,
    remap_instance_segmentation_2d_annotation,
    remap_semantic_segmentation_2d_annotation,
)
from dgp.utils.accumulate import points_in_cuboid


class Compose:
    """Composes several transforms together.

    Parameters
    ----------
    transforms
        List of transforms to compose __call__ method that takes in an OrderedDict

        Example:
            >>> transforms.Compose([
            >>>     transforms.CenterCrop(10),
            >>>     transforms.ToTensor(),
            >>> ])
    """
    def __init__(self, transforms):
        if not all([isinstance(t, BaseTransform) for t in transforms]):
            raise TypeError('All transforms used in Compose should inherit from `BaseTransform`')
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class BaseTransform:
    """
    Base transform class that other transforms should inherit from. Simply ensures that
    input type to `__call__` is an OrderedDict (in general usage this dict will include
    keys such as 'rgb', 'bounding_box_2d', etc. i.e. raw data and annotations)

    cf. `OntologyMapper` for an example
    """
    def __call__(self, data):
        if not isinstance(data, OrderedDict) and not \
                (isinstance(data, list) and isinstance(data[0], list) and isinstance(data[0][0], OrderedDict)):
            raise TypeError('`BaseTransform` expects input of type `OrderedDict` or list of list of `OrderedDict`.')
        return self.transform(data)

    def transform(self, data):
        """
        Parameters
        ----------
        data: OrderedDict or list[list[OrderedDict]]
            dataset item as returned by `_SynchronizedDataset' or `_FrameDataset`.

        Returns
        -------
        OrderedDict or list[list[OrderedDict]]:
            Same type with input with transformations applied to dataset item.
        """
        if isinstance(data, OrderedDict):
            return self.transform_datum(data)
        elif isinstance(data, list):
            # NOTE: For now, only support per-sample transform, i.e. apply sample-wise transform to each timetstamp.
            # TODO: When needed, add support for context-wise transform.
            return [self.transform_sample(sample) for sample in data]

    def transform_datum(self, datum):
        raise NotImplementedError

    def transform_sample(self, sample):
        raise NotImplementedError


class OntologyMapper(BaseTransform):
    """
    Mapping ontology based on a lookup_table.
    The remapped ontology will base on the remapped_ontology_table if provided.
    Otherwise, the remapped ontology will be automatically constructed based on the order of lookup_table.

    Parameters
    ----------
    original_ontology_table: dict[str->dgp.annotations.Ontology]
        Ontology object *per annotation type*
        The original ontology table.
        {
            "bounding_box_2d": BoundingBoxOntology[<ontology_sha>],
            "autolabel_model_1/bounding_box_2d": BoundingBoxOntology[<ontology_sha>],
            "semantic_segmentation_2d": SemanticSegmentationOntology[<ontology_sha>]
            "bounding_box_3d": BoundingBoxOntology[<ontology_sha>],
        }

    lookup_table: dict[str->dict]
        Lookup table *per annotation type* for each of the classes the user wants to remap.
        Lookups are old class name to new class name

        e.g.:
        {
            'bounding_box_2d': {
                'Car': 'Car',
                'Truck': 'Car',
                'Motorcycle': 'Motorcycle'
            }
            ...
        }

    remapped_ontology_table: dict[str->dgp.annotations.Ontology]
        Ontology object *per annotation type*
        If specified, the ontology will be remapped to the given remapped_ontology_table.
        {
            "bounding_box_2d": BoundingBoxOntology[<ontology_sha>],
            "autolabel_model_1/bounding_box_2d": BoundingBoxOntology[<ontology_sha>],
            "semantic_segmentation_2d": SemanticSegmentationOntology[<ontology_sha>]
            "bounding_box_3d": BoundingBoxOntology[<ontology_sha>],
        }
    """
    # This will evolve as handlers for more annotation types are added
    SUPPORTED_ANNOTATION_TYPES = (
        'bounding_box_2d', 'semantic_segmentation_2d', 'bounding_box_3d', 'instance_segmentation_2d'
    )

    def __init__(self, original_ontology_table, lookup_table, remapped_ontology_table=None):
        for annotation_key in lookup_table:
            if annotation_key not in self.SUPPORTED_ANNOTATION_TYPES:
                raise ValueError(f'annotation_key {annotation_key} not supported for remapping yet, we accept PRs')
            if annotation_key not in original_ontology_table:
                raise ValueError(f'annotation_key {annotation_key} needs to be present in `ontology_table`')

        self.lookup_table = lookup_table
        self.original_ontology_table = original_ontology_table
        self.remapped_ontology_table = {}
        for annotation_key, lookup in self.lookup_table.items():
            assert all([
                class_name in original_ontology_table[annotation_key].class_names for class_name in lookup.keys()
            ]), 'All keys in `lookup` need to be valid class names in specified `ontology`'
            if remapped_ontology_table is not None and annotation_key in remapped_ontology_table:
                self.remapped_ontology_table[annotation_key] = remapped_ontology_table[annotation_key]
            else:
                remapped_ontology_pb2 = construct_remapped_ontology(
                    original_ontology_table[annotation_key], lookup, annotation_key
                )
                self.remapped_ontology_table[annotation_key] = ONTOLOGY_REGISTRY[annotation_key](remapped_ontology_pb2)
            assert all([
                class_name in self.remapped_ontology_table[annotation_key].class_names for class_name in lookup.values()
            ]), 'All values in `lookup` need to be valid class names in specified `remapped_ontology`'

    def transform_datum(self, datum):
        """
        Parameters
        ----------
        datum: OrderedDict
            Dictionary containing raw data and annotations, with keys such as:
            'rgb', 'intrinsics', 'bounding_box_2d'.
            All annotation_keys in `self.lookup_table` (and `self.remapped_ontology_table`)
            are expected to be contained

        Returns
        -------
        datum: OrderedDict
            Same dictionary but with annotations in `self.lookup_table` remapped to desired ontologies

        Raises
        ------
        ValueError
            Raised if the datum to remap does not contain all expected annotations.
        """
        if not all([annotation_key in datum for annotation_key in self.remapped_ontology_table]):
            raise ValueError('The data you are trying to remap does not have all annotations it expects')

        for annotation_key, remapped_ontology in self.remapped_ontology_table.items():

            lookup_table = self.lookup_table[annotation_key]
            original_ontology = datum[annotation_key].ontology

            # Need to have specific handlers for each annotation type
            if annotation_key == 'bounding_box_2d' or annotation_key == 'bounding_box_3d':
                datum[annotation_key] = remap_bounding_box_annotations(
                    datum[annotation_key], lookup_table, original_ontology, remapped_ontology
                )
            elif annotation_key == 'semantic_segmentation_2d':
                datum[annotation_key] = remap_semantic_segmentation_2d_annotation(
                    datum[annotation_key], lookup_table, original_ontology, remapped_ontology
                )
            elif annotation_key == 'instance_segmentation_2d':
                datum[annotation_key] = remap_instance_segmentation_2d_annotation(
                    datum[annotation_key], lookup_table, original_ontology, remapped_ontology
                )

        return datum


class AddLidarCuboidPoints(BaseTransform):
    """Populate the num_points field for bounding_box_3d"""
    def __init__(self, subsample: int = 1) -> None:
        """Populate the num_points field for bounding_box_3d. Optionally downsamples the point cloud for speed.

        Parameters
        ----------
        subsample: int, default: 1
            Fraction of point cloud to use for computing the number of points. i.e., subsample=10 indicates that
            1/10th of the points should be used.
        """
        super().__init__()
        self.subsample = subsample

    def transform_datum(self, datum: Dict[str, Any]) -> Dict[str, Any]:
        """Populate the num_points field for bounding_box_3d
        Parameters
        ----------
        datum: Dict[str,Any]
            A dgp lidar or point cloud datum. Must contain keys bounding_box_3d and point_cloud

        Returns
        -------
        datum: Dict[str,Any]
            The datum with num_points added to the cuboids
        """
        if 'bounding_box_3d' not in datum:
            return datum

        boxes = datum['bounding_box_3d']
        if boxes is None or len(boxes) == 0:
            return datum

        assert 'point_cloud' in datum, 'datum should contain point_cloud key'
        point_cloud = datum['point_cloud']
        if self.subsample > 1:
            N = point_cloud.shape[0]
            sample_idx = np.random.choice(N, N // self.subsample)
            point_cloud = point_cloud[sample_idx].copy()

        for box in boxes:
            # If a box is missing this num_points value we expect it will have a value of 0
            # so only run this for boxes that might be missing the value
            if box.num_points == 0:
                in_cuboid = points_in_cuboid(point_cloud, box)
                box._num_points = np.sum(in_cuboid) * self.subsample

        return datum
