# Copyright 2022 Woven Planet NA. All rights reserved.
"""Dataloader for DGP SynchornizedScene Wicker datasets"""
# pylint: disable=missing-param-doc
import logging
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional

from dgp2wicker.ingest import (
    FIELD_TO_WICKER_SERIALIZER,
    ILLEGAL_COMBINATIONS,
    gen_wicker_key,
    parse_wicker_key,
)
from dgp2wicker.serializers import OntologySerializer
from wicker.core.datasets import S3Dataset  # type: ignore

from dgp.annotations import ANNOTATION_REGISTRY  # type: ignore
from dgp.annotations.ontology import Ontology  # type: ignore

logger = logging.getLogger(__name__)


def compute_columns(
    datum_names: List[str],
    datum_types: List[str],
    requested_annotations: List[str],
    cuboid_datum: Optional[str] = None,
    with_ontology_table: bool = True,
) -> List[str]:
    """Method to parse requested datums, types, and annotations into keys for fetching from wicker.

    Parameters
    ----------
    datum_names: List
        List of datum names to load.

    datum_types: List
        List of datum types i.e, 'image', 'point_cloud', 'radar_point_cloud'.

    requested_annotations: List
        List of annotation types to load i.e. 'bounding_box_3d', 'depth' etc.

    cuboid_datum: str, default: None
        Optional datum name to restrict loading of bounding_box_3d annotations to a single datum.
        For example if we do not desire to load bounding_box_3d for both the lidar datum and every
        image datum, we would set this field to 'lidar'.

    with_ontology_table: bool, default: True
        Flag to add loading of ontology tables

    Returns
    -------
    columns_to_load: List
        A list of keys to fetch from wicker.
    """
    # Compute the requested columns.
    columns_to_load = ['scene_index', 'scene_uri', 'sample_index_in_scene']
    for datum_name, datum_type in zip(datum_names, datum_types):
        fields = ['timestamp', 'pose', 'extrinsics', 'datum_type']

        # Add extra information for different datum type.
        if datum_type == 'image':
            fields.extend(['intrinsics', 'rgb', 'distortion'])
        elif datum_type == 'point_cloud':
            fields.extend(['point_cloud', 'extra_channels'])
        elif datum_type == 'radar_point_cloud':
            fields.extend(['point_cloud', 'extra_channels', 'velocity', 'covariance'])

        if requested_annotations is not None:
            fields.extend(requested_annotations)

        for annotation in fields:
            if (datum_type, annotation) in ILLEGAL_COMBINATIONS:
                #print('skip', datum_name, datum_type, annotation)
                continue

            # If this is a cuboid annotation, optionally, only add it for a specific datum
            if annotation == 'bounding_box_3d' and cuboid_datum is not None:
                if datum_name != cuboid_datum:
                    #print('skip', datum_name, datum_type, annotation)
                    continue
            columns_to_load.append(gen_wicker_key(datum_name, annotation))

    if with_ontology_table and requested_annotations is not None:
        for ann in requested_annotations:
            if ann in ANNOTATION_REGISTRY:
                if ann == 'depth':  # DenseDepth does not require an ontology
                    continue
                columns_to_load.append(gen_wicker_key('ontology', ann))

    return columns_to_load


class DGPS3Dataset(S3Dataset):
    """
    S3Dataset for data stored in dgp synchronized scene format in wicker. This is a baseclass
    inteded for use with all DGP wicker datasets. It handles conversion from wicker binary formats
    to DGP datum and annotation objects
    """
    def __init__(self, *args: Any, wicker_sample_index: Optional[List[List[int]]] = None, **kwargs: Any) -> None:
        """S3Dataset for data stored in dgp synchronized scene format in wicker. This is a baseclass
        inteded for use with all DGP wicker datasets. It handles conversion from wicker binary formats
        to DGP datum and annotation objects.

        Parameters
        ----------
        wicker_sample_index: List[List[int]], default: None
            A mapping from this dataset's index to a list of wicker indexes. If None, a mappind for all
            single frames will be generated. 
        """
        super().__init__(*args, **kwargs)

        # All datasets will need a mapping from this datasets index -> [ wicker indexes ]
        self.wicker_sample_index: List[List[int]] = [[]]
        if wicker_sample_index is None:
            N = super().__len__()
            self.wicker_sample_index = [[k] for k in range(N)]
        else:
            self.wicker_sample_index = wicker_sample_index

        self._ontology_table: Optional[Dict[str, Ontology]] = None

    @property
    def ontology_table(self) -> Optional[Dict[str, Ontology]]:
        """Return the ontology table if any.

        Returns
        -------
        ontology_table: Dict
            The ontology table or None if an ontology table has not been assigned with self._create_ontology_table.
        """
        return self._ontology_table

    def __len__(self) -> int:
        """ Number of samples in dataset

        Returns
        -------
        length: int
            The number of samples in the dataset
        """
        return len(self.wicker_sample_index)

    def _create_ontology_table(self, raw_wicker_sample: Dict[str, Any]) -> Dict[str, Ontology]:
        """"Create ontology table based on given wicker item.

        Parameters
        ----------
        raw_wicker_sample: Dict
            A raw wicker sample containing ontology keys, ex: ontology___bounding_box_3d etc.

        Returns
        -------
        ontology_table: Dict
            A dictionary keyed by annotation name holding an ontology for that annotation.
        """
        # Set the ontologies only once.
        # NOTE: We assume every sample has the same ontology
        ontology_table = {}
        ontology_keys = [key for key in raw_wicker_sample if 'ontology' in key]
        for key in ontology_keys:
            _, ontology_type = parse_wicker_key(key)
            serializer = OntologySerializer(ontology_type)
            ontology_table[ontology_type] = serializer.unserialize(raw_wicker_sample[key])

        return ontology_table

    def _process_raw_wicker_sample(self, raw_wicker_sample: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Parse raw data from wicker into datums/fields.

        Parameters
        ----------
        raw_wicker_sample: Dict
            The raw output from wicker S3Dataset.

        Returns
        -------
        sample_dict: Dict[str, Dict[str,Any]]
            A dictionary keyed by datum name holding DGP SynchronizedScene like datums.
        """
        # Lots of annotations require an ontology table, so we process those first
        ontology_table = self._create_ontology_table(raw_wicker_sample)

        if self.ontology_table is None:
            self._ontology_table = ontology_table
        else:
            # Make sure the ontology table has not changed. We expect this to be constant across a dataset
            assert set(ontology_table.keys()) == set(self.ontology_table.keys())
            for field in self.ontology_table:
                assert self.ontology_table[field] == ontology_table[field]

        output_dict: Dict[str, Dict[str, Any]] = defaultdict(OrderedDict)
        for key, raw in raw_wicker_sample.items():
            if key in ['scene_uri', 'scene_index', 'sample_index_in_scene']:
                output_dict['meta'][key] = raw
                continue
            if 'ontology' in key:
                continue

            datum_name, field = parse_wicker_key(key)
            serializer = FIELD_TO_WICKER_SERIALIZER[field]()
            if hasattr(serializer, 'ontology'):
                serializer.ontology = self.ontology_table[field]

            output_dict[datum_name]['datum_name'] = datum_name
            output_dict[datum_name][field] = serializer.unserialize(raw)

        if 'meta' in output_dict:
            output_dict['meta']['datum_name'] = 'meta'
            output_dict['meta']['datum_type'] = 'meta'

        return output_dict

    def __getitem__(self, index: int) -> List[Dict[str, Dict[str, Any]]]:
        """"Get the dataset item at index.

        Parameters
        ----------
        index: int
            The index to get.

        Returns
        -------
        context: List
            A context window with samples as dicts keyed by datum name.
        """
        wicker_samples = self.wicker_sample_index[index]

        context = []
        for idx in wicker_samples:
            raw = super().__getitem__(idx)
            sample = self._process_raw_wicker_sample(raw)
            context.append(sample)

        return context
