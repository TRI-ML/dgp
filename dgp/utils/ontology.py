# Copyright 2020 Toyota Research Institute.  All rights reserved.
import logging
import numpy as np
import warnings
from collections import OrderedDict

from dgp.proto.dataset_pb2 import Ontology as OntologyV1
from dgp.proto.ontology_pb2 import Ontology as OntologyV2

# Special value and class name reserved for pixels that should be ignored
VOID_ID = 255

def build_detection_lookup_tables(dataset, remapped_ontology=None):
    """Build standard lookup tables from metadata for detection tasks"""
    # Note (sudeep.pillai): Move these to ontology[annotation_type].<attr>
    ontology = dataset.dataset_metadata.metadata.ontology if remapped_ontology is None else remapped_ontology
    if isinstance(ontology, OntologyV2):
        logging.info('Building detection lookup with OntologyV2 spec.')
        return _build_detection_lookup_tables_v2(dataset, remapped_ontology=remapped_ontology)
    elif isinstance(ontology, OntologyV1) or isinstance(ontology, RemappedDetectionOntology):
        logging.info('Building detection lookup with OntologyV1 spec.')
        return _build_detection_lookup_tables_v1(dataset, remapped_ontology=remapped_ontology)
    else:
        raise ValueError('Unknown ontology type={}'.format(type(ontology)))


def _build_detection_lookup_tables_v1(dataset, remapped_ontology=None):
    """Build standard lookup tables from metadata for detection tasks using Ontology spec V1.
    Note: This is soon to be deprecated.
    """
    warnings.warn(
        """Using OntologyV1, this is soon to be deprecated.
    Consider using the new ontology spec (v2) at dgp.proto.ontology_pb2.Ontology."""
    )
    # Note (sudeep.pillai): Move these to ontology[annotation_type].<attr>
    ontology = dataset.dataset_metadata.metadata.ontology if remapped_ontology is None else remapped_ontology

    # The dataset metadata can include "stuff" classes. We restrict to "thing" classes here.
    # All dictionary id lookups below only consider "thing" classes
    # NOTE: NEEDS to be sorted, otherwise ordering of id's gets unpredictably messed up
    thing_class_ids = sorted([int(class_id) for class_id, isthing in ontology.isthing.items() if isthing])

    # Mapping integer JSON class id's to a contiguous set starting at 1 (as 0 is reserved for background)
    dataset.json_category_id_to_contiguous_id = OrderedDict(
        (j_id, c_id + 1) for c_id, j_id in enumerate(thing_class_ids)
    )

    # Reverse lookup from contiguous id's to JSON id's
    dataset.contiguous_category_id_to_json_id = OrderedDict(
        (c_id, j_id) for j_id, c_id in dataset.json_category_id_to_contiguous_id.items()
    )

    # Number of classes (NOT including background)
    dataset.num_classes = len(thing_class_ids)

    # Map class id's to string names (these id's are contiguous starting at 1)
    id_to_name = dict(ontology.id_to_name)
    dataset.id_to_name = OrderedDict(
        (c_id, id_to_name[j_id]) for c_id, j_id in dataset.contiguous_category_id_to_json_id.items()
    )

    # Map class names to ids.
    dataset.name_to_id = OrderedDict({name: c_id for c_id, name in dataset.id_to_name.items()})

    # Map class ids to colors (these id's are contiguous starting at 1)
    colormap = dict(ontology.colormap)
    dataset.colormap = OrderedDict()
    for c_id, j_id in dataset.contiguous_category_id_to_json_id.items():
        color = colormap[j_id]
        dataset.colormap[c_id] = [color.r, color.g, color.b]


def _build_detection_lookup_tables_v2(dataset, remapped_ontology=None):
    """Build standard lookup tables from metadata for detection tasks using Ontology spec V2"""
    # Note (sudeep.pillai): Move these to ontology[annotation_type].<attr>
    ontology = dataset.dataset_metadata.metadata.ontology if remapped_ontology is None else remapped_ontology

    # The dataset metadata can include "stuff" classes. We restrict to "thing" classes here.
    # All dictionary id lookups below only consider "thing" classes
    # NOTE: NEEDS to be sorted, otherwise ordering of id's gets unpredictably messed up
    thing_class_ids = sorted([int(ontology_item.id) for ontology_item in ontology.items if ontology_item.isthing])

    # Mapping integer JSON class id's to a contiguous set starting at 1 (as 0 is reserved for background)
    dataset.json_category_id_to_contiguous_id = OrderedDict(
        (j_id, c_id + 1) for c_id, j_id in enumerate(thing_class_ids)
    )

    # Reverse lookup from contiguous id's to JSON id's
    dataset.contiguous_category_id_to_json_id = OrderedDict(
        (c_id, j_id) for j_id, c_id in dataset.json_category_id_to_contiguous_id.items()
    )

    # Number of classes (NOT including background)
    dataset.num_classes = len(thing_class_ids)

    # Map class id's to string names (these id's are contiguous starting at 1)
    id_to_name = {int(ontology_item.id): ontology_item.name for ontology_item in ontology.items}
    dataset.id_to_name = OrderedDict(
        (c_id, id_to_name[j_id]) for c_id, j_id in dataset.contiguous_category_id_to_json_id.items()
    )

    # Map class names to ids.
    dataset.name_to_id = OrderedDict({name: c_id for c_id, name in dataset.id_to_name.items()})

    # Map class ids to colors (these id's are contiguous starting at 1)
    colormap = {int(ontology_item.id): ontology_item.color for ontology_item in ontology.items}
    dataset.colormap = OrderedDict()
    for c_id, j_id in dataset.contiguous_category_id_to_json_id.items():
        color = colormap[j_id]
        dataset.colormap[c_id] = [color.r, color.g, color.b]


def build_instance_lookup_tables(dataset, remapped_ontology=None):
    """Build standard lookup tables from metadata for instance segmentation tasks"""
    # Note (sudeep.pillai): Move these to ontology[annotation_type].<attr>
    instance_ontology = dataset.dataset_metadata.metadata.ontology if remapped_ontology is None else remapped_ontology
    if isinstance(instance_ontology, OntologyV2):
        logging.info('Building instance lookup with OntologyV2 spec.')
        instance_id_to_name = OrderedDict(
            sorted([(ontology_item.id, ontology_item.name) for ontology_item in instance_ontology.items])
        )
    elif isinstance(ontology, OntologyV1) or isinstance(instance_ontology, RemappedDetectionOntology):
        logging.info('Building instance lookup with OntologyV1 spec.')
        instance_id_to_name = OrderedDict(sorted(instance_ontology.id_to_name.items()))
    else:
        raise ValueError('Unknown ontology type={}'.format(type(instance_ontology)))

    instance_class_ids = sorted(instance_id_to_name.keys())
    dataset.instance_class_id_to_contiguous_id = OrderedDict(
        (class_id, contiguous_id) for contiguous_id, class_id in enumerate(instance_class_ids)
    )
    # Contiguous ID map back to original class ID
    dataset.instance_contiguous_id_to_class_id = OrderedDict(
        (contiguous_id, class_id) for class_id, contiguous_id in dataset.instance_class_id_to_contiguous_id.items()
    )

    # Direct lookups from contiguous ID to name and color as well
    dataset.instance_contiguous_id_to_name = OrderedDict((contiguous_id, instance_id_to_name[class_id])
                                                for contiguous_id, class_id in dataset.instance_contiguous_id_to_class_id.items()
                                                )
    dataset.instance_name_to_contiguous_id = OrderedDict(
        (name, contiguous_id) for contiguous_id, name in dataset.instance_contiguous_id_to_name.items()
    )


def build_semseg_lookup_tables(dataset, remapped_ontology=None):
    """Build standard lookup tables from metadata for semantic segmentation tasks"""
    # Note (sudeep.pillai): Move these to ontology[annotation_type].<attr>
    semseg_ontology = dataset.dataset_metadata.metadata.ontology if remapped_ontology is None else remapped_ontology
    if isinstance(semseg_ontology, OntologyV2):
        logging.info('Building semseg lookup with OntologyV2 spec.')
        semseg_id_to_name = OrderedDict(
            sorted([(ontology_item.id, ontology_item.name) for ontology_item in semseg_ontology.items])
        )
    elif isinstance(semseg_ontology, OntologyV1) or isinstance(semseg_ontology, RemappedDetectionOntology):
        logging.info('Building semseg lookup with OntologyV1 spec.')
        semseg_id_to_name = OrderedDict(sorted(semseg_ontology.id_to_name.items()))
    else:
        raise ValueError('Unknown ontology type={}'.format(type(semseg_ontology)))

    sem_seg_class_ids = sorted(semseg_id_to_name.keys())
    dataset.semseg_class_id_to_contiguous_id = OrderedDict(
        (class_id, contiguous_id) for contiguous_id, class_id in enumerate(semseg_id_to_name)
    )
    # Build lookup table to map raw label data to training IDs
    dataset.semseg_label_lookup = np.ones(max(semseg_id_to_name) + 1, dtype=np.uint8) * VOID_ID
    for class_id, contiguous_id in dataset.semseg_class_id_to_contiguous_id.items():
        dataset.semseg_label_lookup[class_id] = contiguous_id
    dataset.VOID_ID = VOID_ID



class RemappedDetectionOntology:
    """Object to spoof dataset_pb2.metadata.ontology"""
    def __init__(self):
        self.colormap = {}
        self.name_to_id = {}
        self.id_to_name = {}
        self.isthing = {}

    def __repr__(self):
        return 'Colormap: {}, Name to ID: {}, ID to name: {}, isthing: {}'.format(
            self.colormap, self.name_to_id, self.id_to_name, self.isthing
        )
