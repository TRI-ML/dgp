# Copyright 2021 Toyota Research Institute.  All rights reserved.
from collections import OrderedDict

import numpy as np

from dgp.annotations import (
    BoundingBoxOntology,
    InstanceSegmentationOntology,
    Ontology,
    PanopticSegmentation2DAnnotation,
    SemanticSegmentation2DAnnotation,
    SemanticSegmentationOntology,
)
from dgp.proto.ontology_pb2 import Ontology as OntologyPB2
from dgp.proto.ontology_pb2 import OntologyItem


def remap_bounding_box_annotations(bounding_box_annotations, lookup_table, original_ontology, remapped_ontology):
    """
    Parameters
    ----------
    bounding_box_annotations: BoundingBox2DAnnotationList or BoundingBox3DAnnotationList
        Annotations to remap

    lookup_table: dict
        Lookup from old class names to new class names
        e.g.:
            {
                'Car': 'Car',
                'Truck': 'Car',
                'Motorcycle': 'Motorcycle'
            }

    original_ontology: BoundingBoxOntology
        Ontology we are remapping annotations from

    remapped_ontology: BoundingBoxOntology
        Ontology we are mapping annotations to

    Returns
    -------
    remapped_bounding_box_annotations: BoundingBox2DAnnotationList or BoundingBox3DAnnotationList
        Remapped annotations with the same type of bounding_box_annotations
    """
    assert (isinstance(original_ontology, BoundingBoxOntology) and isinstance(remapped_ontology, BoundingBoxOntology))
    # Iterate over boxes constructing box with remapped class for each
    remapped_boxlist = []
    for box in bounding_box_annotations:
        original_class_name = original_ontology.contiguous_id_to_name[box.class_id]
        if original_class_name in lookup_table:
            # Remap class_id in box
            remapped_class_id = remapped_ontology.name_to_contiguous_id[lookup_table[original_class_name]]
            box.class_id = remapped_class_id
            remapped_boxlist.append(box)

    # Instantiate BoundingBox2DAnnotationList or BoundingBox3DAnnotationList with remapped boxlist and remapped BoundingBoxOntology
    annotation_type = type(bounding_box_annotations)
    return annotation_type(remapped_ontology, remapped_boxlist)


def remap_semantic_segmentation_2d_annotation(
    semantic_segmentation_annotation, lookup_table, original_ontology, remapped_ontology
):
    """
    Parameters
    ----------
    semantic_segmentation_annotation: SemanticSegmentation2DAnnotation
        Annotation to remap

    lookup_table: dict
        Lookup from old class names to new class names
        e.g.:
            {
                'Car': 'Car',
                'Truck': 'Car',
                'Motorcycle': 'Motorcycle'
            }

    original_ontology: SemanticSegmentationOntology
        Ontology we are remapping annotation from

    remapped_ontology: SemanticSegmentationOntology
        Ontology we are mapping annotation to

    Returns
    -------
    remapped_semantic_segmentation_2d_annotation: SemanticSegmentation2DAnnotation
        Remapped annotation
    """
    assert (isinstance(original_ontology, SemanticSegmentationOntology) and \
        isinstance(remapped_ontology, SemanticSegmentationOntology))

    original_segmentation_image = semantic_segmentation_annotation.label
    remapped_segmentation_image = np.ones_like(original_segmentation_image) * Ontology.VOID_ID
    for class_name in lookup_table:
        # pylint: disable=E1137
        remapped_segmentation_image[original_segmentation_image == original_ontology.name_to_contiguous_id[class_name]] = \
            remapped_ontology.name_to_contiguous_id[lookup_table[class_name]]
        # pylint: enable=E1137
    # Instantiate SemanticSegmentation2DAnnotation with remapped segmentation image and remapped SemanticSegmentationOntology
    return SemanticSegmentation2DAnnotation(remapped_ontology, remapped_segmentation_image)


def remap_instance_segmentation_2d_annotation(
    instance_segmentation_annotation, lookup_table, original_ontology, remapped_ontology
):
    """
    Parameters
    ----------
    instance_segmentation_annotation: PanopticSegmentation2DAnnotation
        Annotation to remap

    lookup_table: dict
        Lookup from old class names to new class names
        e.g.:
            {
                'Car': 'Car',
                'Truck': 'Car',
                'Motorcycle': 'Motorcycle'
            }

    original_ontology: InstanceSegmentationOntology
        Ontology we are remapping annotation from

    remapped_ontology: InstanceSegmentationOntology
        Ontology we are mapping annotation to

    Returns
    -------
    PanopticSegmentation2DAnnotation:
        Remapped annotation
    """
    assert (
        isinstance(original_ontology, InstanceSegmentationOntology)
        and isinstance(remapped_ontology, InstanceSegmentationOntology)
    )
    # Iterate over boxes constructing box with remapped class for each
    remapped_masklist = []
    for instance_mask in instance_segmentation_annotation:
        original_class_name = original_ontology.contiguous_id_to_name[instance_mask.class_id]
        if original_class_name in lookup_table:
            # Remap class_id in box
            remapped_class_id = remapped_ontology.name_to_contiguous_id[lookup_table[original_class_name]]
            instance_mask.class_id = remapped_class_id
            remapped_masklist.append(instance_mask)

    assert isinstance(instance_segmentation_annotation, PanopticSegmentation2DAnnotation)
    return PanopticSegmentation2DAnnotation.from_masklist(
        remapped_masklist, remapped_ontology, instance_segmentation_annotation.panoptic_image.shape,
        instance_segmentation_annotation.panoptic_image_dtype
    )


def construct_remapped_ontology(ontology, lookup, annotation_key):
    """Given an Ontology object and a lookup from old class names to new class names, construct
    an ontology proto for the new ontology that results

    Parameters
    ----------
    ontology: dgp.annotations.Ontology
        Ontology we are trying to remap using `lookup`
        eg. ontology.id_to_name = {0: 'Car', 1: 'Truck', 2: 'Motrocycle'}

    lookup: dict
        Lookup from old class names to new class names
        e.g.:
            {
                'Car': 'Car',
                'Truck': 'Car',
                'Motorcycle': 'Motorcycle'
            }

        NOTE: `lookup` needs to be exhaustive; any classes that the user wants to have in returned
        ontology need to be remapped explicitly

    annotation_key: str
        Annotation key of Ontology
        e.g. `bounding_box_2d`

    Returns
    -------
    remapped_ontology_pb2: dgp.proto.ontology_pb2.Ontology
        Ontology defined by applying `lookup` on original `ontology`

        NOTE: This is constructed by iterating over class names in `lookup.keys()` in
        alphabetical order, so if both 'Car' and 'Motorcycle' get remapped to 'DynamicObject', the
        color for 'DynamicObject' will be the original color for 'Car'

        Any class names not in `lookup` are dropped

    Notes
    -----
    This could be a class function of `Ontology`
    """
    # Will work with top-level Ontology class here for util to be generic
    assert isinstance(ontology, Ontology), f'Expected Ontology, got {type(ontology)}'

    # Construct lookup from new class name to original class names that map to it
    remapped_class_name_to_original_class_names = OrderedDict()
    for class_name, remapped_class_name in lookup.items():  # NOTE: this assumes Ordered
        if remapped_class_name not in remapped_class_name_to_original_class_names:
            remapped_class_name_to_original_class_names[remapped_class_name] = []
        remapped_class_name_to_original_class_names[remapped_class_name].append(class_name)

    # Sort alphabetically
    remapped_class_name_to_original_class_names = {
        k: sorted(v)
        for k, v in remapped_class_name_to_original_class_names.items()
    }

    remapped_ontology_pb2 = OntologyPB2()

    for remapped_class_id, (remapped_class_name,
                            original_class_names) in enumerate(remapped_class_name_to_original_class_names.items()):

        # Get class_id and color for class name that we're remapping
        original_class_ids = [ontology.name_to_id[class_name] for class_name in original_class_names]

        isthing = [ontology.isthing[class_id] for class_id in original_class_ids]
        # NOTE: Except semantic_segmentation_2d, classes being grouped together can only be all fromthings or stuffs classes
        if annotation_key == 'semantic_segmentation_2d':
            # semantic_segmentation_2d should only be stuff
            isthing = False
        else:
            # Enforce that classes mapping to the same class are either all things or all stuff
            assert len(set(isthing)) == 1, "Classes mapping to the same class are either all things or all stuff"
            isthing = isthing[0]

        # Keep first color from original class names (sorted alphabetically)
        remapped_class_color = ontology.colormap[original_class_ids[0]]

        # Construct remapped ontology item
        remapped_ontology_pb2.items.extend([
            OntologyItem(
                name=remapped_class_name,
                id=remapped_class_id,
                isthing=isthing,
                color=OntologyItem.Color(
                    r=remapped_class_color[0], g=remapped_class_color[1], b=remapped_class_color[2]
                )
            )
        ])

    # semantic segmentation 2d will always have a VOID class
    if annotation_key == 'semantic_segmentation_2d' and \
        not Ontology.VOID_CLASS in remapped_class_name_to_original_class_names:
        remapped_ontology_pb2.items.extend([
            OntologyItem(
                name=Ontology.VOID_CLASS, id=Ontology.VOID_ID, isthing=False, color=OntologyItem.Color(r=0, g=0, b=0)
            )
        ])

    return remapped_ontology_pb2
