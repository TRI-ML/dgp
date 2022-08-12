# Copyright 2021 Toyota Research Institute.  All rights reserved.
import os
from collections import OrderedDict, defaultdict

from dgp import ONTOLOGY_FOLDER
from dgp.annotations.ontology import Ontology, open_ontology_pbobject
from dgp.proto import annotations_pb2
from dgp.utils.protobuf import (
    generate_uid_from_pbobject,
    open_pbobject,
    open_remote_pb_object,
)


def get_scene_statistics(scene, verbose=True):
    """Given a DGP scene, return simple statistics (counts) of the scene.

    Parameters
    ----------
    scene: dgp.proto.scene_pb2.Scene
        Scene Object.

    verbose: bool, optional
        Print the stats if True. Default: True.

    Returns
    --------
    scene_stats: OrderedDict
        num_samples: int
            Number of samples in the Scene
        num_images: int
            Number of image datums in the Scene.
        num_point_clouds: int
            Number of point_cloud datums in the Scene.
        <datum_type>_<annotation_type>: int
            Number of <datum_type> with associated <annotation_type> annotation files.
    """
    num_samples = len(scene.samples)
    num_images, num_point_clouds = 0, 0
    annotation_counts = defaultdict(int)
    datum_index = {datum.key: idx for (idx, datum) in enumerate(scene.data)}
    for sample in scene.samples:
        for datum_key in sample.datum_keys:
            datum = scene.data[datum_index[datum_key]]
            if datum.datum.HasField('image'):
                num_images += 1
            elif datum.datum.HasField('point_cloud'):
                num_point_clouds += 1
            else:
                continue
            datum_type = datum.datum.WhichOneof('datum_oneof')
            datum_value = getattr(datum.datum, datum_type)
            annotations = datum_value.annotations
            for key in annotations:
                name = annotations_pb2.AnnotationType.DESCRIPTOR.values_by_number[key].name
                annotation_counts['{}_{}'.format(datum_type.upper(), name)] += 1

    if verbose:
        print('-' * 80)
        sample_info = 'Samples: {},\t Images: {},\t Point Clouds: {}\n\t'.format(
            num_samples, num_images, num_point_clouds
        )
        sample_info += ', '.join(['{}: {}'.format(k, v) for k, v in annotation_counts.items()])
        print('Scene: {}\n\t'.format(scene.name) + sample_info)

    return OrderedDict({
        'num_samples': num_samples,
        'num_images': num_images,
        'num_point_clouds': num_point_clouds,
        **annotation_counts
    })


def _get_bounding_box_annotation_info(annotation_enum):
    """Returns datum_type, annotation_pb given an ontology
    Parameters
    ----------
    annotation_enum: dgp.proto.annotations_pb2.AnnotationType
        Annotation type enum

    Returns
    -------
    str, dgp.proto.annotations_pb2.[AnnotationClass]
        The datum type, and annotation class corresponding to the annotation enum

    Raises
    ------
    Exception
        Raised if annotation_enum value does not map to a supported box type.
    """
    if annotation_enum == annotations_pb2.BOUNDING_BOX_3D:
        return 'point_cloud', annotations_pb2.BoundingBox3DAnnotations
    elif annotation_enum == annotations_pb2.BOUNDING_BOX_2D:
        return 'image', annotations_pb2.BoundingBox2DAnnotations
    else:
        raise Exception('Annotation info not supported')


def get_scene_class_statistics(scene, scene_dir, annotation_enum, ontology=None):
    """Given a DGP scene, return class counts of the annotations in the scene.

    Parameters
    ----------
    scene: dgp.proto.scene_pb2.Scene
        Scene Object.

    scene_dir: string
        s3 URL or local path to scene.

    annotation_enum: dgp.proto.ontology_pb2.AnnotationType
        Annotation type enum

    ontology: dgp.proto.ontology_pb2.Ontology or None
        Stats will be computed for this ontology. If None, the ontology will be read from the scene.

    Returns
    --------
    scene_stats: OrderedDict
        class_name: int
            Counts of annotations for each class.
    """
    datum_type, annotation_pb = _get_bounding_box_annotation_info(annotation_enum)

    if ontology is not None:
        # Scene ontology must match input ontology
        class_counts = OrderedDict({item.name: 0 for item in ontology.items})
        id_name_map = {item.id: item.name for item in ontology.items}
        ontology_sha = generate_uid_from_pbobject(ontology)
        assert annotation_enum in scene.ontologies, 'Given annotation_enum not in scene.ontologies!'
        assert scene.ontologies[annotation_enum] == ontology_sha, 'Input ontology does not match Scene ontology!'
    else:
        # Just grab the ontology in the scene
        ontology_path = os.path.join(scene_dir, ONTOLOGY_FOLDER, scene.ontologies[annotation_enum] + '.json')
        ontology = Ontology(open_ontology_pbobject(ontology_path))
        id_name_map = ontology.id_to_name
        class_counts = OrderedDict({name: 0 for name in ontology.name_to_id})

    for datum in scene.data:
        # Get the annotation file for each object
        if not datum.datum.WhichOneof('datum_oneof') == datum_type:
            continue

        datum_value = getattr(datum.datum, datum_type)
        annotations = datum_value.annotations
        if annotation_enum not in annotations:
            continue

        annotation_path = os.path.join(scene_dir, annotations[annotation_enum])

        if annotation_path.startswith('s3://'):
            annotation_object = open_remote_pb_object(annotation_path, annotation_pb)
        else:
            annotation_object = open_pbobject(annotation_path, annotation_pb)

        # Update class counts
        for annotation in annotation_object.annotations:
            class_counts[id_name_map[annotation.class_id]] += 1

    return class_counts
