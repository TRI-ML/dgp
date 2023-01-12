# Copyright 2021-2022 Toyota Research Institute.  All rights reserved.
import hashlib
import json
import logging
import os
from io import StringIO

from google.protobuf.json_format import MessageToDict, Parse

from dgp.proto.dataset_pb2 import Ontology as OntologyV1Pb2
from dgp.proto.ontology_pb2 import FeatureOntology as FeatureOntologyPb2
from dgp.proto.ontology_pb2 import Ontology as OntologyV2Pb2
from dgp.proto.scene_pb2 import Scene
from dgp.utils.cloud.s3 import (
    convert_uri_to_bucket_path,
    get_string_from_s3_file,
)


def open_pbobject(path, pb_class):
    """Load JSON as a protobuf (pb2) object.

    Any calls to load protobuf objects from JSON in this repository should be through this function.
    Returns `None` if the loading failed.

    Parameters
    ----------
    path: str
        Local JSON file path or remote scene dataset JSON URI to load.

    pb_class: object
        Protobuf pb2 object we want to load into.

    Returns
    ----------
    pb_object: pb2 object
        Desired pb2 object to be opened.
    """
    assert path.endswith(".json"), 'File extension for {} needs to be json.'.format(path)
    if path.startswith('s3://'):
        return open_remote_pb_object(path, pb_class)
    assert os.path.exists(path), f'Path not found: {path}'
    with open(path, 'r', encoding='UTF-8') as json_file:
        pb_object = Parse(json_file.read(), pb_class())
    return pb_object


def parse_pbobject(source, pb_class):
    """Like open_pboject but source can be a path or a bytestring

    Parameters
    ----------
    source: str or bytes
        Local JSON file path, remote s3 path to object, or bytestring of serialized object

    pb_class: object
        Protobuf pb2 object we want to load into.

    Returns
    -------
    pb_object: pb2 object or None
        Desired pb2 ojbect to be parsed or None if loading fails
    """
    if isinstance(source, str):
        return open_pbobject(source, pb_class)
    elif isinstance(source, bytes):
        pb_object = pb_class()
        pb_object.ParseFromString(source)
        return pb_object
    else:
        logging.error(f'cannot parse type {type(source)}')


def open_remote_pb_object(s3_object_uri, pb_class):
    """Load JSON as a protobuf (pb2) object from S3 remote

    Parameters
    ----------
    s3_object_uri: str
        Remote scene dataset JSON URI.

    pb_class: object
        Protobuf pb2 object we want to load into.

    Returns
    ----------
    pb_object: pb2 object
        Desired pb2 object to be opened.

    Raises
    ------
    ValueError
        Raised if s3_object_uri is not a valid S3 path.
    """
    if s3_object_uri.startswith('s3://'):
        bucket_name, s3_base_path = convert_uri_to_bucket_path(s3_object_uri)
    else:
        raise ValueError("Expected path to S3 bucket but got {}".format(s3_object_uri))

    pb_object = Parse(get_string_from_s3_file(bucket_name, s3_base_path), pb_class())

    return pb_object


def save_pbobject_as_json(pb_object, save_path):
    """
    Save protobuf (pb2) object to JSON file with our standard indent, key ordering, and other
    settings.

    Any calls to save protobuf objects to JSON in this repository should be through this function.

    Parameters
    ----------
    pb_object: object
        Protobuf pb2 object we want to save to file

    save_path: str
        If save path is a JSON, serialized object is saved to that path. If save path is directory,
        the `pb_object` is saved in <save_path>/<pb_object_sha>.json.

    Returns
    -------
    save_path: str
        Returns path to saved pb_object JSON
    """
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, generate_uid_from_pbobject(pb_object) + ".json")

    assert save_path.endswith(".json"), 'File extension for {} needs to be json.'.format(save_path)
    with open(save_path, "w", encoding='UTF-8') as _f:
        json.dump(
            MessageToDict(pb_object, including_default_value_fields=True, preserving_proto_field_name=True),
            _f,
            indent=2,
            sort_keys=True
        )
    return save_path


def open_ontology_pbobject(ontology_file):
    """Open ontology objects, first attempt to open V2 before trying V1.

    Parameters
    ----------
    ontology_file: str or bytes
        JSON ontology file path to load or bytestring.

    Returns
    -------
    ontology: Ontology object
        Desired Ontology pb2 object to be opened (either V2 or V1). Returns
        None if neither fails to load.
    """
    try:
        ontology = parse_pbobject(ontology_file, OntologyV2Pb2)
        if ontology is not None:
            logging.info('Successfully loaded Ontology V2 spec.')
            return ontology
    except Exception:
        logging.error('Failed to load ontology file with V2 spec, trying V1 spec.')
    try:
        ontology = parse_pbobject(ontology_file, OntologyV1Pb2)
        if ontology is not None:
            logging.info('Successfully loaded Ontology V1 spec.')
            return ontology
    except Exception:
        if isinstance(ontology_file, str):
            logging.error('Failed to load ontology file' + ontology_file + 'with V1 spec also, returning None.')
        else:
            logging.error('Failed to load ontology file with V1 spec also, returning None.')


def open_feature_ontology_pbobject(ontology_file):
    """Open feature ontology objects.

    Parameters
    ----------
    ontology_file: str
        JSON ontology file path to load.

    Returns
    -------
    ontology: FeatureOntology object
        Desired Feature Ontology pb2 object to be opened. Returns
        None if neither fails to load.
    """
    try:
        ontology = open_pbobject(ontology_file, FeatureOntologyPb2)
        if ontology is not None:
            logging.info('Successfully loaded FeatureOntology spec.')
            return ontology
    except Exception:
        logging.error('Failed to load ontology file' + ontology_file + '.')


def generate_uid_from_pbobject(pb_object):
    """Given a pb object, return the deterministic SHA1 hash hexdigest.
    Used for creating unique IDs.

    Parameters
    ----------
    pb_object: object
        A protobuf pb2 object to be hashed.

    Returns
    -------
    Hexdigest of annotation content
    """
    json_string = json.dumps(
        MessageToDict(pb_object, including_default_value_fields=True, preserving_proto_field_name=True),
        indent=2,
        sort_keys=True
    )
    out = StringIO()
    out.write(json_string)
    uid = hashlib.sha1(out.getvalue().encode('utf-8')).hexdigest()
    out.close()
    return uid


def get_latest_scene(s3_scene_jsons):
    """From a list of 'scene.json' and/or 'scene_<sha1>.json' paths in s3,
    return a Scene object for the one with the latest timestamp.
    Parameters
    ----------
    s3_scene_jsons: List[str]
        List of 'scene.json' or 'scene_<sha1>.json' paths in s3
    Returns
    -------
    latest_scene: dgp.proto.scene_pb2.Scene
        Scene pb object with the latest creation timestamp.
    scene_json_path: str
        S3 Path to the latest scene JSON.
    Notes
    -----
    This function can be called on the output:
        out, _ = s3_recursive_list(os.path.join(scene_s3_dir, 'scene')
    which is grabbing all 'scene*' files from the Scene directory
    """
    # Fetch all 'scene*.json' files and load Scenes
    scenes = [open_remote_pb_object(scene_json, Scene) for scene_json in s3_scene_jsons]

    # Find Scene with latest creation timestamp
    creation_ts = [_s.creation_date.ToMicroseconds() for _s in scenes]
    index = creation_ts.index(max(creation_ts))
    return scenes[index], s3_scene_jsons[index]
