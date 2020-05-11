# Copyright 2019 Toyota Research Institute.  All rights reserved.
import json
import logging

from google.protobuf.json_format import MessageToDict, Parse

from dgp.proto.dataset_pb2 import Ontology as OntologyV1Pb2
from dgp.proto.ontology_pb2 import Ontology as OntologyV2Pb2


def open_pbobject(path, pb_class, verbose=True):
    """Load JSON as a protobuf (pb2) object.

    Any calls to load protobuf objects from JSON in this repository should be through this function.
    Returns `None` if the loading failed.

    Parameters
    ----------
    path: str
        JSON file path to load

    pb_class: pb2 object class
        Protobuf object we want to load into.

    verbose: bool, default: True
        Verbose prints on failure

    Returns
    ----------
    pb_object: pb2 object
        Desired pb2 object to be opened.
    """
    assert path.endswith(".json"), 'File extension for {} needs to be json.'.format(path)
    with open(path, 'r') as json_file:
        try:
            pb_object = Parse(json_file.read(), pb_class())
        except Exception as e:
            if verbose:
                print('open_pbobject: Failed to load pbobject {}'.format(e))
            return None
    return pb_object


def save_pbobject_as_json(pb_object, save_path):
    """
    Save protobuf (pb2) object to JSON file with our standard indent, key ordering, and other
    settings.

    Any calls to save protobuf objects to JSON in this repository should be through this function.

    Parameters
    ----------
    pb_object: pb2 object
        Protobuf object we want to save to file

    save_path: str
        JSON file path to save to
    """
    assert save_path.endswith(".json"), 'File extension for {} needs to be json.'.format(save_path)
    with open(save_path, "w") as _f:
        json.dump(
            MessageToDict(pb_object, including_default_value_fields=True, preserving_proto_field_name=True),
            _f,
            indent=2,
            sort_keys=True
        )


def open_ontology_pbobject(ontology_file, verbose=True):
    """Open ontology objects, first attempt to open V2 before trying V1.

    Parameters
    ----------
    ontology_file: str
        JSON ontology file path to load.

    verbose: bool, default: True
        Verbose prints on failure.

    Returns
    ----------
    ontology: Ontology object
        Desired Ontology pb2 object to be opened (either V2 or V1). Returns
        None if neither fails to load.
    """
    ontology = open_pbobject(ontology_file, OntologyV2Pb2, verbose=verbose)
    if ontology is not None:
        logging.info('Successfully loaded Ontology V2 spec.')
        return ontology
    logging.info('Failed to load ontology file with V2 spec, trying V1 spec.')
    ontology = open_pbobject(ontology_file, OntologyV1Pb2, verbose=verbose)
    if ontology is not None:
        logging.info('Successfully loaded Ontology V1 spec.')
        return ontology
    logging.info('Failed to load ontology file with V1 spec also, returning None.')
