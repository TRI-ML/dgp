# Copyright 2021-2022 Toyota Research Institute. All rights reserved.

import os
from collections import OrderedDict

from dgp.proto.ontology_pb2 import FeatureOntology as FeatureOntologyPb2
from dgp.proto.ontology_pb2 import FeatureOntologyItem
from dgp.utils.protobuf import (
    generate_uid_from_pbobject,
    open_feature_ontology_pbobject,
    save_pbobject_as_json,
)


class FeatureOntology:
    """Feature ontology object. At bare minimum, we expect ontologies to provide:
        ID: (int) identifier for feature field name
        Name: (str) string identifier for feature field name

    Based on the task, additional fields may be populated. Refer to `dataset.proto` and `ontology.proto`
    specifications for more details. Can be constructed from file or from deserialized proto object.

    Parameters
    ----------
    feature_ontology_pb2: OntologyPb2
        Deserialized ontology object.
    """

    # Special value and class name reserved for pixels that should be ignored
    VOID_ID = 255
    VOID_CLASS = "Void"

    def __init__(self, feature_ontology_pb2):
        self._ontology = feature_ontology_pb2

        if isinstance(self._ontology, FeatureOntologyPb2):
            self._name_to_id = OrderedDict(
                sorted([(ontology_item.name, ontology_item.id) for ontology_item in self._ontology.items])
            )
            self._id_to_name = OrderedDict(
                sorted([(ontology_item.id, ontology_item.name) for ontology_item in self._ontology.items])
            )
            self._id_to_feature_value_type = OrderedDict(
                sorted([(ontology_item.id, ontology_item.feature_value_type) for ontology_item in self._ontology.items])
            )

        else:
            raise TypeError("Unexpected type {}, expected FeatureOntologyV2".format(type(self._ontology)))

        self._feature_ids = sorted(self._id_to_name.keys())
        self._feature_names = [self._id_to_name[c_id] for c_id in self._feature_ids]

    @classmethod
    def load(cls, ontology_file):
        """Construct an ontology from an ontology JSON.

        Parameters
        ----------
        ontology_file: str
            Path to ontology JSON

        Raises
        ------
        FileNotFoundError
            Raised if ontology_file does not exist.
        TypeError
            Raised if we could not read an ontology out of the ontology file.
        """
        if os.path.exists(ontology_file):
            feature_ontology_pb2 = open_feature_ontology_pbobject(ontology_file)
        else:
            raise FileNotFoundError("Could not find {}".format(ontology_file))

        if feature_ontology_pb2 is not None:
            return cls(feature_ontology_pb2)
        raise TypeError("Could not open ontology {}".format(ontology_file))

    def to_proto(self):
        """Serialize ontology. Only supports exporting in OntologyV2.

        Returns
        -------
        OntologyPb2
            Serialized ontology
        """
        return FeatureOntologyPb2(
            items=[
                FeatureOntologyItem(
                    name=name, id=feature_id, feature_value_type=self.id_to_feature_value_type[feature_id]
                ) for feature_id, name in self._id_to_name.items()
            ]
        )

    def save(self, save_dir):
        """Write out ontology items to `<sha>.json`. SHA generated from Ontology proto object.

        Parameters
        ----------
        save_dir: str
            Directory in which to save serialized ontology.

        Returns
        -------
        output_ontology_file: str
            Path to serialized ontology file.
        """
        os.makedirs(save_dir, exist_ok=True)
        return save_pbobject_as_json(self.to_proto(), save_path=save_dir)

    @property
    def num_classes(self):
        return len(self._feature_ids)

    @property
    def class_names(self):
        return self._feature_names

    @property
    def class_ids(self):
        return self._feature_ids

    @property
    def name_to_id(self):
        return self._name_to_id

    @property
    def id_to_name(self):
        return self._id_to_name

    @property
    def id_to_feature_value_type(self):
        return self._id_to_feature_value_type

    @property
    def hexdigest(self):
        """Hash object"""
        return generate_uid_from_pbobject(self.to_proto())

    def __eq__(self, other):
        return self.hexdigest == other.hexdigest

    def __repr__(self):
        return "{}[{}]".format(self.__class__.__name__, os.path.basename(self.hexdigest))


class AgentFeatureOntology(FeatureOntology):
    """Agent feature ontologies derive directly from Ontology"""
