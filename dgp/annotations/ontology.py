# Copyright 2021 Toyota Research Institute.  All rights reserved.
import os
from collections import OrderedDict

import numpy as np

from dgp.proto.dataset_pb2 import Ontology as OntologyV1Pb2
from dgp.proto.ontology_pb2 import Ontology as OntologyV2Pb2
from dgp.proto.ontology_pb2 import OntologyItem
from dgp.utils.protobuf import (
    generate_uid_from_pbobject,
    open_ontology_pbobject,
    save_pbobject_as_json,
)


class Ontology:
    """Ontology object. At bare minimum, we expect ontologies to provide:
        ID: (int) identifier for class
        Name: (str) string identifier for class
        Color: (tuple) color RGB tuple

    Based on the task, additional fields may be populated. Refer to `dataset.proto` and `ontology.proto`
    specifications for more details. Can be constructed from file or from deserialized proto object.

    Parameters
    ----------
    ontology_pb2: [OntologyV1Pb2,OntologyV2Pb2]
        Deserialized ontology object.
    """

    # Special value and class name reserved for pixels that should be ignored
    VOID_ID = 255
    VOID_CLASS = "Void"

    def __init__(self, ontology_pb2):
        self._ontology = ontology_pb2

        # NOTE: OntologyV1 (defined in dataset.proto) to be deprecated
        if isinstance(self._ontology, OntologyV1Pb2):
            self._name_to_id = OrderedDict(sorted(self._ontology.name_to_id.items()))
            self._id_to_name = OrderedDict(sorted(self._ontology.id_to_name.items()))
            self._colormap = OrderedDict(
                sorted([(_id, (_color.r, _color.g, _color.b)) for _id, _color in self._ontology.colormap.items()])
            )
            self._isthing = OrderedDict(sorted(self._ontology.isthing.items()))

        elif isinstance(self._ontology, OntologyV2Pb2):
            self._name_to_id = OrderedDict(
                sorted([(ontology_item.name, ontology_item.id) for ontology_item in self._ontology.items])
            )
            self._id_to_name = OrderedDict(
                sorted([(ontology_item.id, ontology_item.name) for ontology_item in self._ontology.items])
            )
            self._colormap = OrderedDict(
                sorted([(ontology_item.id, (ontology_item.color.r, ontology_item.color.g, ontology_item.color.b))
                        for ontology_item in self._ontology.items])
            )
            self._isthing = OrderedDict(
                sorted([(ontology_item.id, ontology_item.isthing) for ontology_item in self._ontology.items])
            )

        else:
            raise TypeError("Unexpected type {}, expected OntologyV1 or OntologyV2".format(type(self._ontology)))

        self._class_ids = sorted(self._id_to_name.keys())
        self._class_names = [self._id_to_name[c_id] for c_id in self._class_ids]

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
        Exception
            Raised if we could not open the ontology file for some reason.
        """
        if os.path.exists(ontology_file):
            ontology_pb2 = open_ontology_pbobject(ontology_file)
        else:
            raise FileNotFoundError("Could not find {}".format(ontology_file))

        if ontology_pb2 is not None:
            return cls(ontology_pb2)
        raise Exception('Could not open ontology {}'.format(ontology_file))

    def to_proto(self):
        """Serialize ontology. Only supports exporting in OntologyV2.

        Returns
        -------
        OntologyV2Pb2
            Serialized ontology
        """
        return OntologyV2Pb2(
            items=[
                OntologyItem(
                    name=name,
                    id=class_id,
                    color=OntologyItem.
                    Color(r=self._colormap[class_id][0], g=self._colormap[class_id][1], b=self._colormap[class_id][2]),
                    isthing=self._isthing[class_id]
                ) for class_id, name in self._id_to_name.items()
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
        return len(self._class_ids)

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_ids(self):
        return self._class_ids

    @property
    def name_to_id(self):
        return self._name_to_id

    @property
    def id_to_name(self):
        return self._id_to_name

    @property
    def colormap(self):
        return self._colormap

    @property
    def isthing(self):
        return self._isthing

    @property
    def hexdigest(self):
        """Hash object"""
        return generate_uid_from_pbobject(self.to_proto())

    def __eq__(self, other):
        return self.hexdigest == other.hexdigest

    def __repr__(self):
        return "{}[{}]".format(self.__class__.__name__, os.path.basename(self.hexdigest))


class BoundingBoxOntology(Ontology):
    """Implements lookup tables specific to 2D bounding box tasks.

    Parameters
    ----------
    ontology_pb2: [OntologyV1Pb2,OntologyV2Pb2]
        Deserialized ontology object.
    """
    def __init__(self, ontology_pb2):
        super().__init__(ontology_pb2)

        # Extract IDs for `thing` (object) classes
        self._thing_class_ids = [class_id for class_id, isthing in self._isthing.items() if isthing]

        # Map`thing` class IDs, which may not be contiguous in the raw ontology,
        # to a contiguous set starting at 1 (as 0 is reserved for background)
        self._class_id_to_contiguous_id = OrderedDict(
            (class_id, contiguous_id + 1) for contiguous_id, class_id in enumerate(self._thing_class_ids)
        )

        # Contiguous ID map back to original class ID
        self._contiguous_id_to_class_id = OrderedDict(
            (contiguous_id, class_id) for class_id, contiguous_id in self._class_id_to_contiguous_id.items()
        )

        # Direct lookups from contiguous ID to name and color as well
        self._contiguous_id_to_name = OrderedDict((contiguous_id, self._id_to_name[class_id])
                                                  for contiguous_id, class_id in self._contiguous_id_to_class_id.items()
                                                  )
        self._name_to_contiguous_id = OrderedDict(
            (name, contiguous_id) for contiguous_id, name in self._contiguous_id_to_name.items()
        )
        self._contiguous_id_colormap = OrderedDict(
            (contiguous_id, self._colormap[class_id])
            for contiguous_id, class_id in self._contiguous_id_to_class_id.items()
        )
        self._class_names = [self._id_to_name[c_id] for c_id in self._thing_class_ids]

    @property
    def num_classes(self):
        return len(self._thing_class_ids)

    @property
    def class_names(self):
        return self._class_names

    @property
    def thing_class_ids(self):
        return self._thing_class_ids

    @property
    def class_id_to_contiguous_id(self):
        return self._class_id_to_contiguous_id

    @property
    def contiguous_id_to_class_id(self):
        return self._contiguous_id_to_class_id

    @property
    def contiguous_id_to_name(self):
        return self._contiguous_id_to_name

    @property
    def name_to_contiguous_id(self):
        return self._name_to_contiguous_id

    @property
    def contiguous_id_colormap(self):
        return self._contiguous_id_colormap


class AgentBehaviorOntology(BoundingBoxOntology):
    """Agent behavior ontologies derive directly from bounding box ontologies"""


class KeyPointOntology(BoundingBoxOntology):
    """Keypoint ontologies derive directly from bounding box ontologies"""


class KeyLineOntology(BoundingBoxOntology):
    """Keyline ontologies derive directly from bounding box ontologies"""


class InstanceSegmentationOntology(BoundingBoxOntology):
    """Instance segmentation ontologies derive directly from bounding box ontologies"""


class SemanticSegmentationOntology(Ontology):
    """Implements lookup tables for semantic segmentation

    Parameters
    ----------
    ontology_pb2: [OntologyV1Pb2,OntologyV2Pb2]
        Deserialized ontology object.
    """
    def __init__(self, ontology_pb2):
        super().__init__(ontology_pb2)

        # Map class IDs, which may not be contiguous in the raw ontology, to a contiguous set starting at 0.
        self._class_id_to_contiguous_id = OrderedDict(
            (class_id, contiguous_id) for contiguous_id, class_id in enumerate(self._class_ids)
        )

        # Contiguous ID map back to original class ID
        self._contiguous_id_to_class_id = OrderedDict(
            (contiguous_id, class_id) for class_id, contiguous_id in self._class_id_to_contiguous_id.items()
        )

        # Direct lookups from contiguous ID to name and color as well
        self._contiguous_id_to_name = OrderedDict((contiguous_id, self._id_to_name[class_id])
                                                  for contiguous_id, class_id in self._contiguous_id_to_class_id.items()
                                                  )
        self._name_to_contiguous_id = OrderedDict(
            (name, contiguous_id) for contiguous_id, name in self._contiguous_id_to_name.items()
        )
        self._contiguous_id_colormap = OrderedDict(
            (contiguous_id, self._colormap[class_id])
            for contiguous_id, class_id in self._contiguous_id_to_class_id.items()
        )

        # Build lookup table to map raw label data to training IDs
        self._label_lookup = np.ones(max(self.class_ids) + 1, dtype=np.uint8) * self.VOID_ID
        for class_id, contiguous_id in self._class_id_to_contiguous_id.items():
            self._label_lookup[class_id] = contiguous_id

    @property
    def label_lookup(self):
        return self._label_lookup

    @property
    def class_id_to_contiguous_id(self):
        return self._class_id_to_contiguous_id

    @property
    def contiguous_id_to_class_id(self):
        return self._contiguous_id_to_class_id

    @property
    def contiguous_id_to_name(self):
        return self._contiguous_id_to_name

    @property
    def name_to_contiguous_id(self):
        return self._name_to_contiguous_id

    @property
    def contiguous_id_colormap(self):
        return self._contiguous_id_colormap
