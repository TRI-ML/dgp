# Copyright 2021 Toyota Research Institute.  All rights reserved.
import os
from abc import ABC, abstractmethod

from dgp.annotations.ontology import Ontology


class Annotation(ABC):
    """Base annotation type. All other annotations should inherit from this type and implement
    member functions.

    Parameters
    ----------
    ontology: Ontology, default: None
        Ontology object for the annotation key
    """
    def __init__(self, ontology=None):
        if ontology is not None:
            assert isinstance(ontology, Ontology), "Invalid ontology!"
        self._ontology = ontology

    @property
    def ontology(self):
        return self._ontology

    @classmethod
    @abstractmethod
    def load(cls, annotation_file, ontology):
        """Loads annotation from file into a canonical format for consumption in __getitem__ function in BaseDataset.
        Format/data structure for annotations will vary based on task.

        Parameters
        ----------
        annotation_file: str
            Full path to annotation

        ontology: Ontology
            Ontology for given annotation
        """

    @abstractmethod
    def save(self, save_dir):
        """Serialize annotation object if possible, and saved to specified directory.
        Annotations are saved in format <save_dir>/<sha>.<ext>

        Parameters
        ----------
        save_dir: str
            Path to directory to saved annotation
        """

    @abstractmethod
    def render(self):
        """Return a rendering of the annotation. Expected format is a PIL.Image or np.array"""

    @property
    @abstractmethod
    def hexdigest(self):
        """Reproducible hash of annotation."""

    def __eq__(self, other):
        return self.hexdigest == other.hexdigest

    def __repr__(self):
        return f"{self.__class__.__name__}[{os.path.basename(self.hexdigest)}]"
