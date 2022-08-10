# Copyright 2021 Toyota Research Institute.  All rights reserved.

from abc import ABC, abstractmethod

from dgp.annotations.ontology import Ontology


class AgentSnapshotList(ABC):
    """Base agent snapshot list type. All other agent snapshot lists should inherit from this type and implement
    abstractmethod.

    Parameters
    ----------
    ontology: Ontology, default:None
        Ontology object for the annotation key.

    """
    def __init__(self, ontology=None):
        if ontology is not None:
            assert isinstance(ontology, Ontology), "Invalid ontology!"
        self._ontology = ontology

    @property
    def ontology(self):
        return self._ontology

    @classmethod
    def load(cls, agent_snapshots_pb2, ontology, feature_ontology_table):
        """Loads agent snapshot list from prot into a canonical format for consumption in __getitem__ function in
        BaseDataset.
        Format/data structure for annotations will vary based on task.

        Parameters
        ----------
        agent_snapshots_pb2: object
            An agent proto message holding agent information.

        ontology: Ontology
            Ontology for given agent.

        feature_ontology_table: dict, optional
            A dictionary mapping feature type key(s) to Ontology(s), i.e.:
            {
                "agent_2d": AgentFeatureOntology[<ontology_sha>],
                "agent_3d": AgentFeatureOntology[<ontology_sha>]
            }
            Default: None.
        """

    @abstractmethod
    def render(self):
        """Return a rendering of the agent snapshot list. Expected format is a PIL.Image or np.array"""
