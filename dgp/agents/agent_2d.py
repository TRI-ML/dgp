# Copyright 2021-2022 Toyota Research Institute.  All rights reserved.

import numpy as np

from dgp.agents.base_agent import AgentSnapshotList
from dgp.annotations.ontology import BoundingBoxOntology
from dgp.constants import FEATURE_TYPE_ID_TO_KEY
from dgp.utils.structures.bounding_box_2d import BoundingBox2D


class AgentSnapshot2DList(AgentSnapshotList):
    """Container for 2D agent list.

    Parameters
    ----------
    ontology: BoundingBoxOntology
        Ontology for 2D bounding box tasks.
    
    TODO : Add support for BoundingBox2DAnnotationList.
    boxlist: list[BoundingBox2D]
        List of BoundingBox2D objects. See `utils/structures/bounding_box_2d`
        for more details.
    """
    def __init__(self, ontology, boxlist):
        super().__init__(ontology)
        assert isinstance(self._ontology, BoundingBoxOntology), "Trying to load AgentSnapshot2DList with wrong type of " \
                                                                "ontology!"

        for box in boxlist:
            assert isinstance(
                box, BoundingBox2D
            ), f"Can only instantiate an agent snapshot list from a list of BoundingBox2D, not {type(box)}"
        self.boxlist = boxlist

    @classmethod
    def load(cls, agent_snapshots_pb2, ontology, feature_ontology_table):
        """Loads agent snapshot list from proto into a canonical format for consumption in __getitem__ function in
        BaseDataset.
        Format/data structure for agent types will vary based on task.

        Parameters
        ----------
        agent_snapshots_pb2: dgp.proto.agent.AgentsSlice.agent_snapshots or dgp.proto.agent.AgentTrack.agent_snapshots
            A proto message holding list of agent snapshot.

        ontology: Ontology
            Ontology for given agent.

        feature_ontology_table: dict, optional
            A dictionary mapping feature type key(s) to Ontology(s), i.e.:
            {
                "agent_2d": AgentFeatureOntology[<ontology_sha>],
                "agent_3d": AgentFeatureOntology[<ontology_sha>]
            }
            Default: None.

        Returns
        -------
        AgentSnapshot2DList
            Agent Snapshot list object instantiated from proto object.
        """
        boxlist = []
        for agent_snapshot_2d in agent_snapshots_pb2:
            feature_type = agent_snapshot_2d.agent_snapshot_2D.feature_type
            feature_ontology = feature_ontology_table[FEATURE_TYPE_ID_TO_KEY[feature_type]]
            boxlist.append(
                BoundingBox2D(
                    box=np.float32([
                        agent_snapshot_2d.agent_snapshot_2D.box.x, agent_snapshot_2d.agent_snapshot_2D.box.y,
                        agent_snapshot_2d.agent_snapshot_2D.box.w, agent_snapshot_2d.agent_snapshot_2D.box.h
                    ]),
                    class_id=ontology.class_id_to_contiguous_id[agent_snapshot_2d.agent_snapshots_2D.class_id],
                    instance_id=agent_snapshot_2d.agent_snapshot_2D.instance_id,
                    color=ontology.colormap[agent_snapshot_2d.agent_snapshot_2D.class_id],
                    attributes=dict([(feature_ontology.id_to_name[feature_id], feature)
                                     for feature_id, feature in enumerate(agent_snapshot_2d.agent_snapshot_2D.features)]
                                    ),
                )
            )

        return cls(ontology, boxlist)

    def __len__(self):
        return len(self.boxlist)

    def __getitem__(self, index):
        """Return a single 3D bounding box"""
        return self.boxlist[index]

    def render(self):
        """TODO: Batch rendering function for bounding boxes."""

    @property
    def class_ids(self):
        """Return class ID for each box, with ontology applied:
        0 is background, class IDs mapped to a contiguous set.
        """
        return np.int64([box.class_id for box in self.boxlist])

    @property
    def attributes(self):
        """Return a list of dictionaries of attribute name to value."""
        return [box.attributes for box in self.boxlist]

    @property
    def instance_ids(self):
        return np.int64([box.instance_id for box in self.boxlist])
