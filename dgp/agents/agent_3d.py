# Copyright 2021-2022 Toyota Research Institute.  All rights reserved.

import numpy as np

from dgp.agents.base_agent import AgentSnapshotList
from dgp.annotations.ontology import BoundingBoxOntology
from dgp.constants import FEATURE_TYPE_ID_TO_KEY
from dgp.utils.camera import Camera
from dgp.utils.pose import Pose
from dgp.utils.structures.bounding_box_3d import BoundingBox3D


class AgentSnapshot3DList(AgentSnapshotList):
    """Container for 3D agent list.

    Parameters
    ----------
    ontology: BoundingBoxOntology
        Ontology for 3D bounding box tasks.

    boxlist: list[BoundingBox3D]
        List of BoundingBox3D objects. See `utils/structures/bounding_box_3d`
        for more details.
    """
    def __init__(self, ontology, boxlist):
        super().__init__(ontology)
        assert isinstance(self._ontology, BoundingBoxOntology), "Trying to load AgentSnapshot3DList with wrong type of " \
                                                                "ontology!"

        for box in boxlist:
            assert isinstance(
                box, BoundingBox3D
            ), f"Can only instantiate an agent snapshot list from a list of BoundingBox3D, not {type(box)}"
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

        feature_ontology_table: dict
            A dictionary mapping feature type key(s) to Ontology(s), i.e.:
            {
                "agent_2d": AgentFeatureOntology[<ontology_sha>],
                "agent_3d": AgentFeatureOntology[<ontology_sha>]
            }

        Returns
        -------
        AgentSnapshot3DList
            Agent Snapshot list object instantiated from proto object.
        """
        boxlist = []
        for agent_snapshot_3d in agent_snapshots_pb2:
            feature_type = agent_snapshot_3d.agent_snapshot_3D.feature_type
            feature_ontology = feature_ontology_table[FEATURE_TYPE_ID_TO_KEY[feature_type]]
            boxlist.append(
                BoundingBox3D(
                    pose=Pose.load(agent_snapshot_3d.agent_snapshot_3D.box.pose),
                    sizes=np.float32([
                        agent_snapshot_3d.agent_snapshot_3D.box.width, agent_snapshot_3d.agent_snapshot_3D.box.length,
                        agent_snapshot_3d.agent_snapshot_3D.box.height
                    ]),
                    class_id=ontology.class_id_to_contiguous_id[agent_snapshot_3d.agent_snapshot_3D.class_id],
                    instance_id=agent_snapshot_3d.agent_snapshot_3D.instance_id,
                    sample_idx=agent_snapshot_3d.slice_id.index,
                    color=ontology.colormap[agent_snapshot_3d.agent_snapshot_3D.class_id],
                    attributes=dict([(feature_ontology.id_to_name[feature_id], feature)
                                     for feature_id, feature in enumerate(agent_snapshot_3d.agent_snapshot_3D.features)]
                                    ),
                )
            )

        return cls(ontology, boxlist)

    def __len__(self):
        return len(self.boxlist)

    def __getitem__(self, index):
        """Return a single 3D bounding box"""
        return self.boxlist[index]

    def render(self, image, camera, line_thickness=2, font_scale=0.5):
        """Render the 3D boxes in this agents on the image in place

        Parameters
        ----------
        image: np.ndarray
            Image (H, W, C) to render the bounding box onto. We assume the input image is in *RGB* format.
            Data type is uint8.

        camera: dgp.utils.camera.Camera
            Camera used to render the bounding box.

        line_thickness: int, optional
            Thickness of bounding box lines. Default: 2.

        font_scale: float, optional
            Font scale used in text labels. Default: 0.5.

        Raises
        ------
        ValueError
            Raised if `image` is not a 3-channel uint8 numpy array.
        TypeError
            Raised if `camera` is not an instance of Camera.
        """
        if (
            not isinstance(image, np.ndarray) or image.dtype != np.uint8 or len(image.shape) != 3 or image.shape[2] != 3
        ):
            raise ValueError('`image` needs to be a 3-channel uint8 numpy array')
        if not isinstance(camera, Camera):
            raise TypeError('`camera` should be of type Camera')
        for box in self.boxlist:
            box.render(
                image,
                camera,
                line_thickness=line_thickness,
                class_name=self._ontology.contiguous_id_to_name[box.class_id],
                font_scale=font_scale
            )

    @property
    def poses(self):
        """Get poses for bounding boxes in agent list."""
        return [box.pose for box in self.boxlist]

    @property
    def sizes(self):
        return np.float32([box.sizes for box in self.boxlist])

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
