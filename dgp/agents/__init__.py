# Copyright 2021-2022 Toyota Research Institute. All rights reserved.

from dgp.agents.agent_2d import AgentSnapshot2DList  # isort:skip
from dgp.agents.agent_3d import AgentSnapshot3DList  # isort:skip

# Agents objects for each agent type
AGENT_REGISTRY = {"agent_2d": AgentSnapshot2DList, "agent_3d": AgentSnapshot3DList}

# Annotation groups for each annotation type: 2d/3d
AGENT_TYPE_TO_ANNOTATION_GROUP = {"agent_2d": "2d", "agent_3d": "3d"}

AGENT_TYPE_TO_ANNOTATION_TYPE = {"agent_2d": "bounding_box_2d", "agent_3d": "bounding_box_3d"}

ANNOTATION_TYPE_TO_AGENT_TYPE = {"bounding_box_2d": "agent_2d", "bounding_box_3d": "agent_3d"}
