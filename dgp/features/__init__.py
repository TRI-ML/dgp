# Copyright 2021-2022 Toyota Research Institute. All rights reserved.

from dgp.features.feature_ontology import AgentFeatureOntology

# Ontology handlers for each annotation type
FEATURE_ONTOLOGY_REGISTRY = {
    "agent_2d": AgentFeatureOntology,
    "agent_3d": AgentFeatureOntology,
    "ego_intention": AgentFeatureOntology,
    "parked_car": AgentFeatureOntology
}
