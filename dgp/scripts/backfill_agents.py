#!/usr/bin/env python
# Copyright 2021 Toyota Research Institute. All rights reserved.
"""Script that backfill agents into original DGP format."""
import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

from dgp import (AGENT_FOLDER, FEATURE_ONTOLOGY_FOLDER,
                           TRI_DGP_AGENT_TRACKS_JSON_NAME,
                           TRI_DGP_AGENTS_JSON_NAME,
                           TRI_DGP_AGENTS_SLICES_JSON_NAME,
                           TRI_DGP_S3_BUCKET_URL, TRI_RAW_S3_BUCKET_URL)
from dgp.datasets.prediction_dataset import PredictionAgentDataset
from dgp.proto import dataset_pb2, features_pb2
from dgp.proto.agent_pb2 import (AgentGroup, AgentsSlice,
                                           AgentsSlices, AgentTracks)
from dgp.proto.ontology_pb2 import FeatureOntology, FeatureOntologyItem
from dgp.utils.dataset_conversion import get_date, get_datetime_proto
from dgp.utils.protobuf import (generate_uid_from_pbobject,
                                          open_pbobject,
                                          save_pbobject_as_json)
from utils.s3 import s3_copy

TRIPCCOntology = FeatureOntology(
    items=[
        FeatureOntologyItem(name='ParkedVehicleState', id=0, feature_value_type=0)
    ]
)


class AgentBackfillDemo:
    """
    Class to backfill agents information into ioda scene dataset to create a
    demo dataset.
    """

    def __init__(self, scene_dataset_json):
        self.agent_dataset_name = "agents_pcc_mini"
        self.version = "1"
        self.description = 'agents of pcc mini dataset'
        self.EMAIL = 'chao.fang@tri.global'
        self.public = False
        self.scene_dataset_json = scene_dataset_json
        self.scene_dataset = open_pbobject(scene_dataset_json, dataset_pb2.SceneDataset)
        self.agents_dataset_pb2 = dataset_pb2.Agents()
        self.local_output_path = Path(scene_dataset_json).parent.as_posix()
        self.ontologies = {features_pb2.PARKED_CAR: TRIPCCOntology}
        self.populate_metadata()

    def populate_metadata(self):
        """Populate boilerplate fields agent metadata"""
        self.agents_dataset_pb2.metadata.name = self.agent_dataset_name
        self.agents_dataset_pb2.metadata.version = self.version
        self.agents_dataset_pb2.metadata.creation_date = get_date()
        self.agents_dataset_pb2.metadata.creator = self.EMAIL

        self.agents_dataset_pb2.metadata.bucket_path.value = os.path.join(TRI_DGP_S3_BUCKET_URL,
                                                                          self.agent_dataset_name)
        self.agents_dataset_pb2.metadata.raw_path.value = os.path.join(TRI_RAW_S3_BUCKET_URL, self.agent_dataset_name)

        self.agents_dataset_pb2.metadata.description = self.description
        self.agents_dataset_pb2.metadata.origin = self.agents_dataset_pb2.metadata.PUBLIC if self.public \
            else self.agents_dataset_pb2.metadata.INTERNAL

    def generate(self):
        for (split_type, split) in zip([dataset_pb2.TRAIN, dataset_pb2.TEST, dataset_pb2.VAL],
                                       ['train', 'test', 'val']):
            if split_type not in self.scene_dataset.scene_splits:
                continue

            logging.info('Processing {}'.format(split))
            original_dataset = PredictionAgentDataset(
                self.scene_dataset_json,
                split=split,
                datum_names=('LIDAR',),
                requested_main_agent_types=('Person',
                                            'Truck',
                                            'Car',
                                            'Bus/RV/Caravan',
                                            'Motorcycle',
                                            'Wheeled Slow',
                                            'Train',
                                            'Towed Object',
                                            'Bicycle',
                                            'Trailer',),
                batch_per_agent=True
            )
            self.backfill(original_dataset, split_type)
        agent_json_path = self.write_agents()

        return agent_json_path

    def backfill(self, original_dataset, split_type):
        """Backfill agent information.

        Parameters
        ----------
        original_dataset: PredictionAgentDataset
            DGP PredictionAgentDataset object

        split_type: DatasetSplit
            Split of dataset to read ("train" | "val" | "test" | "train_overfit").

        """
        # Map from scene idx to list agent
        scene_agent_map = defaultdict(list)
        for agent_idx, agent_track in enumerate(original_dataset.dataset_agent_index):
            scene_idx = agent_track[1]
            scene_agent_map[scene_idx].append(agent_idx)
        for scene_idx, scene in enumerate(original_dataset.scenes):
            output_scene_dirname = scene.directory
            agent_pb2 = AgentGroup()
            agent_pb2.feature_ontologies[features_pb2.PARKED_CAR] = \
                generate_uid_from_pbobject(self.ontologies[features_pb2.PARKED_CAR])
            for k, v in scene.scene.ontologies.items():
                agent_pb2.agent_ontologies[k] = v
            agent_tracks_pb2 = AgentTracks()
            agents_slices_pb2 = AgentsSlices()
            sample_agent_snapshots_dict = defaultdict(AgentsSlice)
            for agent_idx in scene_agent_map[scene_idx]:
                main_agent_id, scene_idx, sample_idx_in_scene_start, _, _ = \
                    original_dataset.dataset_agent_index[agent_idx]
                agent_track_original = original_dataset[agent_idx]
                agent_track = agent_tracks_pb2.agent_tracks.add()
                agent_track.instance_id = main_agent_id
                agent_track.class_id = original_dataset.dataset_metadata.ontology_table[
                    original_dataset.annotation_reference].contiguous_id_to_class_id[
                    agent_track_original[0][0]['bounding_box_3d'][
                        agent_track_original[0][0]['main_agent_idx']].class_id]
                sample_idx = sample_idx_in_scene_start
                for agent_snapshot_original in agent_track_original:
                    try:
                        box = agent_snapshot_original[0]['bounding_box_3d'][
                            int(agent_snapshot_original[0]['main_agent_idx'])]
                    except TypeError:  # pylint: disable=bare-except
                        sample_idx = sample_idx + 1
                        continue
                    if sample_idx not in sample_agent_snapshots_dict:
                        sample_agent_snapshots_dict[sample_idx].slice_id.CopyFrom(scene.samples[sample_idx].id)
                        sample_agent_snapshots_dict[sample_idx].slice_id.index = sample_idx
                    agent_snapshot = agent_track.agent_snapshots.add()
                    agent_snapshot.agent_snapshot_3D.box.CopyFrom(box.to_proto())
                    agent_snapshot.slice_id.CopyFrom(scene.samples[sample_idx].id)
                    agent_snapshot.slice_id.index = sample_idx
                    agent_snapshot.agent_snapshot_3D.class_id = agent_track.class_id

                    agent_snapshot.agent_snapshot_3D.instance_id = main_agent_id
                    # The feature fields need to backfill from
                    agent_snapshot.agent_snapshot_3D.features.extend(["dynamic"])

                    # 5 for parked car features
                    agent_snapshot.agent_snapshot_3D.feature_type = features_pb2.PARKED_CAR

                    # Handle agent slices
                    sample_agent_snapshots_dict[sample_idx].agent_snapshots.extend([agent_snapshot])

                    sample_idx = sample_idx + 1

            for sample_idx in range(len(scene.samples)):
                if sample_idx in sample_agent_snapshots_dict:
                    agents_slices_pb2.agents_slices.extend([sample_agent_snapshots_dict[sample_idx]])
                else:
                    agents_slices_pb2.agents_slices.extend([AgentsSlice()])

            # Save agent_tracks file
            os.makedirs(os.path.join(
                self.local_output_path, output_scene_dirname, AGENT_FOLDER), exist_ok=True)
            agent_tracks_filename = os.path.join(
                AGENT_FOLDER,
                TRI_DGP_AGENT_TRACKS_JSON_NAME.format(track_hash=generate_uid_from_pbobject(agents_slices_pb2))
            )
            save_pbobject_as_json(agent_tracks_pb2, os.path.join(
                self.local_output_path, output_scene_dirname, agent_tracks_filename))
            agent_pb2.agent_tracks_file = agent_tracks_filename

            # Save agents_slices file
            agents_slices_filename = os.path.join(AGENT_FOLDER,
                                                  TRI_DGP_AGENTS_SLICES_JSON_NAME.format(
                                                      slice_hash=generate_uid_from_pbobject(agent_tracks_pb2))
                                                  )
            save_pbobject_as_json(agents_slices_pb2, os.path.join(
                self.local_output_path, output_scene_dirname, agents_slices_filename))
            agent_pb2.agents_slices_file = agents_slices_filename

            # Save AgentGroup
            agent_pb2.log = scene.scene.log
            agent_pb2.name = scene.scene.name
            agent_pb2.creation_date.CopyFrom(get_datetime_proto())
            agents_group_filename = os.path.join(
                output_scene_dirname,
                TRI_DGP_AGENTS_JSON_NAME.format(agent_hash=generate_uid_from_pbobject(agent_pb2))
            )
            self.agents_dataset_pb2.agents_splits[split_type].filenames.append(agents_group_filename)
            save_pbobject_as_json(agent_pb2, os.path.join(self.local_output_path, agents_group_filename))

            # Populate ontologies
            os.makedirs(os.path.join(
                self.local_output_path, output_scene_dirname, FEATURE_ONTOLOGY_FOLDER), exist_ok=True)
            for feature_type, ontology_id in agent_pb2.feature_ontologies.items():
                ontology_filename = os.path.join(
                    self.local_output_path, output_scene_dirname, FEATURE_ONTOLOGY_FOLDER,
                    "{}.json".format(ontology_id)
                )
                save_pbobject_as_json(
                    self.ontologies[feature_type], ontology_filename)

    def write_agents(self, upload=False):
        """Write the final scene dataset JSON.
        Parameters
        ----------
        upload: bool, default: False
            If true, upload the dataset to the scene pool in s3.
        Returns
        -------
        scene_dataset_json_path: str
            Path of the scene dataset JSON file created.
        """
        agent_dataset_json_path = os.path.join(
            self.local_output_path,
            '{}_v{}.json'.format(self.agents_dataset_pb2.metadata.name, self.agents_dataset_pb2.metadata.version)
        )
        save_pbobject_as_json(self.agents_dataset_pb2, agent_dataset_json_path)

        # Printing SceneDataset scene counts per split (post-merging)
        logging.info('-' * 80)
        logging.info(
            'Output SceneDataset {} has: {} train, {} val, {} test'.format(
                agent_dataset_json_path, len(self.agents_dataset_pb2.agents_splits[dataset_pb2.TRAIN].filenames),
                len(self.agents_dataset_pb2.agents_splits[dataset_pb2.VAL].filenames),
                len(self.agents_dataset_pb2.agents_splits[dataset_pb2.TEST].filenames)
            )
        )

        s3_path = os.path.join(
            self.agents_dataset_pb2.metadata.bucket_path.value, os.path.basename(agent_dataset_json_path)
        )
        if upload:
            s3_copy(agent_dataset_json_path, s3_path)

        else:
            logging.info(
                'Upload the DGP-compliant agent dataset JSON to s3 via `aws s3 cp --acl bucket-owner-full-control {} '
                '{}`'.format(agent_dataset_json_path, s3_path)
            )

        return agent_dataset_json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=True
    )
    parser.add_argument("-i", "--scene-dataset-json", help="Input dataset json", required=True)
    parser.add_argument('--verbose', action='store_true')
    args, other_args = parser.parse_known_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    dataset = AgentBackfillDemo(args.scene_dataset_json)
    dataset.generate()
