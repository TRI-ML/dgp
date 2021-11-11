# Copyright 2021 Toyota Research Institute.  All rights reserved.
import os
from collections import OrderedDict

from dgp.proto import dataset_pb2


def get_split_to_scenes(dataset):
    """
    Retrieve a dictionary of split to scenes of a DGP-compliant scene dataset.

    Parameters
    ----------
    dataset: dgp.proto.dataset_pb2.SceneDataset
        SceneDataset proto object.

    Returns
    -------
    split_to_scene_json: dict
        A dictionary of split to a list of scene JSONs.

    split_to_scene_dir: dict
        A dictionary of split to a list of scene_dirs.
    """
    split_to_scene_json = OrderedDict()
    split_to_scene_dir = OrderedDict()
    for k, v in dataset_pb2.DatasetSplit.DESCRIPTOR.values_by_number.items():
        scene_jsons = dataset.scene_splits[k].filenames
        split_to_scene_json[v.name] = scene_jsons
        split_to_scene_dir[v.name] = [os.path.dirname(scene_json) for scene_json in scene_jsons]
        assert len(set(split_to_scene_dir[v.name])) == len(split_to_scene_dir[v.name])
    return split_to_scene_json, split_to_scene_dir
