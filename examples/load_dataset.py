# Copyright 2021 Toyota Research Institute.  All rights reserved.
"""Simple example script to load a scene-dataset.

Usage:
$ python examples/load_dataset.py \
     --scene-dataset-json tests/data/dgp/test_scene/scene_dataset_v1.0.json \
     --split train
"""
import argparse
import logging
import time

from tqdm import tqdm

# Required for Parallel Domain scene metadata
import dgp.contribs.pd.metadata_pb2  # pylint: disable=unused-import
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene-dataset-json', type=str, required=True, help='Path to local SceneDataset JSON.')
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        required=False,
        help='Split [train, val, test].',
        choices=['train', 'val', 'test']
    )
    parser.add_argument(  # pylint: disable=W0106
        '--datum-names',
        required=False,
        default=['LIDAR', 'CAMERA_01'],
        nargs='+',
        help='Requested datum names (case-sensitive).'
    ),
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Verbose prints.
    if args.verbose:
        logging.getLogger().setLevel(level=logging.INFO)

    # Load the dataset and build an index into the annotations requested.
    # If previously loaded/initialized, load the pre-built dataset.
    st = time.time()
    dataset = SynchronizedSceneDataset(
        scene_dataset_json=args.scene_dataset_json,
        split=args.split,
        datum_names=args.datum_names,
        requested_annotations=('bounding_box_3d', ),
        only_annotated_datums=True
    )
    print('Loading dataset took {:.2f} s'.format(time.time() - st))

    # Iterate through the dataset.
    for _ in tqdm(dataset, desc='Loading samples from the dataset'):
        pass


if __name__ == '__main__':
    main()
