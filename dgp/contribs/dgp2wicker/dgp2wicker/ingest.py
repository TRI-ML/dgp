# Copyright 2022 Woven Planet.  All rights reserved.
"""DGP to Wicker ingestion methods
"""
# pylint: disable=missing-param-doc
import logging
import os
import tempfile
import time
import traceback
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import dgp2wicker.serializers as ws
import numpy as np
import pyspark
import wicker
import wicker.plugins.spark as wsp
from wicker.schema import IntField, StringField

from dgp.datasets import ParallelDomainScene, SynchronizedScene
from dgp.proto import dataset_pb2
from dgp.proto.dataset_pb2 import SceneDataset
from dgp.utils.cloud.s3 import sync_dir
from dgp.utils.protobuf import open_pbobject

PC_DATUMS = ('point_cloud', 'radar_point_cloud')
NON_PC_FIELDS = (
    'depth', 'semantic_segmentation_2d', 'instance_segmentation_2d', 'bounding_box_2d', 'key_point_2d', 'key_line_2d'
)
ILLEGAL_COMBINATIONS = {(pc_datum, field) for pc_datum in PC_DATUMS for field in NON_PC_FIELDS}
WICKER_KEY_SEPARATOR = '____'

# Map keys in SynchronizedScene output to wicker serialization methods
FIELD_TO_WICKER_SERIALIZER = {
    'datum_type': ws.StringSerializer,
    'intrinsics': ws.IntrinsicsSerializer,
    'distortion': ws.DistortionSerializer,
    'extrinsics': ws.PoseSerializer,
    'pose': ws.PoseSerializer,
    'rgb': ws.RGBSerializer,
    'timestamp': ws.LongSerializer,
    'point_cloud': ws.PointCloudSerializer,
    'extra_channels': ws.PointCloudSerializer,
    'bounding_box_2d': ws.BoundingBox2DSerializer,
    'bounding_box_3d': ws.BoundingBox3DSerializer,
    'semantic_segmentation_2d': ws.SemanticSegmentation2DSerializer,
    'instance_segmentation_2d': ws.InstanceSegmentation2DSerializer,
    'depth': ws.DepthSerializer,
    'velocity': ws.PointCloudSerializer,
    'covariance': ws.PointCloudSerializer,
    'key_point_2d': ws.KeyPoint2DSerializer,
    'key_line_2d': ws.KeyLine2DSerializer,
    'key_point_3d': ws.KeyPoint3DSerializer,
    'key_line_3d': ws.KeyLine3DSerializer,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def gen_wicker_key(datum_name: str, field: str) -> str:
    """Generate a key from a datum name and field i.e 'rgb', 'pose' etc

    Parameters
    ----------
    datum_name: str
        The name of the datum

    field: str
        The field of the datum

    Returns
    -------
    key: str
        The wicker key name formed from datum_name and field
    """
    return f'{datum_name}{WICKER_KEY_SEPARATOR}{field}'


def parse_wicker_key(key: str) -> Tuple[str, str]:
    """Parse a wicker dataset key into a datum and field combination

    Parameters
    ----------
    key: str
        The wicker key name formed from datum_name and field

    Returns
    -------
    datum_name: str
        The name of the datum

    field: str
        The field of the datum
    """
    return tuple(key.split(WICKER_KEY_SEPARATOR))  # type: ignore


def wicker_types_from_sample(
    sample: List[List[Dict]],
    ontology_table: Optional[Dict] = None,
    skip_camera_cuboids: bool = True,
) -> Dict[str, Any]:
    """Get the wicker keys and types from an existing dgp sample.

    Parameters
    ----------
    sample: List[List[Dict]]
        SynchronizedSceneDataset-style sample datum.

    ontology_table: Dict, default: None
        A dictionary mapping annotation key(s) to Ontology(s).

    skip_camera_cuboids: bool, default: True
        Flag to skip processing bounding_box_3d for image datums

    Returns
    -------
    wicker_types: List
        The Wicker schema types corresponding to the `wicker_keys`.
    """
    wicker_types = {}
    for datum in sample:
        datum_name = datum['datum_name']
        datum_type = datum['datum_type']
        for k, v in datum.items():
            if k == 'datum_name' or (datum_type, k) in ILLEGAL_COMBINATIONS:
                continue
            if datum_type == 'image' and k == 'bounding_box_3d' and skip_camera_cuboids:
                continue
            key = gen_wicker_key(datum_name, k)
            serializer = FIELD_TO_WICKER_SERIALIZER[k]
            wicker_types[key] = serializer().schema(key, v)

    if ontology_table is not None:
        for k, v in ontology_table.items():
            key = gen_wicker_key('ontology', k)
            wicker_types[key] = ws.OntologySerializer(k).schema(key, v)

    wicker_types['scene_index'] = IntField('scene_index')
    wicker_types['sample_index_in_scene'] = IntField('sample_index_in_scene')
    wicker_types['scene_uri'] = StringField('scene_uri')

    return wicker_types


def dgp_to_wicker_sample(
    sample: List[List[Dict]],
    wicker_keys: List[str],
    scene_index: Optional[int],
    sample_index_in_scene: Optional[int],
    ontology_table: Optional[Dict],
    scene_uri: Optional[str],
) -> Dict:
    """Convert a DGP sample to the Wicker format.

    Parameters
    ----------
    sample: List[List[Dict]]
        SynchronizedSceneDataset-style sample datum.

    wicker_keys: List[str]
        Keys to be used in Wicker.

    scene_index: int, default: None
        Index of current scene.

    sample_index_in_scene: int, default: None
        Index of the sample in current scene.

    ontology_table: Dict, default: None
        A dictionary mapping annotation key(s) to Ontology(s).

    scene_uri: str
        Relative path to this specific scene json file.

    Returns
    -------
    wicker_sample: Dict
        DGP sample in the Wicker format.
    """
    wicker_sample = {}
    for datum in sample:
        datum_name = datum['datum_name']
        for k, v in datum.items():
            key = gen_wicker_key(datum_name, k)
            if key not in wicker_keys:
                continue
            serializer = FIELD_TO_WICKER_SERIALIZER[k]
            wicker_sample[key] = serializer().serialize(v)

    if ontology_table is not None:
        for k, v in ontology_table.items():
            key = gen_wicker_key('ontology', k)
            wicker_sample[key] = ws.OntologySerializer(k).serialize(v)

    wicker_sample['scene_index'] = scene_index
    wicker_sample['sample_index_in_scene'] = sample_index_in_scene
    wicker_sample['scene_uri'] = scene_uri

    return wicker_sample


def get_scenes(scene_dataset_json: str, data_uri: Optional[str] = None) -> List[Tuple[int, str, str]]:
    """Get all the scene files from scene_dataset_json

    Parameters
    ----------
    scene_dataset_json: str
        Path ot dataset json in s3 or local.

    data_uri: str, default: None
        Optional path to location of raw data. If None, we assume the data is stored alongside scene_dataset_json.

    Returns
    -------
    scenes: List[int,str,str]
        A list of tuples(<index>, <split name>, <path to scene.json>) for each scene in scene_dataset_json.
    """
    if data_uri is None:
        data_uri = os.path.dirname(scene_dataset_json)

    dataset = open_pbobject(scene_dataset_json, SceneDataset)
    split_id_to_name = {
        dataset_pb2.TRAIN: 'train',
        dataset_pb2.VAL: 'val',
        dataset_pb2.TEST: 'test',
        dataset_pb2.TRAIN_OVERFIT: 'train_overfit',
    }

    scenes = []
    for k in dataset.scene_splits:
        files = [(split_id_to_name[k], os.path.join(data_uri, f)) for f in dataset.scene_splits[k].filenames]
        scenes.extend(files)
        logger.info(f'found {len(files)} in split {split_id_to_name[k]}')

    logger.info(f'found {len(scenes)} in {scene_dataset_json}')

    # Add the scene index
    scenes = [(k, *x) for k, x in enumerate(scenes)]

    return scenes


def chunk_scenes(scenes: List[Tuple[int, str, str]],
                 max_len: int = 200,
                 chunk_size: int = 100) -> List[Tuple[int, str, str, Tuple[int, int]]]:
    """Split each scene into chunks of max length chunk_size samples.

    Parameters
    ----------
    scenes: List[str,str]
        List of scene split/path tuples.

    max_len: int, default: 200
        Expected maximum length of each scene.

    chunk_size: int, default: 100
        Maximum size of each chunk.

    Returns
    -------
    scenes: List[Tuple[int,str,str,Tuple[int,int]]]
        A list of scenes with (<index>, <split>, <path>, (<sample index start>, <sample index end>)) tuples
    """
    new_scenes = []
    # Note by using chunks, we download the same scene multiple times.
    for c in range(max_len // chunk_size):
        chunk = (int(c * chunk_size), int((c + 1) * chunk_size))
        new_scenes.extend([(*x, chunk) for x in scenes])
    return new_scenes


def local_spark() -> pyspark.SparkContext:
    """Generate a spark context for local testing of small datasets

    Returns
    -------
    spark_context: A spark context
    """
    spark = pyspark.sql.SparkSession.builder \
    .master('local[*]') \
    .appName("dgp2wicker") \
    .config("spark.driver.memory", "56G") \
    .config("spark.executor.memory", "56G") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.maxExecutors", "4") \
    .config("spark.dynamicAllocation.minExecutors","1") \
    .config("spark.executor.cores", "1") \
    .config('spark.task.maxFailures', '4') \
    .config('spark.driver.maxResultSize','4G') \
    .config('spark.python.worker.memory','24G')\
    .getOrCreate()
    return spark.sparkContext


def ingest_dgp_to_wicker(
    scene_dataset_json: str,
    wicker_dataset_name: str,
    wicker_dataset_version: str,
    dataset_kwargs: Dict,
    spark_context: pyspark.SparkContext,
    pipeline: Optional[List[Callable]] = None,
    max_num_scenes: int = None,
    max_len: int = 1000,
    chunk_size: int = 1000,
    skip_camera_cuboids: bool = True,
    num_partitions: int = None,
    num_repartitions: int = None,
    is_pd: bool = False,
    data_uri: str = None,
    alternate_scene_uri: str = None,
) -> Dict[str, int]:
    """Ingest DGP dataset into Wicker datastore

    Parameters
    ----------
    scene_dataset_json: str
        Path to scenes dataset json.

    wicker_dataset_name: str
        Name of the dataset used in Wicker datastore.

    wicker_dataset_version: str
        Semantic version of the dataset (i.e., xxx.xxx.xxx).

    spark_context: pyspark.SparkContext, default: None
        A spark context. If None, will generate one using dgp2wicker.ingest.local_spark() and default settings

    dataset_kwargs: dict
        Arguments for dataloader.

    spark_context: A spark context, default: None
        A spark context. If None, will use a local spark context.

    pipeline: List[Callable], default: None
        A list of transformations to apply to every sample.

    max_num_scenes: int, default: None
        An optional upper bound on the number of scenes to process. Typically used for testing.

    max_len: int, default: 1000
        Maximum expected length of a scene

    chunk_size: int, default: 1000
        Chunk size to split scenes into. If less than max_len, the same scene will be downloaded
        multiple times.

    skip_camera_cuboids: bool, default: True
        Optional to flag to skip converting 'bounding_box_3d' for image_datum types.

    num_partitions: int, default: None
        Number of partitions to map scenes over. If None, defaults to number of scenes

    num_repartitions: int, default: None
        Number of partitions to shuffle all samples over. If none, defaults to num_scenes*5

    is_pd: bool, default: False
        Flag to indicate if the dataset to laod is a Parallel Domain dataset. If true, the scenes
        will be loaded with ParallelDomainScene with use_virtual_cameras set to False.

    data_uri: str
        Optional path to raw data location if raw data is not stored alongside scene_dataset_json.

    alternate_scene_uri:
        If provided, download additional scene data from an alternate location. This happens before the
        scene containing scene_json_uri is downloaded and everything in scene_json_uri's location will
        overwrite this. This also expects that the scenes are structured as <data_uri>/<scene_dir>/scene.json
        and so any addtional data for this scene should be in alternate_scene_uri/<scene_dir>.
        This is useful if for some reason a scene json and an additional annotation are in a different location
        than the rest of the scene data.

    """
    def open_scene(
        scene_json_uri: str,
        temp_dir: str,
        dataset_kwargs: Dict[str, Any],
        alternate_scene_uri: Optional[str] = None,
    ) -> Union[SynchronizedScene, ParallelDomainScene]:
        """Utility function to download a scene and open it

        Parameters
        ----------
        scene_json_uri: str
            Path to scene json.

        temp_dir: str
            Path to directory to store scene if downloaded from s3. Not used if scene_json is local

        dataset_kwargs: dict
            Arguments for data loader. i.e, datum_names, requested annotations etc. If this is a PD scene
            the dataset_kwargs should contain an `is_pd` key set to True.

        alternate_scene_uri: str, default = None
            Optional additional location to sync

        Returns
        -------
        dataset: A DGP dataset
        """
        scene_dir_uri = os.path.dirname(scene_json_uri)
        scene_json = os.path.basename(scene_json_uri)

        if scene_dir_uri.startswith('s3://'):
            # If the scene is in s3, fetch it
            local_path = temp_dir
            assert not temp_dir.startswith('s3'), f'{temp_dir}'
            if alternate_scene_uri is not None:
                alternate_scene_dir = os.path.join(alternate_scene_uri, os.path.basename(scene_dir_uri))
                logger.info(f'downloading additional scene data from {alternate_scene_dir} to {local_path}')
                sync_dir(alternate_scene_dir, local_path)

            logger.info(f'downloading scene from {scene_dir_uri} to {local_path}')
            sync_dir(scene_dir_uri, local_path)
        else:
            # Otherwise we expect the scene is on disk somewhere, so we just ignore the temp_dir
            local_path = scene_dir_uri
            logger.info(f'Using local scene from {scene_dir_uri}')

        dataset_kwargs = deepcopy(dataset_kwargs)
        dataset_kwargs['scene_json'] = os.path.join(local_path, scene_json)

        is_pd = dataset_kwargs.pop('is_pd')

        if is_pd:
            dataset_kwargs['use_virtual_camera_datums'] = False
            dataset = ParallelDomainScene(**dataset_kwargs)
        else:
            dataset = SynchronizedScene(**dataset_kwargs)

        return dataset

    def process_scene(
        partition: List[Tuple[int, str, str, Tuple[int, int]]], dataset_kwargs: Dict, pipeline: List[Callable],
        wicker_types: List[str]
    ) -> Generator[Tuple[str, Any], None, None]:
        """Main task to parrallelize. This takes a list of scene chunks and sequentially
        downloads the scene to a temporary directory, opens the scene, applies any transformations,
        and yields wicker serialized samples.

        Parameters
        ----------
        partition: tuple
            A list of scenes to process with this spark partition.
            Each entry should be a tuple with <index in dataset, split, scene_uri, (chunk start, chunk end)>.

        dataset_kwargs: Dict
            Arguments for data loader. See open_scene for details.

        pipeline: List[Callable]
            List of transformations to apply to samples.

        wicker_types: List
            A list of keys in the wicker schema.

        Returns
        -------
        wicker_sample: (split, sample)
            Yields a wicker converted sample
        """
        for scene_info in partition:
            yield_count = 0
            global_scene_index, split, scene_json_uri, chunk = scene_info
            chunk_start, chunk_end = chunk
            scene_dir_uri = os.path.dirname(scene_json_uri)
            scene_json = os.path.basename(scene_json_uri)

            st = time.time()
            with tempfile.TemporaryDirectory() as temp_dir:
                # TODO (chris.ochoa): check the number of samples before syncing the entire scene
                try:
                    # Download and open the scene
                    dataset = open_scene(
                        scene_json_uri, temp_dir, dataset_kwargs, alternate_scene_uri=alternate_scene_uri
                    )
                except Exception as e:
                    logger.error(f'Failed to open {scene_json_uri}, skipping...')
                    logger.error(e)
                    traceback.print_exc()
                    continue

                ontology_table = dataset.dataset_metadata.ontology_table

                for i in range(chunk_start, chunk_end):
                    if i >= len(dataset):
                        break

                    try:
                        _, sample_index_in_scene, _ = dataset.dataset_item_index[i]
                        sample = dataset[i][0]  # explicitly using 0 context here

                        # Apply any transformations
                        for transform in pipeline:
                            sample = transform(sample)

                        wicker_sample = dgp_to_wicker_sample(
                            sample,
                            wicker_keys=wicker_types.keys(),
                            scene_index=int(global_scene_index),
                            sample_index_in_scene=int(sample_index_in_scene),
                            ontology_table=ontology_table,
                            scene_uri=os.path.join(os.path.basename(scene_dir_uri), scene_json),
                        )
                        #import pdb; pdb.set_trace()

                        assert wicker_sample is not None
                        for k, v in wicker_sample.items():
                            assert v is not None, f'{k} has invalid wicker serialized item'

                        yield_count += 1
                        yield (split, deepcopy(wicker_sample))

                    except Exception as e:
                        logger.error('failed to get sample, skipping...')
                        logger.error(e)
                        traceback.print_exc()
                        continue

            dt = time.time() - st
            logger.info(
                f'Finished {global_scene_index} {split}/{scene_dir_uri}, chunk:{chunk_start}->{chunk_end}.\
                  Yielded {yield_count}, took {dt:.2f} seconds'
            )

    dataset_kwargs['is_pd'] = is_pd

    if pipeline is None:
        pipeline = []

    # Parse the dataset json and get the scene list. This is a list of tuple split, fully qualified scene uri
    scenes = get_scenes(scene_dataset_json, data_uri=data_uri)
    if max_num_scenes is not None:
        scenes = scenes[:int(max_num_scenes)]

    if num_partitions is None:
        num_partitions = len(scenes)

    if num_repartitions is None:
        num_repartitions = 5 * len(scenes)

    # Use first sample to obtain schema. NOTE: We actually don't need the sample here for this
    # TODO (chris.ochoa): generate the keys we expect for a specific combination of datums/annotations/ontologies
    _, _, scene_json_uri = scenes[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = open_scene(scene_json_uri, temp_dir, dataset_kwargs, alternate_scene_uri=alternate_scene_uri)
        logger.info(f'Got dataset with {len(dataset)} samples')
        sample = dataset[0][0]
        # Apply any transformations
        for transform in pipeline:
            sample = transform(sample)
        # TODO (chris.ochoa): make sure this has all the keys we care about in this ontology!
        # it is possible (though never encountered) that for some reason the first scene may not have annotations
        # and therefore no ontologies and conversion will fail.
        ontology_table = dataset.dataset_metadata.ontology_table

    # Build the schema from the sample
    wicker_types = wicker_types_from_sample(
        sample=sample,
        ontology_table=ontology_table,
        skip_camera_cuboids=skip_camera_cuboids,
    )

    wicker_dataset_schema = wicker.schema.DatasetSchema(
        primary_keys=['scene_index', 'sample_index_in_scene'], fields=list(wicker_types.values())
    )

    # Chunk the scenes
    scenes = chunk_scenes(scenes, max_len=max_len, chunk_size=chunk_size)

    # Shuffle the scenes
    scene_shuffle_idx = np.random.permutation(len(scenes)).tolist()
    scenes = [scenes[i] for i in scene_shuffle_idx]
    if len(scenes) < 2:
        wsp.SPARK_PARTITION_SIZE = 3

    # Setup spark
    if spark_context is None:
        spark_context = local_spark()

    process = partial(process_scene, dataset_kwargs=dataset_kwargs, pipeline=pipeline, wicker_types=wicker_types)
    rdd = spark_context.parallelize(scenes,
                                    numSlices=num_partitions).mapPartitions(process).repartition(num_repartitions)

    return wsp.persist_wicker_dataset(
        wicker_dataset_name,
        wicker_dataset_version,
        wicker_dataset_schema,
        rdd,
    )
