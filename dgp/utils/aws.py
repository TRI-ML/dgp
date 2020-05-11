# Copyright 2019 Toyota Research Institute.  All rights reserved.
"""Various utility functions for AWS"""
import functools
import hashlib
import logging
import multiprocessing
import os
import subprocess
import tempfile
from multiprocessing import Pool
from urllib.parse import urlparse

import boto3
import numpy as np
from google.protobuf.json_format import Parse

from dgp import DGP_CACHE_DIR
from dgp.proto.dataset_pb2 import SceneDataset
from dgp.utils.protobuf import save_pbobject_as_json

# This is useful when using multiprocessing with S3.
S3_CLIENT, ATHENA_CLIENT = None, None


def _prefetch_file(filename):
    """Helper of `prefetch_lustre_files`.

    Read a single byte force lustre to fetch the full file over the network
    and cache the file in the object-store (OST)
    """
    with open(filename, 'rb') as f:
        f.read(1)
    return filename


def prefetch_lustre_files(files):
    """Force LustreFSX to prefetch files by touching them.

    Parameters
    ----------
    files: list
        List of files to prefetch.
    """

    fetched_files = []

    def _add_file(filename):
        fetched_files.append(filename)

    prefetch_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for filename in files:
        prefetch_pool.apply_async(_prefetch_file, args=(filename, ), callback=_add_file)
    prefetch_pool.close()
    prefetch_pool.join()

    if len(files) - len(fetched_files) > 0:
        unfetched_files = list(set(files) - set(fetched_files))
        for file in unfetched_files:
            logging.critical("Prefetch failed: {:s}".format(file))
    else:
        logging.info("All {:d} files are pre-fetched.".format(len(files)))


def load_remote_scene_dataset_json(scene_dataset_json_url, save_dir=None):
    """Fetch scene dataset JSON from remote given its remote uri.

    Parameters
    ----------
    scene_dataset_json_uri: str
        Remote scene dataset JSON URI.

    save_dir: str, default: None
        Directory to save dataset JSON.

    Returns
    -------
    dataset: dgp.proto.dataset_pb2.SceneDataset
    SceneDataset proto object
    """
    logging.info('Fetching scene dataset {}'.format(scene_dataset_json_url))
    bucket_name, s3_base_path = convert_uri_to_bucket_path(scene_dataset_json_url)
    dataset_blob = get_string_from_s3_file(s3_bucket(bucket_name), s3_base_path)
    dataset = Parse(dataset_blob, SceneDataset())
    if save_dir is not None:
        dataset_dir = os.path.join(
            save_dir, '{name}_{version}'.format(name=dataset.metadata.name, version=dataset.metadata.version)
        )
        save_path = os.path.join(dataset_dir, os.path.basename(s3_base_path))
        if not os.path.exists(save_path):
            os.makedirs(dataset_dir, exist_ok=True)
            save_pbobject_as_json(dataset, save_path)
    return dataset


def fetch_remote_scene(remote_scene_json):
    """Fetch remote scene directory from S3 given the remote scene json url.

    Parameters
    ----------
    remote_scene_json: str
        Remote scene json corresponding to the remote scene to be fetched.

    Returns
    -------
    scene_json: str
      Location of the local scene json file.
    """
    source_dir, basename = os.path.split(remote_scene_json)
    os.path.split(source_dir)
    target_dir = os.path.join(DGP_CACHE_DIR, 'scenes', os.path.basename(source_dir))
    sync_dir(source_dir, target_dir, verbose=True)
    return os.path.join(target_dir, basename)


def s3_copy(source_path, target_path, verbose=True):
    """Copy single file from local to s3, s3 to local, or s3 to s3.

    Parameters
    ----------
    source_path: str
        Path of file to copy

    target_path: str
        Path to copy file to

    verbose: bool, default: True
        If True print some helpful messages
    """
    command_str = "aws s3 cp {} {}".format(source_path, target_path)
    if verbose:
        logging.info("Copying file with '{}'".format(command_str))
    subprocess.check_output(command_str, shell=True)
    if verbose:
        logging.info("Done copying file")


def sync_dir(source, target, verbose=True):
    """Sync a directory from source to target (either local to s3, s3 to s3, s3 to local)

    Parameters
    ----------
    source: str
        Directory from which we want to sync files

    target: str
        Directory to which all files will be synced

    verbose: bool, default: True
        If True, log some helpful messages
    """
    assert source.startswith('s3://') or target.startswith('s3://')
    command_str = "aws s3 sync --quiet {} {}".format(source, target)
    if verbose:
        logging.info("Syncing with '{}'".format(command_str))
    subprocess.check_output(command_str, shell=True)
    if verbose:
        logging.info("Done syncing")


def s3_bucket(bucket_name):
    """Instantiate S3 bucket object from its bucket name

    Parameters
    ----------
    bucket : str
        Bucket name to instantiate.

    Returns
    -------
    S3 bucket
    """
    return boto3.session.Session().resource('s3').Bucket(bucket_name)


def init_aws_client(service):
    """Initiate S3 or Athena AWS client.
    Parameters
    ----------
    service: str
        Either `S3` or `Athena` (case insensitive).
    """

    if service.lower() == "s3":
        global S3_CLIENT
        if S3_CLIENT is None:
            S3_CLIENT = boto3.client("s3")
        return S3_CLIENT
    elif service.lower() == "athena":
        global ATHENA_CLIENT
        if ATHENA_CLIENT is None:
            ATHENA_CLIENT = boto3.client("athena")
        return ATHENA_CLIENT
    else:
        raise ValueError("Only supports S3 and Athena.")


def exists_s3_object(bucket, url):
    """Uses a valid S3 bucket to check the existence of a remote file as AWS URL.

    Parameters
    ----------
    bucket : S3 bucket
        Must have been created similar to
        ``boto3.session.Session().resource('s3').Bucket(BUCKET_NAME)``

    url : str
        The remote URL of the object to be fetched

    Returns
    -------
    bool
    """

    # Retrieve a collection of S3 objects with that prefix and check if non-empty
    return len(list(bucket.objects.filter(Prefix=url))) != 0


def convert_uri_to_bucket_path(uri):
    """Parse a URI into a bucket and path.
    Parameters
    ----------
    uri : str
      S3 URI of an object.

    Returns
    -------
    str :
      The bucket of the S3 object
    str :
      The path within the bucket of the S3 object
    """
    s3_uri = urlparse(uri)
    s3_path = s3_uri.path.lstrip('/')
    s3_path = s3_path.rstrip('/')

    return (s3_uri.netloc, s3_path)


def parallel_download_s3_objects(s3_files, destination_filepaths, bucket, process_pool_size=None):
    """Download a list of s3 objects using a processpool.

    Parameters
    ----------
    s3_files: list
      List of all files from s3 to download

    destination_filepaths: list
      List of where to download all files locally

    bucket: str
      S3 bucket to pull from

    process_pool_size: int, default: None
        Number of threads to use to fetch these files. If not specified, will default to
        number of cores on the machine.

    """
    if process_pool_size == None:
        process_pool_size = multiprocessing.cpu_count()

    s3_and_destination = zip(s3_files, destination_filepaths)
    with Pool(process_pool_size, functools.partial(init_aws_client, service="s3")) as proc:
        proc.starmap(functools.partial(download_s3_object_to_path, bucket), s3_and_destination)


def parallel_upload_s3_objects(source_filepaths, s3_destinations, bucket, process_pool_size=None):
    """Upload a list of s3 objects using a processpool.

    Parameters
    ----------
    source_filepaths: list
      List of all local files to upload

    s3_destinations: list
      Paths relative to bucket in S3 where files are uploaded

    bucket: str
      S3 bucket to upload to

    process_pool_size: int, default: None
        Number of threads to use to fetch these files. If not specified, will default to
        number of cores on the machine.
    """
    if process_pool_size == None:
        process_pool_size = multiprocessing.cpu_count()

    s3_and_destination = zip(source_filepaths, s3_destinations)
    with Pool(process_pool_size, functools.partial(init_aws_client, service="s3")) as proc:
        proc.starmap(functools.partial(upload_file_to_s3, bucket), s3_and_destination)


def download_s3_object_to_path(bucket, s3_path, local_path):
    """Download an object from s3 to a path on the local filesystem.

    Parameters
    ----------
    bucket: str
        S3 bucket from which to fetch.

    s3_path: str
        The path on the s3 bucket to the file to fetch.

    local_path: str
        Path on the local filesystem to place the object.
    """
    s3 = boto3.client('s3')
    logging.debug("Downloading file {} => {}".format(s3_path, local_path))

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
    except OSError:
        # Silently handle this error. Likely the directory already exists in
        # another thread.
        pass
    s3.download_file(bucket, s3_path, local_path)


def upload_file_to_s3(bucket, local_file_path, bucket_rel_path):
    """
    Parameters
    ----------
    local_file_path: str
        Local path to file we want to upload

    bucket_rel_path: str
        Path where file is uploaded, relative to S3 bucket root

    Returns
    -------
    s3_url: str
        s3_url to which file was uploaded.

    """
    assert os.path.exists(local_file_path)
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).upload_file(local_file_path, bucket_rel_path)
    return os.path.join("s3://", bucket, bucket_rel_path)


def get_s3_object(bucket, url):
    """Uses a valid S3 bucket to retrieve a file from a remote AWS URL.
    Raises ValueError if non-existent.

    Parameters
    ----------
    bucket : S3 bucket
        Must have been created similar to
        ``boto3.session.Session().resource('s3').Bucket(BUCKET_NAME)``

    url : str
        The remote URL of the object to be fetched

    Returns
    -------
    S3 object
    """

    # Retrieve a collection of S3 objects with that prefix and check first item in it
    result = list(bucket.objects.filter(Prefix=url))
    if len(result) == 0:
        raise ValueError(url, 'cannot be found!')
    return result[0]


def get_string_from_s3_file(bucket, url):
    """Uses a valid S3 bucket to retrieve an UTF-8 decoded string from a remote AWS URL.
    Raises ValueError if non-existent.

    Parameters
    ----------
    bucket : S3 bucket
        Must have been created similar to
        ``boto3.session.Session().resource('s3').Bucket(BUCKET_NAME)``

    url : str
        The remote URL of the object to be fetched

    Returns
    -------
    A string representation of the remote file
    """
    s3_obj = get_s3_object(bucket, url)
    return s3_obj.get()['Body'].read().decode('utf-8')


def get_bytes_from_s3_file(bucket, url):
    """Uses a valid S3 bucket to retrieve an uint8 Numpy array from a remote AWS URL.
    Raises ValueError if non-existent.

    Parameters
    ----------
    bucket : S3 bucket
        Must have been created similar to
        ``boto3.session.Session().resource('s3').Bucket(BUCKET_NAME)``

    url : str
        The remote URL of the object to be fetched

    Returns
    -------
    A numpy array (dtype=np.uint8) representation of the remote file
    """

    raw_data = get_s3_object(bucket, url).get()['Body'].read()
    return np.frombuffer(raw_data, dtype=np.uint8)
