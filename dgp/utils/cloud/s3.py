# Copyright 2020-21 Toyota Research Institute.  All rights reserved.
"""Various utility functions for AWS S3"""
import functools
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import traceback
from multiprocessing import Pool, cpu_count
from urllib.parse import urlparse

import boto3
import botocore
import tenacity

# This is useful when using multiprocessing with S3.
S3_CLIENT_SSL = None
S3_CLIENT_NO_SSL = None


def init_s3_client(use_ssl=False):
    """Initiate S3 AWS client.

    Parameters
    ----------
    use_ssl: bool, optional
        Use secure sockets layer. Provieds better security to s3, but
        can fail intermittently in a multithreaded environment. Default: False.

    Returns
    -------
    service: boto3.client
        S3 resource service client.
    """
    global S3_CLIENT_SSL
    global S3_CLIENT_NO_SSL
    if use_ssl:
        if S3_CLIENT_SSL is None:
            S3_CLIENT_SSL = boto3.client("s3")
        return S3_CLIENT_SSL
    if not S3_CLIENT_NO_SSL:
        S3_CLIENT_NO_SSL = boto3.client("s3", use_ssl=False)
    return S3_CLIENT_NO_SSL


def s3_recursive_list(s3_prefix):
    """List all files contained in an s3 location recursively and also return their md5_sums
    NOTE: this is different from 'aws s3 ls' in that it will not return directories, but instead
    the full paths to the files contained in any directories (which is what s3 is actually tracking)

    Parameters
    ----------
    s3_prefix: str
        s3 prefix which we want the returned files to have

    Returns
    -------
    all_files: list[str]
        List of files (with full path including 's3://...')

    md5_sums: list[str]
        md5 sum for each of the files as returned by boto3 'ETag' field
    """
    assert s3_prefix.startswith('s3://')

    bucket_name, prefix = convert_uri_to_bucket_path(s3_prefix, strip_trailing_slash=False)

    s3_client = init_s3_client(use_ssl=False)

    # `list_objects_v2` is an updated version of `list_objects` with very similar functionality
    s3_metadata = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if 'Contents' not in s3_metadata:
        return [], []

    # Need to create a paginator to scroll through items at s3 location (only 1000 are returned at a time by default)
    paginator = s3_client.get_paginator('list_objects_v2').paginate(Bucket=bucket_name, Prefix=prefix)
    s3_metadata = [single_object for page in paginator for single_object in page['Contents']]

    all_files = [os.path.join('s3://', bucket_name, _file_metadata['Key']) for _file_metadata in s3_metadata]
    md5_sums = [_file_metadata['ETag'].strip('"') for _file_metadata in s3_metadata]

    return all_files, md5_sums


def return_last_value(retry_state):
    """Return the result of the last call attempt.

    Parameters
    ----------
    retry_state: tenacity.RetryCallState
        Retry-state metadata for a flaky call.
    """
    return retry_state.outcome.result()


def is_false(value):
    return value is False


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    retry=tenacity.retry_if_result(is_false),
    retry_error_callback=return_last_value
)
def s3_copy(source_path, target_path, verbose=True):
    """Copy single file from local to s3, s3 to local, or s3 to s3.

    Parameters
    ----------
    source_path: str
        Path of file to copy

    target_path: str
        Path to copy file to

    verbose: bool, optional
        If True print some helpful messages. Default: True.

    Returns
    -------
    bool: True if successful
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    success = False
    command_str = "aws s3 cp --acl bucket-owner-full-control {} {}".format(source_path, target_path)
    logging.debug("Copying file with '{}'".format(command_str))
    try:
        subprocess.check_output(command_str, shell=True)
        success = True
    except subprocess.CalledProcessError as e:
        success = False
        logging.error("{} failed with error code {}".format(command_str, e.returncode))
        logging.error(e.output)
    if verbose:
        logging.info("Done copying file")

    return success


def parallel_s3_copy(source_paths, target_paths, threadpool_size=None):
    """Copy files from local to s3, s3 to local, or s3 to s3 using a threadpool.
    Retry the operation if any files fail to copy. Throw an AssertionError if it fails the 2nd time.

    Parameters
    ----------
    source_paths: List of str
        Full paths of files to copy.

    target_paths: List of str
        Full paths to copy files to.

    threadpool_size: int
        Number of threads to use to fetch these files. If not specified, will default to
        number of cores on the machine.

    """
    if threadpool_size is None:
        threadpool_size = cpu_count()
    s3_and_destination = zip(source_paths, target_paths)
    with Pool(threadpool_size) as thread_pool:
        s3_copy_function = functools.partial(s3_copy)
        success_list = thread_pool.starmap(s3_copy_function, s3_and_destination)

    num_success = sum(success_list)
    logging.info(f'{num_success} / {len(success_list)} files copied successfully.')
    # Copy the failed files sequentially.
    for success, source, target in zip(success_list, source_paths, target_paths):
        if not success:
            assert s3_copy(source, target), f'Failed to copy {source} to {target} on 2nd try'


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    retry=tenacity.retry_if_result(is_false),
    retry_error_callback=return_last_value
)
def sync_dir(source, target, file_ext=None, verbose=True):
    """Sync a directory from source to target (either local to s3, s3 to s3, s3 to local)

    Parameters
    ----------
    source: str
        Directory from which we want to sync files

    target: str
        Directory to which all files will be synced

    file_ext: str, optional
        Only sync files ending with this extension. Eg: 'csv' or 'json'. If None, sync all files. Default: None.

    verbose: bool, optional
        If True, log some helpful messages. Default: True.

    Returns
    -------
    bool: True if operation was successful
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    assert source.startswith('s3://') or target.startswith('s3://')
    additional_flags = ''
    if file_ext is not None:
        additional_flags = f'--exclude=* --include=*{file_ext}'

    command_str = "aws s3 sync --quiet --acl bucket-owner-full-control {} {} {}".format(
        additional_flags, source, target
    )
    success = False
    logging.debug("Syncing with '{}'".format(command_str))
    try:
        subprocess.check_output(command_str, shell=True)
        success = True
    except subprocess.CalledProcessError as e:
        success = False
        logging.error("{} failed with error code {}".format(command_str, e.returncode))
        logging.error(e.output)
    if verbose:
        logging.info("Done syncing")

    return success


def parallel_s3_sync(source_paths, target_paths, threadpool_size=None):
    """Copy directories from local to s3, s3 to local, or s3 to s3 using a threadpool.
    Retry the operation if any files fail to sync. Throw an AssertionError if it fails the 2nd time.

    Parameters
    ----------
    source_paths: List of str
        Directories from which we want to sync files.

    target_paths: List of str
        Directories to which all files will be synced.

    threadpool_size: int
        Number of threads to use to fetch these files. If not specified, will default to
        number of cores on the machine.

    """
    if threadpool_size is None:
        threadpool_size = cpu_count()
    s3_and_destination = zip(source_paths, target_paths)
    with Pool(threadpool_size) as thread_pool:
        s3_sync_function = functools.partial(sync_dir)
        success_list = thread_pool.starmap(s3_sync_function, s3_and_destination)

    num_success = sum(success_list)
    logging.info(f'{num_success} / {len(success_list)} files copied successfully.')
    # Copy the failed files sequentially.
    for success, source, target in zip(success_list, source_paths, target_paths):
        if not success:
            assert sync_dir(source, target), f'Failed to copy {source} to {target} on 2nd try'


def sync_dir_safe(source, target, verbose=True):
    """Sync a directory from local to s3 by first ensuring that NONE of the files in `target`
    have been edited in `source` (new files can exist in `source` that are not in `target`)
    (NOTE: only checks files from `target` that exist in `source`)

    Parameters
    ----------
    source: str
        Directory from which we want to sync files

    target: str
        Directory to which all files will be synced

    verbose: bool, optional
        If True, log some helpful messages. Default: True.

    Returns
    -------
    files_fail_to_sync: list
        List of files fail to sync to S3 due to md5sum mismatch.
    """

    assert not source.startswith('s3://') and target.startswith('s3://')

    # NOTE: Add trailing '/' to `target`.
    target_files, target_file_md5_sums = s3_recursive_list(os.path.join(target, ''))

    rel_target_files = [os.path.relpath(_f, target) for _f in target_files]

    files_fail_to_sync = []
    for rel_target_file, target_file_md5_sum in zip(rel_target_files, target_file_md5_sums):

        local_file = os.path.join(source, rel_target_file)

        # Only assert md5_sum match if the file actually exists locally
        if os.path.exists(local_file):
            local_file_md5_sum = hashlib.md5(open(local_file, 'rb').read()).hexdigest()

            if local_file_md5_sum != target_file_md5_sum:
                files_fail_to_sync.append(rel_target_file)
                logging.error('Cannot sync to s3, you have made changes to "{}" locally'.format(rel_target_file))

    sync_dir(source, target, verbose=verbose)
    if verbose:
        logging.info(
            '{} | {} files synced to S3'.format(len(rel_target_files) - len(files_fail_to_sync), len(rel_target_files))
        )
    return files_fail_to_sync


def s3_bucket(bucket_name):
    """Instantiate S3 bucket object from its bucket name

    Parameters
    ----------
    bucket_name : str
        Bucket name to instantiate.

    Returns
    -------
    S3 bucket
    """
    return boto3.session.Session().resource('s3').Bucket(bucket_name)


def list_s3_objects(bucket_name, url):
    """List all files within a valid S3 bucket

    Parameters
    ----------
    bucket_name : str
        AWS S3 root bucket name

    url : str
        The remote URL of the object to be fetched
    """
    return list(s3_bucket(bucket_name).objects.filter(Prefix=url))


def exists_s3_object(bucket_name, url):
    """Uses a valid S3 bucket to check the existence of a remote file as AWS URL.

    Parameters
    ----------
    bucket_name : str
        AWS S3 root bucket name

    url : str
        The remote URL of the object to be fetched

    Returns
    -------
    bool:
        Wether or not the object exists in S3.
    """

    # Retrieve a collection of S3 objects with that prefix and check if non-empty
    return len(list(s3_bucket(bucket_name).objects.filter(Prefix=url))) != 0


def convert_uri_to_bucket_path(uri, strip_trailing_slash=True):
    """Parse a URI into a bucket and path.

    Parameters
    ----------
    uri: str
        A full s3 path (e.g. 's3://<s3-bucket>/<s3-prefix>')

    strip_trailing_slash: bool, optional
        If True, we strip any trailing slash in `s3_path` before returning it
        (i.e. s3_path='<s3-prefix>' is returned for uri='s3://<s3-bucket>/<s3-prefix>/').
        Otherwise, we do not strip it
        (i.e. s3_path='<s3-prefix>' is returned for uri='s3://<s3-bucket>/<s3-prefix>/').

        If there is no trailing slash in `uri` then there will be no trailing slash in `s3_path` regardless
        of the value of this parameter.
        Default: True.

    Returns
    -------
    str:
        The bucket of the S3 object (e.g. '<s3-bucket>')

    str:
        The path within the bucket of the S3 object (e.g. '<s3-prefix>')
    """
    s3_uri = urlparse(uri)
    s3_path = s3_uri.path.lstrip('/')
    if strip_trailing_slash:
        s3_path = s3_path.rstrip('/')
    return (s3_uri.netloc, s3_path)


def parallel_download_s3_objects(s3_files, destination_filepaths, bucket_name, process_pool_size=None):
    """Download a list of s3 objects using a processpool.

    Parameters
    ----------
    s3_files: list
      List of all files from s3 to download

    destination_filepaths: list
      List of where to download all files locally

    bucket_name: str
      S3 bucket to pull from

    process_pool_size: int, optional
        Number of threads to use to fetch these files. If not specified, will default to
        number of cores on the machine. Default: None.
    """
    if process_pool_size is None:
        process_pool_size = cpu_count()

    s3_and_destination = zip(s3_files, destination_filepaths)
    with Pool(process_pool_size, init_s3_client) as proc:
        results = proc.starmap(functools.partial(download_s3_object_to_path, bucket_name), s3_and_destination)

    failed_files = [
        os.path.join('s3://', bucket_name, s3_file) for result, s3_file in zip(results, s3_files) if not result
    ]
    assert len(failed_files) == 0, 'Failed downloading {}/{} files:\n{}'.format(
        len(failed_files), len(results), '\n'.join(failed_files)
    )


def parallel_upload_s3_objects(source_filepaths, s3_destinations, bucket_name, process_pool_size=None):
    """Upload a list of s3 objects using a processpool.

    Parameters
    ----------
    source_filepaths: list
      List of all local files to upload.

    s3_destinations: list
      Paths relative to bucket in S3 where files are uploaded.

    bucket_name: str
      S3 bucket to upload to.

    process_pool_size: int, optional
        Number of threads to use to fetch these files. If not specified, will default to
        number of cores on the machine. Default: None.
    """
    if process_pool_size is None:
        process_pool_size = cpu_count()

    s3_and_destination = zip(source_filepaths, s3_destinations)
    with Pool(process_pool_size, init_s3_client) as proc:
        proc.starmap(functools.partial(upload_file_to_s3, bucket_name), s3_and_destination)


def download_s3_object_to_path(bucket_name, s3_path, local_path):
    """Download an object from s3 to a path on the local filesystem.

    Parameters
    ----------
    bucket_name: str
        S3 bucket from which to fetch.

    s3_path: str
        The path on the s3 bucket to the file to fetch.

    local_path: str
        Path on the local filesystem to place the object.

    Returns
    -------
        bool
            True if successful, False if not.
    """
    s3_client = init_s3_client()
    logging.debug("Downloading file {} => {}".format(s3_path, local_path))

    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
    except OSError:
        # Silently handle this error. Likely the directory already exists in
        # another thread.
        pass
    try:
        s3_client.download_file(bucket_name, s3_path, local_path)
        return True
    except botocore.exceptions.ClientError:
        logging.error(
            "Traceback downloading {}:\n{}".format(os.path.join('s3://', bucket_name, s3_path), traceback.format_exc())
        )
        return False


def delete_s3_object(bucket_name, object_key, verbose=True):
    """Delete an object in s3.

    Parameters
    ----------
    bucket_name: str
        S3 bucket of the object to delete.

    object_key: str
        Key name of the object to delete.

    verbose: bool, optional
        If True print messages. Default: True.

    """
    s3_client = init_s3_client()
    if verbose:
        logging.warning("Deleting S3 Object {}".format(os.path.join("s3://", bucket_name, object_key)))
    s3_client.delete_object(Bucket=bucket_name, Key=object_key)


def upload_file_to_s3(bucket_name, local_file_path, bucket_rel_path):
    """
    Parameters
    ----------
    bucket_name : str
        AWS S3 root bucket name

    local_file_path: str
        Local path to file we want to upload

    bucket_rel_path: str
        Path where file is uploaded, relative to S3 bucket root

    Returns
    -------
    s3_url: str
        s3_url to which file was uploaded
        e.g. "s3://<s3-bucket>/<task-name>/<wandb-run>/<model-weights-file>"

    """
    assert os.path.exists(local_file_path)
    s3_bucket(bucket_name).upload_file(local_file_path, bucket_rel_path)
    return os.path.join("s3://", bucket_name, bucket_rel_path)


def get_s3_object(bucket_name, url):
    """Uses a valid S3 bucket to retrieve a file from a remote AWS URL.
    Raises ValueError if non-existent.

    Parameters
    ----------
    bucket_name : str
        AWS S3 root bucket name

    url : str
        The remote URL of the object to be fetched

    Returns
    -------
    S3 object

    Raises
    ------
    ValueError
        Raised if `url` cannot be found in S3.
    """

    # Retrieve a collection of S3 objects with that prefix and check first item in it
    result = list(s3_bucket(bucket_name).objects.filter(Prefix=url))
    if not result:
        raise ValueError(url, 'cannot be found!')
    return result[0]


def get_string_from_s3_file(bucket_name, url):
    """Uses a valid S3 bucket to retrieve an UTF-8 decoded string from a remote AWS URL.
    Raises ValueError if non-existent.

    Parameters
    ----------
    bucket_name : str
        AWS S3 root bucket name

    url : str
        The remote URL of the object to be fetched

    Returns
    -------
    A string representation of the remote file
    """
    s3_obj = get_s3_object(bucket_name, url)
    return s3_obj.get()['Body'].read().decode('utf-8')


def open_remote_json(s3_path):
    """Loads a remote JSON file

    Parameters
    ----------
    s3_path: str
        Full s3 path to JSON file

    Returns
    -------
    dict:
        Loaded JSON
    """
    assert s3_path.startswith('s3://') and s3_path.endswith('.json')
    bucket, rel_path = convert_uri_to_bucket_path(s3_path)
    return json.loads(get_string_from_s3_file(bucket, rel_path))


class RemoteArtifactFile:
    def __init__(self, artifact):
        """Context manager for a remote file on S3.

        Parameters
        ----------
        artifact : mgp.remote_pb2.RemoteArtifact
            Remote artifact object that contains the url and the sha1sum of the file

        url : str
            The remote URL of the object to be fetched

        Returns
        -------
        bool
        """
        self._artifact = artifact
        self._dirname = None
        url = artifact.url.value
        self._bucket = url.split('//')[-1].split('/')[0]
        self._url = url
        self._relpath = url[url.rfind(self._bucket) + len(self._bucket) + 1:]

        self._s3_client = init_s3_client()
        _s3_bucket = boto3.session.Session().resource('s3').Bucket(self._bucket)
        assert exists_s3_object(_s3_bucket, self._relpath), \
            'Remote artifact does not exist {}'.format(self._url)

    def __enter__(self):
        self._dirname = tempfile.TemporaryDirectory()
        local_path = os.path.join(self._dirname.name, os.path.basename(self._relpath))
        logging.info('Downloading {} to {}'.format(self._url, local_path))
        self._s3_client.download_file(self._bucket, self._relpath, local_path)
        sha1 = hashlib.sha1(open(local_path, 'rb').read()).hexdigest()
        assert sha1 == self._artifact.sha1, \
            'sha1sum inconsistent {} != {}'.format(sha1, self._artifact.sha1)
        return local_path

    def __exit__(self, *args):
        self._dirname.__exit__(*args)


def list_prefixes_in_s3_dir(s3_path):
    """
    List prefixes in S3 path.
    *CAVEAT*: This function was only tested for S3 path that contains purely one-level prefix, i.e. no regular objects.
    Parameters
    ----------
    s3_path: str
        An S3 path.
    Returns
    -------
    prefixes: List of str
        List of prefixes under `s3_path`
    """

    s3_client = init_s3_client()

    assert s3_path.startswith('s3://')
    bucket, path = convert_uri_to_bucket_path(s3_path)

    prefixes = []
    kwargs = {'Bucket': bucket, "Prefix": os.path.join(path, ''), "Delimiter": '/'}
    while True:
        response = s3_client.list_objects_v2(**kwargs)
        for obj in response['CommonPrefixes']:
            prefixes.append(os.path.basename(obj['Prefix'].rstrip('/')))
        try:
            kwargs['ContinuationToken'] = response['NextContinuationToken']
        except KeyError:
            break

    return prefixes


def list_prefixes_in_s3(s3_prefix):
    """
    List prefixes in S3 path.
    *CAVEAT*: This function was only tested for S3 prefix for files. E.g., if the bucket looks like the following,
        - s3://aa/bb/cc_v01.json
        - s3://aa/bb/cc_v02.json
        - s3://aa/bb/cc_v03.json
        then list_prefixes_in_s3("s3://aa/bb/cc_v") returns ['cc_v01.json', 'cc_v02.json', 'cc_v03.json']
    Parameters
    ----------
    s3_prefix: str
        An S3 prefix.
    Returns
    -------
    prefixes: List of str
        List of basename prefixes that starts with `s3_prefix`.
    """
    bucket_name, prefix = convert_uri_to_bucket_path(s3_prefix)
    bucket = s3_bucket(bucket_name)
    prefixes = [os.path.basename(obj.key) for obj in bucket.objects.filter(Prefix=prefix)]

    return prefixes
