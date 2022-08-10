# Copyright 2021 Toyota Research Institute.  All rights reserved.
import hashlib
import logging
import os
import pickle
import shutil
from functools import wraps

import numpy as np

from dgp import DGP_CACHE_DIR


# The DGP cache dir can be cleaned up on every run to avoid hashing the
# function / class instance method that may be tricky to implement, especially
# in the case of varying arg/kwargs in the function / class-method to be
# decorated.
def clear_cache(directory=DGP_CACHE_DIR):
    """Clear DGP cache to avoid growing disk-usage.

    Parameters
    ----------
    directory: str, optional
        A pathname to the directory used by DGP for caching. Default: DGP_CACHE_DIR.
    """
    if os.path.isdir(directory):
        logging.info('Clearing dgp disk-cache.')
        try:
            shutil.rmtree(directory)
        except OSError as e:
            logging.warning('Failed to clear cache {}'.format(e))


def diskcache(protocol='npz', cache_dir=None):
    """Disk-caching method/function decorator that caches results into
    dgp cache for arbitrary pickle-able / numpy objects.

    Parameters
    ----------
    protocol: str, optional
        Serialization protocol. Choices: {npz, pkl}. Default: "npz" (numpy).

    cache_dir: str, optional
        Directory to cache instead of the default DGP cache. Default: None.
    """
    assert protocol in ('npz', 'pkl'), 'Unknown protocol {}'.format(protocol)
    logging.info('Using dgp disk-cache.')
    if cache_dir is None:
        cache_dir = DGP_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)  # pylint: disable=unexpected-keyword-arg

    def wrapped_diskcache(func):
        def serialize(_result, _filename, _protocol):
            """Serialize result based on protocol"""
            if _protocol == 'npz':
                np.savez_compressed(_filename, data=_result)
            elif protocol == 'pkl':
                with open(_filename, 'wb') as f:
                    pickle.dump(_result, f)
            else:
                raise ValueError('Unknown serialization protocol {}'.format(_protocol))

        def deserialize(_filename, _protocol):
            """De-serialize result based on protocol"""
            if _protocol == 'npz':
                return np.load(_filename)['data']
            elif _protocol == 'pkl':
                with open(_filename, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError('Unknown de-serialization protocol {}'.format(_protocol))

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # Hash args/kwargs with `pickle.dumps(...)
            # to ensure that the disk-cached objects are re-loaded across runs.
            # Include function name in case different subclasses are instantiated with the same signature.
            try:
                # Cross-process deterministic hash via `pickle.dumps()`.
                data = pickle.dumps((args, kwargs, func.__name__))
                h = hashlib.md5(data)
            except Exception as e:  # pylint: disable=bare-except
                raise Exception('Failed to hash: (args={}, kwargs={}): {}'.format(args, kwargs, str(e))) from e
            filename = os.path.join(cache_dir, '{}.{}'.format(h.hexdigest(), protocol))
            # Try loading the cached object, otherwise re-compute.
            try:
                if os.path.exists(filename):
                    logging.info(
                        'Attempting to load disk-cached object {} [{:.2f} MiB] .'.format(
                            filename,
                            os.stat(filename).st_size / 1024 / 1024
                        )
                    )
                    result = deserialize(filename, protocol)
                    logging.info(
                        'Successfully loaded disk-cached object ({} at {}) .'.format(
                            result.__class__.__name__, filename
                        )
                    )
                    return result
            except:  # pylint: disable=bare-except
                logging.info('Failed to load cached object {}'.format(filename))
            # Evaluate the function and cache it.
            result = func(*args, **kwargs)
            serialize(result, filename, protocol)
            return result

        return wrapped_func

    return wrapped_diskcache
