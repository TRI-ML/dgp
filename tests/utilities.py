import os

import pytest


def requires_env(*envs):
    """
    annotation that only runs tests in specified workflow or environment

    Parameters
    ----------
    *envs: Sequence of str
        One or more string names identifying the runtime environment we're in. Ex: `test`.

    Returns
    -------
    None
    """
    env = os.environ.get('ENVIRONMENT', 'test')

    envs = envs if isinstance(envs, list) else [*envs]

    return pytest.mark.skipif(env not in envs, reason=f"Not suitable envrionment {env} for current test")
