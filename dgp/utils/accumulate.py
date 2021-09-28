# Copyright 2019-2021 Toyota Research Institute.  All rights reserved.
"""Useful utilities for accumulating sensory information over time"""
from collections import defaultdict
from copy import deepcopy

import numpy as np


def accumulate_points(point_datums, target_datum):
    """Accumulates lidar or radar points by transforming all datums in point_datums to the frame used in target_datum.

    Parameters
    ----------
        point_datums: list[dict]
            List of datum dictionaries to accumulate

        target_datum: dict
            A single datum to use as the reference frame and reference time.

    Returns
    -------
        p_target: dict
            A new datum with accumulated points and an additional field 'accumulation_offset_t'
            that indicates the delta in microseconds between the target_datum and the given point.
    """

    assert 'point_cloud' in point_datums[0], "Accumulation is only defined for radar and lidar currently."

    p_target = deepcopy(target_datum)
    pose_target_w = p_target['pose'].inverse()

    new_fields = defaultdict(list)

    for p in point_datums:
        # Move p into global world frame, then transform from world to p_target.
        pose_p2p1 = pose_target_w * p['pose']

        new_points = pose_p2p1 * p['point_cloud']
        new_fields['point_cloud'].append(new_points)

        if 'velocity' in p_target:
            # Transform the velocity by moving head and tail.
            new_vel = pose_p2p1 * (p['point_cloud'] + p['velocity']) - new_points
            new_fields['velocity'].append(new_vel)

        if 'covariance' in p_target:
            # The covariance matrix only needs to be rotated.
            R = pose_p2p1.rotation_matrix
            new_cov = R @ p['covariance'] @ R.T
            new_fields['covariance'].append(new_cov)

        if 'extra_channels' in p_target:
            # The extra channels should not have any geometry, just append them.
            new_fields['extra_channels'].append(p['extra_channels'])

        dt = p_target['timestamp'] - p['timestamp']
        new_fields['accumulation_offset_t'].append(dt * np.ones(len(new_points)))

    for k in new_fields:
        p_target[k] = np.concatenate(new_fields[k], axis=0)

    return p_target
