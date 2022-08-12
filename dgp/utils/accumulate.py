# Copyright 2019-2021 Toyota Research Institute.  All rights reserved.
"""Useful utilities for accumulating sensory information over time"""
from collections import defaultdict
from copy import deepcopy

import numpy as np


def points_in_cuboid(query_points, cuboid):
    """Tests if a point is contained by a cuboid. Assumes points are in the same frame as the cuboid, 
    i.e, cuboid.pose translates points in the cuboid local frame to the query_point frame.

    Parameters
    ----------
    query_points: np.ndarray
        Numpy array shape (N,3) of points.

    cuboid: dgp.utils.structures.bounding_box_3d.BoundingBox3D
        Cuboid in same reference frame as query points

    Returns
    -------
    in_view: np.ndarray
        Numpy boolean array shape (N,) where a True entry denotes a point insided the cuboid
    """
    # To test if a point is inside a cuboid, we select a reference point on the cube,
    # and construct three vectors denoting the offests to the three parallel sides of the cuboid.
    # Testing if a query point is insde the cuboid then amounts to testing if the vector from
    # the refrence point to the query point, when projected on to these three vectors, is not
    # longer than these three vectors.

    # Use cuboid.corners to get the corner points and use these to construct our three offset vectors.
    # Note: we could just grab these directly from the columns of the cuboid rotation matrix, however
    # this would then require we transform all the query points to the cuboid frame to test. Likely faster
    # to compute these vectors instead of doing a large matrix multiply.

    # Cuboid corners returns 8 points:
    # idx 0: L/2, W/2, H/2
    # idx 1 L/2 -W/2, H/2
    # idx 3: L/2, W/2,-H/2
    # idx 4: -L/2 W/2 H/2

    # Get some vectors starting at the anchor point 0
    corners = cuboid.corners
    a = corners[1] - corners[0]
    b = corners[3] - corners[0]
    c = corners[4] - corners[0]

    # Get the length of these vectors and normalize them to make testing easier
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    mc = np.linalg.norm(c)
    a = a / ma
    b = b / mb
    c = c / mc

    # Get the vector from the query point to the refrence point
    v = query_points - corners[0]

    # Get the length of v projected on to a, b, c
    proj_a = np.dot(v, a)
    proj_b = np.dot(v, b)
    proj_c = np.dot(v, c)

    in_view = np.logical_and.reduce([proj_a >= 0, proj_b >= 0, proj_c >= 0, proj_a <= ma, proj_b <= mb, proj_c <= mc])

    return in_view


def accumulate_points(point_datums, target_datum, transform_boxes=False):
    """Accumulates lidar or radar points by transforming all datums in point_datums to the frame used in target_datum.

    Parameters
    ----------
    point_datums: list[dict]
        List of datum dictionaries to accumulate

    target_datum: dict
        A single datum to use as the reference frame and reference time.

    transform_boxes: bool, optional
        Flag to denote if cuboid annotations and instance ids should be used to warp points to the target frame.
        Only valid for Lidar.
        Default: False.

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

    target_boxes = {}
    if transform_boxes and 'bounding_box_3d' in target_datum:
        target_boxes = {box.instance_id: box for box in target_datum['bounding_box_3d']}

    for _, p in enumerate(point_datums):
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

        if transform_boxes and 'bounding_box_3d' in p:
            # Skip if this is a radar
            if 'velocity' in p:
                continue

            for _box in p['bounding_box_3d']:
                if _box.instance_id not in target_boxes:
                    continue

                # Move box to local frame (since points are now in local)
                box = deepcopy(_box)
                box._pose = pose_p2p1 * box.pose

                # TODO: maybe shrink cuboid slightly to prevent accidentally moving ground points
                in_box = points_in_cuboid(new_fields['point_cloud'][-1], box)

                if np.any(in_box):
                    points_to_move = new_fields['point_cloud'][-1][in_box]
                    # move the points local A -> cuboid -> target local
                    pose_p2box1 = target_boxes[box.instance_id].pose * box.pose.inverse()
                    moved_points = pose_p2box1 * points_to_move
                    new_fields['point_cloud'][-1][in_box] = moved_points

                # TODO: check that no point belongs to more than one cuboid,
                # if so, perhaps break these ties by distance to org cuboid center?

    for k in new_fields:
        p_target[k] = np.concatenate(new_fields[k], axis=0)

    return p_target
