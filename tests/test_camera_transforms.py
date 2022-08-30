# Copyright 2021-2022 Woven Planet. All rights reserved.
import os
import unittest

import cv2
import numpy as np

from dgp.annotations.camera_transforms import (
    AffineCameraTransform,
    CompositeAffineTransform,
    CropScaleTransform,
    ScaleAffineTransform,
    ScaleHeightTransform,
    calc_affine_transform,
)
from dgp.annotations.key_line_2d_annotation import KeyLine2DAnnotationList
from dgp.annotations.key_point_2d_annotation import KeyPoint2DAnnotationList
from dgp.annotations.ontology import KeyLineOntology, KeyPointOntology
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.proto.ontology_pb2 import Ontology as OntologyV2Pb2
from dgp.utils.structures.key_line_2d import KeyLine2D
from dgp.utils.structures.key_point_2d import KeyPoint2D
from dgp.utils.visualization_utils import visualize_cameras
from tests import TEST_DATA_DIR

# Flag to render test images
DEBUG = False


def assert_almost_equal(datum1, datum2, valid_region=None):
    """Test if two camera datums are the same by comparing their annotations and rgb values.
    Since we intend to use this to test operations that can remove information (like borders when you rotate)
    we can compare values at a central region by passing valid_region.

    Parameters
    ----------
    datum1: dict(str, any)
        A camera datum

    datum2: dict(str, any)
        Another camera datum

    valid_region: tuple
        An [x1,y2, x1,y2] region on which to compare image values. If None, uses entire image
    """

    # Check that we have the same keys
    keys1 = set(datum1.keys())
    keys2 = set(datum2.keys())
    assert len(keys1 - keys2) == 0

    if 'intrinsincs' in keys1:
        assert np.allclose(datum1['intrinsics'], datum2['intrinsics'])
        # Validate our assumptions about the form of the intrinsics:
        # Upper triangular
        assert np.allclose(datum2['intrinsics'], np.triu(datum2['intrinsincs']))
        # z scale is 1
        assert np.abs(datum2['intrinsics'][2, 2] - 1) < 1e-3
        # no skew! Note: this is actually valid, just not supported well
        assert np.abs(datum2['intrinsics'][0, 1]) < 1e-3

    if 'extrinsics' in keys1:
        assert np.allclose(
            datum1['extrinsics'].matrix, datum2['extrinsics'].matrix, atol=1e-03
        ), f"{datum1['extrinsics'].rotation_matrix}, {datum2['extrinsics'].rotation_matrix}"

    if 'pose' in keys1:
        assert np.allclose(datum1['pose'].matrix, datum2['pose'].matrix, atol=1e-3)

    if 'rgb' in keys1:
        rgb1 = np.array(datum1['rgb'])
        rgb2 = np.array(datum2['rgb'])

        assert rgb1.shape == rgb2.shape

        if valid_region is None:
            h, w = rgb1.shape[:2]
            x1, y1, x2, y2 = (0, 0, w, h)
        else:
            x1, y1, x2, y2 = valid_region

        rgb1 = rgb1[y1:y2, x1:x2]
        rgb2 = rgb2[y1:y2, x1:x2]

        if DEBUG:
            idx = np.random.randint(1000)
            cv2.imwrite(f'rgb_{idx}.jpeg', rgb1)
            cv2.imwrite(f'rgb_{idx}_2.jpeg', rgb2)

        # We cannot directly compare two images. One image may have been scaled heavily and therefore blurred
        # so we compare peak signal to noise. The threshold here is abitrary and set manually
        # TODO(chrisochoatri): get better threshold
        assert cv2.PSNR(rgb1, rgb2) >= 30.0

    if 'bounding_box_2d' in keys1:
        # We cannot easily test bounding box 2d. This is because when we transform the corners
        # we then replace the box with the smallest axis aligned box that contains those corners.
        # Consider for example rotating an image by 45 degrees. The resulting box in the rotated version
        # will be much larger than the original box. When we rotate back by -45, the box will yet again
        # be bigger. We can however at least test that center of the box has not changed.
        boxes1 = datum1['bounding_box_2d']
        boxes2 = datum2['bounding_box_2d']

        assert len(boxes1) == len(boxes2)

        for box1, box2 in zip(boxes1, boxes2):
            x1, y1, x2, y2 = box1.ltrb
            center1 = np.array([(x2 + x1) / 2, (y2 + y1) / 2])
            x1, y1, x2, y2 = box2.ltrb
            center2 = np.array([(x2 + x1) / 2, (y2 + y1) / 2])
            assert np.allclose(center1, center2, atol=1e-3)

    if 'bounding_box_3d' in keys1:
        boxes1 = datum1['bounding_box_3d']
        boxes2 = datum2['bounding_box_3d']

        assert len(boxes1) == len(boxes2)

        for box1, box2 in zip(boxes1, boxes2):
            assert np.allclose(box1.corners, box2.corners, atol=1e-3), f'{box1.corners}, {box2.corners}'
            assert box1.class_id == box2.class_id

    if 'key_point_2d' in keys1:
        points1 = datum1['key_point_2d']
        points2 = datum2['key_point_2d']
        assert np.allclose(points1.xy, points2.xy)

    if 'rgb' in keys1 and 'bounding_box_3d' in keys1:
        # Render the cuboids on the image, check that both images are similar
        rgb1 = visualize_cameras(
            [datum1],
            {i: ''
             for i in range(100)},
            None,
        )[0]

        rgb2 = visualize_cameras(
            [datum2],
            {i: ''
             for i in range(100)},
            None,
        )[0]

        assert rgb1.shape == rgb2.shape

        if valid_region is None:
            h, w = rgb1.shape[:2]
            x1, y1, x2, y2 = (0, 0, w, h)
        else:
            x1, y1, x2, y2 = valid_region

        rgb1 = rgb1[y1:y2, x1:x2]
        rgb2 = rgb2[y1:y2, x1:x2]

        if DEBUG:
            idx = np.random.randint(1000)
            cv2.imwrite(f'box_vis_{idx}.jpeg', rgb1)
            cv2.imwrite(f'box_vis_{idx}_2.jpeg', rgb2)

        assert cv2.PSNR(rgb1, rgb2) >= 20.0

    if 'rgb' in keys1 and 'key_point_2d' in keys1:
        # Render the points on the images and compare images
        rgb1 = np.array(datum1['rgb'])
        for point in datum1['key_point_2d'].xy:
            rgb1 = cv2.circle(rgb1, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

        rgb2 = np.array(datum2['rgb'])
        for point in datum2['key_point_2d'].xy:
            rgb2 = cv2.circle(rgb2, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

        if valid_region is None:
            h, w = rgb1.shape[:2]
            x1, y1, x2, y2 = (0, 0, w, h)
        else:
            x1, y1, x2, y2 = valid_region

        rgb1 = rgb1[y1:y2, x1:x2]
        rgb2 = rgb2[y1:y2, x1:x2]

        if DEBUG:
            idx = np.random.randint(1000)
            cv2.imwrite(f'kp_{idx}.jpeg', rgb1)
            cv2.imwrite(f'kp_{idx}_2.jpeg', rgb2)

        # We cannot directly compare two images. One image may have been scaled heavily and therefore blurred
        # so we compare peak signal to noise. The threshold here is abitrary and set manually
        # TODO(chrisochoatri): get better threshold
        assert cv2.PSNR(rgb1, rgb2) >= 20.0

    if 'rgb' in keys1 and 'key_line_2d' in keys1:
        rgb1 = np.array(datum1['rgb'])
        for line in datum1['key_line_2d'].linelist:
            polyline = line.xy.T.astype(np.int32)
            polyline = np.expand_dims(polyline, 0)
            rgb1 = cv2.polylines(rgb1, polyline, False, (0, 255, 0), 10, cv2.LINE_AA)

        rgb2 = np.array(datum2['rgb'])
        for line in datum2['key_line_2d'].linelist:
            polyline = line.xy.T.astype(np.int32)
            polyline = np.expand_dims(polyline, 0)
            rgb2 = cv2.polylines(rgb2, polyline, False, (0, 255, 0), 10, cv2.LINE_AA)

        if valid_region is None:
            h, w = rgb1.shape[:2]
            x1, y1, x2, y2 = (0, 0, w, h)
        else:
            x1, y1, x2, y2 = valid_region

        rgb1 = rgb1[y1:y2, x1:x2]
        rgb2 = rgb2[y1:y2, x1:x2]

        if DEBUG:
            idx = np.random.randint(1000)
            cv2.imwrite(f'kl_{idx}.jpeg', rgb1)
            cv2.imwrite(f'kl_{idx}_2.jpeg', rgb2)

        assert cv2.PSNR(rgb1, rgb2) >= 20.0

    # TODO(chrisochoatri): test other annotations


def add_keypoints2d(datum):
    """Helper function to add some keypoints to a datum for testing

    Parameters
    ----------
    datum: dict
        The datum to process

    Returns
    -------
    new_datum: dict
        A datum with keypoints added
    """

    # Make a mini ontology
    ontology_pb2 = OntologyV2Pb2()
    item = ontology_pb2.items.add()
    item.id = 1
    item.isthing = True
    item.name = 'Orb'
    ontology = KeyPointOntology(ontology_pb2)

    # Make some points
    pointlist = [
        KeyPoint2D(np.array([1.0 * i, j]), class_id=1) for i in range(0, 1000, 100) for j in range(0, 1000, 20)
    ]

    datum['key_point_2d'] = KeyPoint2DAnnotationList(ontology, pointlist)
    return datum


def add_keylines2d(datum):
    """Helper function to add some keylines to a datum for testing

    Parameters
    ----------
    datum: dict
        The datum to process

    Returns
    -------
    new_datum: dict
        A datum with keylines added
    """

    # Make a mini ontology
    ontology_pb2 = OntologyV2Pb2()
    item = ontology_pb2.items.add()
    item.id = 1
    item.isthing = True
    item.name = 'SomeLine'
    ontology = KeyLineOntology(ontology_pb2)

    # Make some lines
    linelist = []
    for y in range(0, 1000, 200):
        points = np.stack([np.array([x, 1.0 * y]) for x in range(0, 1000, 100)])
        line = KeyLine2D(line=points, class_id=1)
        linelist.append(line)

    datum['key_line_2d'] = KeyLine2DAnnotationList(ontology, linelist)
    return datum


class TestTransforms(unittest.TestCase):
    """Test camera datum transformations"""
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")

    def setUp(self):
        # Initialize synchronized dataset
        scenes_dataset_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
        self.dataset = SynchronizedSceneDataset(
            scenes_dataset_json,
            split='train',
            datum_names=[
                'camera_01',
            ],
            backward_context=0,
            requested_annotations=(
                "bounding_box_2d",
                "bounding_box_3d",
            )
        )

    def test_affine_transform(self):
        """Test base class by generating a transform, applying it, and then applying the inverse"""
        cam_datum = self.dataset[0][0][0]

        # This dataset does not have keypoints, add some for testing
        cam_datum = add_keypoints2d(cam_datum)
        cam_datum = add_keylines2d(cam_datum)

        # Initial image size. Note: pil size is w,h not h,w
        w, h = cam_datum['rgb'].size
        A = calc_affine_transform(theta=45, scale=.9, flip=1, shiftx=10, shifty=-20, shear=0, img_shape=(h, w))
        tr = AffineCameraTransform(A=A, shape=(h, w))
        cam_datum2 = tr(cam_datum)

        if DEBUG:
            rgb_viz = visualize_cameras(
                [cam_datum2],
                {i: ''
                 for i in range(32)},
                None,
            )[0]

            if 'key_point_2d' in cam_datum2:
                for point in cam_datum2['key_point_2d'].xy:
                    rgb_viz = cv2.circle(rgb_viz, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

            if 'key_line_2d' in cam_datum2:
                lines = cam_datum2['key_line_2d'].linelist
                for line in lines:
                    ll = line.xy.T.astype(np.int32)
                    ll = np.expand_dims(ll, 0)
                    rgb_viz = cv2.polylines(rgb_viz, ll, False, (0, 255, 0), 10, cv2.LINE_AA)

            cv2.imwrite('affine_test_intermediate.jpeg', rgb_viz)

        # Test round trip. We should be able to (mostly) recover the initial datum
        Ainv = np.linalg.inv(tr.A)
        target_shape = (h, w)
        tr_inv = AffineCameraTransform(A=Ainv, shape=target_shape)

        cam_datum3 = tr_inv(cam_datum2)

        # Due to border issues, only check rgb values at central region
        dw = w // 4
        dh = h // 4
        valid_region = (dw, dh, w - dw, h - dh)

        assert_almost_equal(cam_datum, cam_datum3, valid_region=valid_region)

    def test_scale_transform(self):
        """Test scale transform"""

        cam_datum = self.dataset[0][0][0]
        cam_datum = add_keypoints2d(cam_datum)
        cam_datum = add_keylines2d(cam_datum)

        # Initial image size. Note: pil size is w,h not h,w
        w, h = cam_datum['rgb'].size

        s = .5
        tr = ScaleAffineTransform(s)
        cam_datum2 = tr(cam_datum)

        w2, h2 = cam_datum2['rgb'].size
        assert int(w * s) == w2
        assert int(h * s) == h2

        # Apply the inverse transform and verify everything is the same
        tr_inv = ScaleAffineTransform(1 / s)
        cam_datum3 = tr_inv(cam_datum2)

        assert_almost_equal(cam_datum, cam_datum3)

    def test_scale_height_transform(self):
        """Test scale by height transform"""
        cam_datum = self.dataset[0][0][0]
        cam_datum = add_keypoints2d(cam_datum)
        cam_datum = add_keylines2d(cam_datum)
        _, h = cam_datum['rgb'].size

        s = 2
        hs = int(h * s)
        tr = ScaleHeightTransform(hs)
        cam_datum2 = tr(cam_datum)

        _, h2 = cam_datum2['rgb'].size
        assert hs == h2

        # Apply the inverse transform and verify everything is the same
        hs = int(hs * 1 / s)
        tr_inv = ScaleHeightTransform(hs)
        cam_datum3 = tr_inv(cam_datum2)

        assert_almost_equal(cam_datum, cam_datum3)

    def test_crop_scale_transform(self):
        """Test the crop transform"""
        cam_datum = self.dataset[0][0][0]
        cam_datum = add_keypoints2d(cam_datum)
        cam_datum = add_keylines2d(cam_datum)
        w, h = cam_datum['rgb'].size

        target_shape = (h // 2, w // 2)
        tr = CropScaleTransform(target_shape=target_shape, fix_h=True)
        cam_datum2 = tr(cam_datum)

        w2, h2 = cam_datum2['rgb'].size
        assert target_shape == (h2, w2)

        # Apply the inverse transform and verify everything is the same
        Ainv = np.linalg.inv(tr.A)
        target_shape = (h, w)
        tr_inv = AffineCameraTransform(A=Ainv, shape=target_shape)

        cam_datum3 = tr_inv(cam_datum2)

        # Get a region that should be unchanged. There is no way for the inverse
        # transform to restore the borders we cropped. So we cannot evaluate them
        dw = (w - w2) // 2 + 1
        dh = (h - h2) // 2 + 1
        valid_region = (dw, dh, w - dw, h - dh)

        assert_almost_equal(cam_datum, cam_datum3, valid_region=valid_region)

    def test_composite_transform(self):
        """Test that we can compose transforms correctly. We test that we can get the
        same datum by applying multiple transformation consecutively vs all at once.
        We also test that apply the inverse works correctly"""
        cam_datum = self.dataset[0][0][0]
        cam_datum = add_keypoints2d(cam_datum)
        cam_datum = add_keylines2d(cam_datum)

        w, h = cam_datum['rgb'].size
        A1 = calc_affine_transform(theta=15, scale=1, flip=0, shiftx=0, shifty=0, shear=0, img_shape=(h, w))
        tr1 = AffineCameraTransform(A1, (h, w))  # rotation

        A2 = calc_affine_transform(theta=0, scale=1, flip=1, shiftx=0, shifty=0, shear=0, img_shape=(h, w))
        tr2 = AffineCameraTransform(A2, (h, w))  # left right flip

        tr_comp = CompositeAffineTransform(transforms=[tr2, tr1], )

        # The chained transform
        cam_datum2 = tr2(tr1(cam_datum))

        # The composite transform
        cam_datum3 = tr_comp(cam_datum)

        # get a valid region in the center. The composite transform might actuall preseve some of the border information
        # that would otherwise be lost in a sequential operation.
        dw = w // 4
        dh = h // 4
        valid_region = (dw, dh, w - dw, h - dh)

        assert_almost_equal(cam_datum2, cam_datum3, valid_region=valid_region)

        # test round trip
        Ainv = np.linalg.inv(tr_comp.A)
        target_shape = (h, w)
        tr_inv = AffineCameraTransform(A=Ainv, shape=target_shape)

        cam_datum4 = tr_inv(cam_datum3)

        dw = w // 4
        dh = h // 4
        valid_region = (dw, dh, w - dw, h - dh)

        assert_almost_equal(cam_datum, cam_datum4, valid_region=valid_region)

        # Finally, if we compose with the inverse, nothing should really happen
        tr_comp = CompositeAffineTransform(transforms=[tr2, tr1, tr_inv], )
        cam_datum5 = tr_comp(cam_datum)
        assert np.allclose(tr_comp.A, np.eye(3))
        assert_almost_equal(
            cam_datum,
            cam_datum5,
        )


if __name__ == "__main__":
    unittest.main()
