# Transformations for camera datums in DGP Synchronized Scene Format
# Copyright 2021-2022 Woven Planet. All rights reserved.
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import PIL
from PIL.ImageTransform import AffineTransform

from dgp.annotations.bounding_box_2d_annotation import (
    BoundingBox2DAnnotationList,
)
from dgp.annotations.bounding_box_3d_annotation import (
    BoundingBox3DAnnotationList,
)
from dgp.annotations.depth_annotation import DenseDepthAnnotation
from dgp.annotations.key_line_2d_annotation import KeyLine2DAnnotationList
from dgp.annotations.key_point_2d_annotation import KeyPoint2DAnnotationList
from dgp.annotations.panoptic_segmentation_2d_annotation import (
    PanopticSegmentation2DAnnotation,
)
from dgp.annotations.semantic_segmentation_2d_annotation import (
    SemanticSegmentation2DAnnotation,
)
from dgp.annotations.transforms import BaseTransform
from dgp.utils.pose import Pose
from dgp.utils.structures.bounding_box_2d import BoundingBox2D

logger = logging.getLogger(__name__)

# NOTE: Some opencv operations can lead to deadlocks when using multiprocess.fork instead of spawn.
# If you experience deadlocks, try adding cv2.setNumThreads(0)


def calc_affine_transform(
    theta: float,
    scale: float,
    flip: bool,
    shiftx: float,
    shifty: float,
    shear: float,
    img_shape: Union[Tuple[int, int], Tuple[int, int, int]],
) -> np.ndarray:
    """Generates a matrix corresponding to an affine transform for the given inputs.

    Parameters
    ----------
    theta: float
        Rotation angle in degrees.
    scale: float
        Scale factor such as .5 of half size or 2.0 for double size.
    flip: bool
        If true, perform a left right flip.
    shiftx: float
        Amount in pixels to shift horizontally.
    shifty: float
        Amount in pixels to shift vertically.
    shear: float
        Scale factor for image shear.
    img_shape: tuple
        Tuple corresponding to img shape ie (h,w,3) or (h,w).

    Returns
    -------
    A: np.ndarray
        3x3 matrix that expresses the requested transformations.
    """
    h, w = img_shape[:2]

    # Rotate and scale
    # TODO(chrisochoatri): break scale into scale_y and scale_x?
    R = cv2.getRotationMatrix2D((w / 2, h / 2), theta, scale)
    R = np.vstack([R, np.array([0, 0, 1.0])])

    # Shift and shear
    if shear != 0:
        logger.warning('Shear was set to non zero, shear is not well supported by many downstream operations')

    S = np.array([[1.0, shear, shiftx], [0.0, 1.0, shifty], [0.0, 0.0, 1.0]])

    # Left/Right flip
    F = np.eye(3)
    if flip:
        F[0][0] = -1
        F[0][-1] = w

    # TODO(chrisochoatri): expose operation order
    A = F @ S @ R

    return A


def box_crop_affine_transform(
    box_ltrb: Tuple[int, int, int, int],
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Generates a matrix that crops a rectangular area from an image and resizes it to target shape.
    Note, this preserves the aspect ratio in target shape.

    Parameters
    ----------
    box_ltrb: list or tuple
        Box corners expressed as left, top, right, bottom (x1,y1,x2,y2).
    target_shape: tuple
        Desired image shape (h,w) after cropping and resizing.

    Returns
    -------
    A: np.ndarray
        3x3 matrix that expresses the requested transformation.
    """
    # get box center
    x1, y1, x2, y2 = box_ltrb
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    target_aspect_ratio = target_shape[0] / target_shape[1]

    # keep the box height fixed and adjust the width
    new_h = h
    new_w = h / target_aspect_ratio
    scale = target_shape[0] / new_h

    ax, ay = cx - new_w / 2, cy - new_h / 2

    A = np.array([[scale, 0, -ax * scale], [0, scale, -ay * scale], [0, 0, 1.0]])
    return A


def scale_affine_transform(s: float) -> np.ndarray:
    """Generates a matrix performs a unfirom scaling.

    Parameters
    ----------
    s: float
        scale factor

    Returns
    -------
    A: np.ndarray
        3x3 matrix that expresses the requested transformation.
    """
    return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1]])


def transform_box_2d(box: BoundingBox2D, A: np.ndarray) -> BoundingBox2D:
    """Apply an affine transformation to a 2d box annotation.

    Parameters
    ----------

    box: BoundingBox2DAnnotation
        Box to transform.

    A: np.ndarray
        3x3 transformation matrix

    Returns
    -------
    box: BoundingBox2DAnnotation
        Box annotation with updated positions.
    """
    # get the corners of all the boxes
    x1, y1, x2, y2 = box.ltrb

    points = np.array([[x1, y1, 1], [x1, y2, 1], [x2, y2, 1], [x2, y1, 1]])

    new_points = (A[:2, :] @ points.T).T
    x1, y1 = new_points.min(axis=0)
    x2, y2 = new_points.max(axis=0)
    # Note these new points could be outside of the image now.
    # TODO(chrisochoatri): expose option to either clip, remove, or keep these boxes

    box.l = x1
    box.t = y1
    box.w = x2 - x1
    box.h = y2 - y1

    return box


class AffineCameraTransform(BaseTransform):
    """Base transform class for 2d geometric camera transformations. 
    This serves as a base to implement 2d image transforms such as scaling, rotation, left-right flips etc
    as affine transforms. Doing so makes it very easy to apply the same transform to 2d box annotations and
    semantic segmentation maps. Additionally by implementing transforms as a matrix multiplies, multiple transforms
    can be implemented with a single multiply/remap without losing information along the image borders.
    """
    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        shape: Optional[Union[Tuple[int, int], Tuple[int, int, int]]] = None,
        fix_skew: bool = True,
    ) -> None:
        """Implements an affine transform to camera datum. 
        This operates on DGP camera datums (OrderedDict) and returns Camera datums.

        Parameters
        ----------
        A: np.ndarray
            3x3 affine transformation matrix

        shape: tuple
            Desired image shape after applying transformation

        fix_skew: bool
            If true, attempt to remove skew from the operations to comply with camera classes
            that do not model it. If using this, you are not guaranteed to be able to recover the inverse
            operation by inverting the transformation matrix.
        """

        self.A = A
        self.shape = shape
        self.fix_skew = fix_skew

    def _calc_A(
        self,
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],  # pylint : ignore unused
    ) -> np.ndarray:
        """Calculates transformation matrix as a function of input image shape.

        Parameters
        ----------
        input_shape: tuple
            Shape of input camera's image, i.e, (h,w,3) or (h,w).

        Returns
        -------
        A: np.ndarray
            3x3 affine transformation matrix
        """
        return self.A

    def _calc_shape(
        self, input_shape: Union[Tuple[int, int], Tuple[int, int, int]]
    ) -> Union[Tuple[int, int], Tuple[int, int, int]]:
        """Calculates new shape of image after transformation as a function of input image shape.

        Parameters
        ----------
        input_shape: tuple
            Shape of input camera's image, i.e, (h,w,3) or (h,w).
 
        Returns
        -------
        shape: tuple
            new image shape.
        """
        return self.shape

    def transform_image(
        self,
        img: Union[np.ndarray, PIL.Image.Image],
        mode: int = cv2.INTER_LINEAR,
    ) -> Union[np.ndarray, PIL.Image.Image]:
        """Applies transformation to an image.

        Parameters
        ----------
        img: np.ndarray or PIL.Image.Image
            Input image expressed as a numpy array of type np.uint8 or np.float32, or, PIL Image.

        mode: int
            Opencv flag for interpolation mode. When used on masks or label image,
            should be set to cv2.INTER_NEAREST, otherwise defaults to bilnear interpolation.
            NOTE: when image is a PIL Image, the correponding PIL flag is subsitituted automatically.

        Returns
        -------
        new_img: np.ndarray or PIL.Image.Image
            New transformed image.

        Raises
        ------
        ValueError
            If the mode is not one of cv2.INTER_LINEAR or cv2.INTER_NEAREST
        """
        h, w = self.shape[:2]

        if isinstance(img, PIL.Image.Image):
            # Note: PIL transform takes the inverse
            m = np.linalg.inv(self.A)[:2, :].flatten()

            tx = AffineTransform(m)
            if mode == cv2.INTER_LINEAR:
                mode = PIL.Image.BILINEAR
            elif mode == cv2.INTER_NEAREST:
                mode = PIL.Image.NEAREST
            else:
                raise ValueError(f'{mode} not supported')

            new_img = tx.transform((w, h), img, resample=mode)
        else:
            new_img = cv2.warpAffine(img, self.A[:2, :], (int(w), int(h)), mode)

        return new_img

    def transform_camera(
        self,
        cam_datum: Dict[str, Any],
    ) -> Tuple[np.ndarray, Pose, Pose]:
        """Transform camera intrinisics, extrinsics

        Parameters
        ----------
        cam_datum : Dict[str,Any]
            A dgp camera datum

        Returns
        -------
        mtxR: np.array
            new camera intrinsics
        new_pose: Pose
            The new pose (sensor to global)
        new_ext: Pose
            The new extrinsics (sensor to local)
        """

        # Transform the camera matrix
        h, w = self.shape[:2]

        # Flipping leads to the wrong rotation below, so when there is a flip present,
        # we unflip, do everything, and re flip
        flip_mat = np.eye(3)
        flip = False
        if self.A[0, 0] < 0:
            flip = True
            F = np.eye(3)
            F[0][0] = -1
            F[0][-1] = w
            flip_mat = F

        if self.A[1, 1] < 0:
            flip = True
            F2 = np.eye(3)
            F2[1][1] = -1
            F2[1][-1] = h
            flip_mat = F2 * flip_mat

        A = np.linalg.inv(flip_mat) @ self.A

        mtxR = A @ cam_datum['intrinsics']

        # NOTE: some opencv functions expect the camera matrix to be upper triangular, so we have to stuff any rotation
        # into the extrinsics. We decompose the new camera matrix into an upper triangular camera matrix and a rotation,
        # and then bake that rotation into the extrinsics matrix.
        _, mtxR, mtxQ, _, _, _ = cv2.RQDecomp3x3(mtxR)

        # The decomposition may have negated the last column, which in many applications assumes the [2,2] element is 1.0
        if mtxR[2, 2] < 0:
            fix = np.eye(3)
            fix[0, 0] = -1
            fix[2, 2] = -1
            mtxR = mtxR @ fix
            mtxQ = fix.T @ mtxQ  # fix.T == inv(fix)

        # Additionally DGP camera class does not model skewness, which limits this approach
        # we can fix/force skew to be zero at the cost of (potentially) changing our rotation.
        if self.fix_skew:
            shear_mat = np.eye(3)
            shear_mat[0, 1] = mtxR[0, 1] / mtxR[0, 0]  # s/fx
            mtxR = mtxR @ np.linalg.inv(shear_mat)
            mtxQ = shear_mat @ mtxQ

            # This shear_mat may have ruined our rotation (rounding errors or invalid rotation) so re-orthogonalize
            u, s, v = np.linalg.svd(mtxQ)
            mtxQ = u @ v

        # Flip back
        if flip:
            mtxR = flip_mat @ mtxR

        new_R = (mtxQ @ cam_datum['pose'].rotation_matrix.T).T
        new_pose = Pose().from_rotation_translation(new_R, cam_datum['pose'].tvec)

        new_R = (mtxQ @ cam_datum['extrinsics'].rotation_matrix.T).T
        new_ext = Pose().from_rotation_translation(new_R, cam_datum['extrinsics'].tvec)

        # NOTE: distortion parameters are typically independent, so we do not modify them here

        return mtxR, new_pose, new_ext

    def transform_detections_3d(
        self, boxes: BoundingBox3DAnnotationList, pose_correction: Pose
    ) -> BoundingBox3DAnnotationList:
        """Applies trasformation matrix to 3d cuboids

        Parameters
        ----------
        boxes: BoundingBox3DAnnotationList
            The 3d cuboids for this camera

        pose_correction: Pose
            Pose used to correct and change in extrinsics due to rotations

        Returns
        -------
        boxes: BoundingBox3DAnnotationList
        """
        # Pose correction is only relevant for boxes in camera frame
        boxes = deepcopy(boxes)
        for b in boxes:
            b._pose = pose_correction * b.pose
        return boxes

    def transform_detections_2d(
        self,
        boxes: BoundingBox2DAnnotationList,
    ) -> BoundingBox2DAnnotationList:
        """Applies transformation matrix to list of bounding boxes:

        Parameters
        ----------
        boxes: BoundingBox2DAnnotationList
            List of bounding box annotations.

        Returns
        -------
        new_boxes: BoundingBox2DAnnotationList
            List of transformed bounding box annotations.
        """
        new_boxes = deepcopy(boxes)
        for box in new_boxes:
            box = transform_box_2d(box, self.A)

        return new_boxes

    def transform_semantic_segmentation_2d(
        self,
        semantic_segmentation_2d: SemanticSegmentation2DAnnotation,
    ) -> SemanticSegmentation2DAnnotation:
        """Applies transformation to semantic segmentation annotation.

        Parameters
        ----------
        semantic_segmentation_2d: SemanticSegmentation2DAnnotation
            Semantic segmentation input

        Returns
        -------
        new_sem_seg: SemanticSegmentation2DAnnotation
            New transformed semantic segmentation annotation.
        """
        new_sem_seg = deepcopy(semantic_segmentation_2d)
        new_sem_seg._segmentation_image = self.transform_image(new_sem_seg._segmentation_image, mode=cv2.INTER_NEAREST)
        return new_sem_seg

    def transform_depth(
        self,
        depth: DenseDepthAnnotation,
    ) -> DenseDepthAnnotation:
        """Applies transformation to depth annotation.

        Parameters
        ----------
        depth: DenseDepthAnnotation
            Depth input

        Returns
        -------
        new_depth: DenseDepthAnnotation
            New transformed depth annotation.
        """
        new_depth = deepcopy(depth)
        new_depth._depth = self.transform_image(new_depth._depth, mode=cv2.INTER_LINEAR)
        # TODO(chrisochoatri): do we want to scale depth values by the new focal length?
        return new_depth

    def transform_panoptic_segmentation_2d(
        self,
        panoptic_seg: Optional[PanopticSegmentation2DAnnotation],
    ) -> Optional[PanopticSegmentation2DAnnotation]:
        """Applies transformation to panoptic segmentation annotation.

        Parameters
        ----------
        panoptic_seg: PanopticSegmentation2DAnnotation
            Panoptic segmentation input

        Returns
        -------
        new_panoptic_seg: PanopticSegmentation2DAnnotation
            New transformed panoptic segmentation annotation or None if the input panoptic_sg is None
        """
        if panoptic_seg is None:
            return None

        new_panoptic_seg = deepcopy(panoptic_seg)
        for panoptic in new_panoptic_seg:
            panoptic._bitmask = self.transform_image(panoptic.bitmask.astype(np.float32),
                                                     mode=cv2.INTER_NEAREST).astype(np.bool)

            # TODO(chrisochoatri): how to treat masks with no valid pixels? ie if after a transformation,
            # if the bitmask is all zeros, should we delete this mask? currently we just keep the mask

        return new_panoptic_seg

    def transform_mask_2d(
        self,
        mask: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Transform image mask

        Parameters
        ----------
        mask: np.ndarray, optional
            A boolean mask of same shape as the image that denotes a valid pixel

        Returns
        -------
        new_mask: np.ndarray, optional
            The transformed mask or None if the input mask was None
        """

        if mask is None:
            return None

        new_mask = self.transform_image(mask.astype(np.float32), mode=cv2.INTER_NEAREST).astype(np.bool)

        return new_mask

    def transform_keypoints_2d(
        self,
        keypoints: Optional[KeyPoint2DAnnotationList],
    ) -> Optional[KeyPoint2DAnnotationList]:
        """Applies transformation matrix to list of keypoints:

        Parameters
        ----------
        keypoints: KeyPoint2DAnnotationList
            List of keypoint annotations.

        Returns
        -------
        new_keypoints: Keypoint2DAnnotationList
            List of transformed bounding keypoint annotations or None if keypoints is None
        """

        if keypoints is None:
            return None

        new_keypoints = deepcopy(keypoints)

        for kp in new_keypoints:
            x, y = kp.x, kp.y
            new_pt = self.A[:2, :] @ np.array([x, y, 1])
            kp.x, kp.y = new_pt[0], new_pt[1]
            kp._point = np.float32([new_pt[0], new_pt[1]])

        return new_keypoints

    def transform_keylines_2d(
        self,
        keylines: Optional[KeyLine2DAnnotationList],
    ) -> Optional[KeyLine2DAnnotationList]:
        """Applies transformation matrix to key lines:

        Parameters
        ----------
        keylines: KeyLine2dAnnotationList
            Keyline2d annotation list

        Returns
        -------
        new_keylines: Keyline2DAnnotationList
            Transformed keylines or None if keylines is None
        """

        if keylines is None:
            return None

        new_keylines = deepcopy(keylines)

        for line in new_keylines:
            points = line.xy.T  # (N,2)
            ones = np.expand_dims(np.ones(points.shape[0]), 1)  # (N,2)
            new_points = self.A[:2, :] @ np.concatenate([points, ones], axis=-1).T
            new_points = new_points.T[:, :2]
            line._point = new_points
            line.x = new_points[:, 0].tolist()
            line.y = new_points[:, 1].tolist()

        return new_keylines

    def transform_datum(self, cam_datum: Dict[str, Any]) -> Dict[str, Any]:  # pylint: disable=arguments-renamed
        """Applies transformation to a camera datum.

        Parameters
        ----------
        cam_datum: OrderedDict
            Camera datum to transform with at least the following keys: datum_type, rgb, pose, intrinsics, extrinsics

        Returns
        -------
        new_datum: OrderedDict
            Camera datum with transformed image and annotations.

        Raises
        ------
        NotImplementedError
            If any field is not yet supported.
        """

        assert cam_datum['datum_type'] == 'image', 'expected an image datum_type'

        assert 'rgb' in cam_datum, 'datum should contain an image'

        new_datum = cam_datum.copy()

        # We support PIL and raw numpy arrays
        if isinstance(new_datum['rgb'], PIL.Image.Image):
            input_shape = new_datum['rgb'].size[::-1]
        else:
            input_shape = new_datum['rgb'].shape

        # NOTE: we call this here since in general the transformation matrix can depend on the input shape
        self.A = self._calc_A(input_shape)
        self.shape = self._calc_shape(input_shape)
        if self.shape is None:
            self.shape = input_shape

        new_datum['rgb'] = self.transform_image(new_datum['rgb'])

        mtx, pose, ext = self.transform_camera(new_datum)
        if np.abs(mtx[0, 1]) > 1e-3:
            logger.warning('Input camera matrix had skew, this may not work with downstream applications!')

        new_datum['intrinsics'] = mtx
        new_datum['pose'] = pose
        new_datum['extrinsics'] = ext

        # This is not actually part of DGP, but if you define a mask for the image, we can keep track of points
        # that are not part of that mask a result of these operations.
        if 'rgb_mask' in new_datum:
            rgb_mask = new_datum['rgb_mask']
            rgb_mask = self.transform_mask_2d(rgb_mask)
            new_datum['rgb_mask'] = rgb_mask

        if 'bounding_box_3d' in new_datum and new_datum['bounding_box_3d'] is not None:
            # Note: DGP camera class does not model the full camera matrix just focal length and center
            # if using DGP camera class, do not use transformations that add a skew!
            boxes = new_datum['bounding_box_3d']
            pose_correction = new_datum['extrinsics'].inverse() * cam_datum['extrinsics']
            boxes = self.transform_detections_3d(boxes, pose_correction)
            new_datum['bounding_box_3d'] = boxes

        if 'bounding_box_2d' in new_datum and new_datum['bounding_box_2d'] is not None:
            boxes = new_datum['bounding_box_2d']
            boxes = self.transform_detections_2d(boxes, )
            new_datum['bounding_box_2d'] = boxes
            # TODO(chrisochoatri): remove zero w and h boxes
            # TODO(chrisochoatri): clip to image size
            # TODO(chrisochoatri): maybe convert back to int if input is int?
            # TODO(chrisochoatri): re-estimate 2d boxes after transform from instance masks if available

        if 'semantic_segmentation_2d' in new_datum:
            sem_seg = new_datum['semantic_segmentation_2d']
            sem_seg = self.transform_semantic_segmentation_2d(sem_seg, )
            new_datum['semantic_segmentation_2d'] = sem_seg

        if 'depth' in new_datum:
            depth = new_datum['depth']
            depth = self.transform_depth(depth, )
            new_datum['depth'] = depth

        if 'key_point_2d' in new_datum:
            keypoints = new_datum['key_point_2d']
            keypoints = self.transform_keypoints_2d(keypoints, )
            new_datum['key_point_2d'] = keypoints

        if 'instance_segmentation_2d' in new_datum:
            instance_seg = new_datum['instance_segmentation_2d']
            instance_seg = self.transform_panoptic_segmentation_2d(instance_seg, )
            new_datum['instance_segmentation_2d'] = instance_seg

        if 'key_line_2d' in new_datum:
            keylines = new_datum['key_line_2d']
            keylines = self.transform_keylines_2d(keylines, )
            new_datum['key_line_2d'] = keylines

        if 'key_line_3d' in new_datum:
            raise NotImplementedError('key_line_3d not yet supported')

        if 'key_point_3d' in new_datum:
            raise NotImplementedError('key_point_3d not yet supported')

        # TODO(chrisochoatri): verify behavior when Nonetype is passed for each annotation
        # TODO(chrisochoatri): line 2d/3d annotations
        # TODO(chrisochoatri): polygon annotation
        # TODO(chrisochoatri): flow 2d

        return new_datum


class ScaleAffineTransform(AffineCameraTransform):
    def __init__(self, s: float) -> None:
        """Scale a camera datum.

        Parameters
        ----------
        s: float
            Scale factor.
        """
        self.s = s
        self.A = scale_affine_transform(s)
        super().__init__(A=self.A)

    def _calc_shape(
        self,
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> Tuple[int, int]:
        h, w = input_shape[:2]
        shape = (int(h * self.s), int(w * self.s))
        return shape


class ScaleHeightTransform(AffineCameraTransform):
    def __init__(self, h: int) -> None:
        """Scale a camera datum to a specific image height.

        Parameters
        ----------
        h: float
            new height.
        """
        assert h > 0
        self.h = h
        super().__init__()

    def _calc_A(
        self,
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> np.ndarray:
        """Calculate transformation matrix. See AffineCameraTransform._calc_A"""
        h, _ = input_shape[:2]
        s = self.h / h
        return scale_affine_transform(s)

    def _calc_shape(
        self,
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> Tuple[int, int]:
        h, w = input_shape[:2]
        s = self.h / h
        shape = (self.h, int(w * s))
        return shape


class CropScaleTransform(AffineCameraTransform):
    def __init__(self, target_shape: Tuple[int, int], fix_h: bool = True) -> None:
        """Extracts a crop from the center of an image and resizes to target_shape.
        This attempts to match the aspect ratio of target_shape and does not stretch the crop.

        Parameters
        ----------
        target_shape: tuple
            Shape (h,w) after transformation.
        fix_h: bool, default=True
            If True, fixes the height and modifies the width to maintain the desired aspect ratio.
            Otherwise fixes the width and moifies the height.
        """
        self.shape = target_shape[:2]
        self.fix_h = fix_h
        super().__init__(shape=self.shape)

    def _calc_A(
        self,
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> np.ndarray:
        """Calculate transformation matrix. See AffineCameraTransform._calc_A"""

        # Get the center crop box
        h, w = input_shape[:2]
        H, W = self.shape[:2]
        aspect_ratio = H / W
        if self.fix_h:  # leaves h unchanged , crops the x
            newx = w - h / aspect_ratio
            box = [newx / 2, 0, w - newx / 2, h]
        else:
            newy = h - w * aspect_ratio
            box = [0, newy / 2, w, h - newy]

        return box_crop_affine_transform(box, self.shape)


class CompositeAffineTransform(AffineCameraTransform):
    def __init__(self, transforms: List[AffineCameraTransform]) -> None:
        """Squashes multiple affine transformations into a single transformation.

        Parameters
        ----------
        transforms: list of AffineCameraTransform
            List of transformations to be executed from right to left.
        """
        self.transforms = transforms
        super().__init__()

    def _calc_A(
        self,
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> np.ndarray:
        """Calculate transformation matrix. See AffineCameraTransform._calc_A"""
        A = np.eye(3)
        for tr in reversed(self.transforms):
            A = tr._calc_A(input_shape) @ A
            input_shape = tr._calc_shape(input_shape)
        return A

    def _calc_shape(
        self,
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> Tuple[int, int]:
        """Calculate output shape. See AffineCameraTransform._calc_shape"""
        for tr in reversed(self.transforms):
            input_shape = tr._calc_shape(input_shape)
        return input_shape


class RandomCropTransform(AffineCameraTransform):
    def __init__(
        self,
        crop_shape: Tuple[int, int],
    ) -> None:
        """Extracts random crops of crop_shape.

        Paramters
        ---------
        crop_shape: tuple
            Shape after transformation
        """
        self.shape = crop_shape
        super().__init__(shape=self.shape)

    def _calc_A(
        self,
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> np.ndarray:
        """Calculate transformation matrix. See AffineCameraTransform._calc_A"""
        # Sample a random crop of size self.shape
        h, w = input_shape[:2]
        h_target, w_target = self.shape

        xc = w / 2
        if w - w_target // 2 > w_target // 2:
            xc = np.random.randint(w_target // 2, w - w_target // 2)

        yc = h / 2
        if h - h_target // 2 > h_target // 2:
            yc = np.random.randint(h_target // 2, h - h_target // 2)

        box = [xc - w_target // 2, yc - h_target // 2, xc + w_target // 2, yc + h_target // 2]
        return box_crop_affine_transform(box, self.shape)


class RandomAffineTransform(AffineCameraTransform):
    def __init__(self, args: Optional[Dict[str, Dict[str, Union[float, bool]]]] = None) -> None:
        """Applies a random affine transformation.

        Parameters
        ----------
        args: dict of dict
            Dictionary of augmentation values. Augmentation values are dictionaries with keys 
            'center', 'low','high' and 'p'. Augementation values are sampled from a uniform
            distribution [center+low, center+high] with probabilty p, otherwise the central value is returned.
        """
        if args is None:
            args = dict()

        for k in ['theta', 'scale', 'flip', 'shiftx', 'shifty', 'shear', 'flip']:
            center = 0.0
            if k == 'scale':
                center = 1.0

            if k not in args:
                args[k] = {'p': 0, 'center': center, 'low': 0, 'high': 0}

        self.args = args
        super().__init__()

    def _calc_A(
        self,
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> np.ndarray:
        """Calculate transformation matrix. See AffineCameraTransform._calc_A"""
        def maybe_sample(p, center, low, high):
            if np.random.rand() <= p:
                return center + ((high - low) * np.random.rand() + low)
            return center

        theta_args = self.args['theta']
        theta = maybe_sample(**theta_args)

        scale_args = self.args['scale']
        scale = maybe_sample(**scale_args)

        flip_args = self.args['flip']
        flip = np.random.rand() < flip_args['p']

        shiftx_args = self.args['shiftx']
        shiftx = maybe_sample(**shiftx_args)

        shiftx_args = self.args['shiftx']
        shiftx = maybe_sample(**shiftx_args)

        shifty_args = self.args['shifty']
        shifty = maybe_sample(**shifty_args)

        shear_args = self.args['shear']
        shear = maybe_sample(**shear_args)

        return calc_affine_transform(theta, scale, flip, shiftx, shifty, shear, input_shape[:2])

    def _calc_shape(
        self,
        input_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    ) -> Tuple[int, int]:
        """Calculate output shape. See AffineCameraTransform._calc_shape"""
        # NOTE: this operation intentionally does not modify the output shape. If we zoom out, there will be black borders
        return input_shape[:2]
