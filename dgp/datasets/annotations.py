import json
import os
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

from dgp.datasets.cache import diskcache
from dgp.proto.annotations_pb2 import (BoundingBox2DAnnotations,
                                       BoundingBox3DAnnotations)
from dgp.utils.camera import Camera, generate_depth_map
from dgp.utils.geometry import BoundingBox3D, Pose
from dgp.utils.protobuf import open_pbobject


def is_empty_annotation(annotation_file, annotation_type):
    """Check if JSON style annotation files are empty

    Parameters
    ----------
    annotations: str
        Path to JSON file containing annotations for 2D/3D bounding boxes

    Returns
    -------
    bool:
        True if empty annotation, otherwise False
    """
    with open(annotation_file) as _f:
        annotations = open_pbobject(annotation_file, annotation_type)
        return len(list(annotations.annotations)) == 0


#-------------------------- 3D Bounding Box Annotations ----------------------------------------#


def load_aligned_bounding_box_annotations(annotations, annotations_dir, json_category_id_to_contiguous_id):
    """Load 2D/3D bounding box annotations as an OrderedDict. An annotation is considered
    aligned if there exists both a 2D and 3D bounding box for a given instance ID.

    Parameters
    ----------
    annotations: dict
        Dictionary mapping annotation keys to annotation files for corresponding datum.

    annotations_dir: str
        Path to the annotations directory of the datum to be queried.

    json_category_id_to_contiguous_id: dict
        Lookup from COCO style JSON id's to contiguous id's

    Returns
    -------
    data: OrderedDict

        "bounding_box_2d": np.ndarray dtype=np.float32
            Tensor containing bounding boxes for this sample
            (x, y, w, h) in absolute pixel coordinates

        "bounding_box_3d": list of BoundingBox3D
            3D Bounding boxes for this sample specified in this point cloud
            sensor's reference frame. (i.e. this provides the bounding box
            (B) in the sensor's (S) reference frame `box_SB`).

        "class_ids": np.ndarray dtype=np.int64
            Tensor containing class ids (aligned with ``bounding_box_3d``)

        "instance_ids": np.ndarray dtype=np.int64
            Tensor containing instance ids (aligned with ``bounding_box_3d``)
    """
    _, _, id_to_box_2d, _ = parse_annotations_2d_proto(
        os.path.join(annotations_dir, annotations["bounding_box_2d"]), json_category_id_to_contiguous_id
    )
    _, _, id_to_box_3d = parse_annotations_3d_proto(
        os.path.join(annotations_dir, annotations["bounding_box_3d"]), json_category_id_to_contiguous_id
    )

    # Extract 2D and 3D annotations with class and instance ids.
    # Align 2D and 3D data
    boxes_2d, boxes_3d, class_ids, instance_ids = [], [], [], []
    for instance_id in id_to_box_2d:
        if instance_id not in id_to_box_3d:
            continue
        box_2d, class_id_2d = id_to_box_2d[instance_id]
        box_3d, class_id_3d = id_to_box_3d[instance_id]
        assert class_id_2d == class_id_3d, "Misaligned annotations between 2D and 3D"
        boxes_2d.append(box_2d)
        boxes_3d.append(box_3d)
        class_ids.append(class_id_2d)
        instance_ids.append(instance_id)
    return OrderedDict({
        "bounding_box_2d": np.float32(boxes_2d),
        "bounding_box_3d": boxes_3d,
        "class_ids": np.int64(class_ids),
        "instance_ids": np.int64(instance_ids)
    })


def load_bounding_box_3d_annotations(annotations, annotations_dir, json_category_id_to_contiguous_id):
    """Load 3D bounding box annotations as an OrderedDict.

    Parameters
    ----------
    annotations: dict
        Dictionary mapping annotation keys to annotation files for corresponding datum.

    annotations_dir: str
        Path to the annotations directory of the datum to be queried.

    json_category_id_to_contiguous_id: dict
        Lookup from COCO style JSON id's to contiguous id's

    Returns
    -------
    data: OrderedDict

        "bounding_box_3d": list of BoundingBox3D
            3D Bounding boxes for this sample specified in this point cloud
            sensor's reference frame. (i.e. this provides the bounding box
            (B) in the sensor's (S) reference frame `box_SB`).

        "class_ids": np.ndarray dtype=np.int64
            Tensor containing class ids (aligned with ``bounding_box_3d``)

        "instance_ids": np.ndarray dtype=np.int64
            Tensor containing instance ids (aligned with ``bounding_box_3d``)
    """
    _, _, id_to_box_3d = parse_annotations_3d_proto(
        os.path.join(annotations_dir, annotations["bounding_box_3d"]), json_category_id_to_contiguous_id
    )

    # Extract 3D annotations with class and instance ids
    boxes_3d, class_ids, instance_ids = [], [], []
    for instance_id in id_to_box_3d:
        box_3d, class_id_3d = id_to_box_3d[instance_id]
        boxes_3d.append(box_3d)
        class_ids.append(class_id_3d)
        instance_ids.append(instance_id)
    return OrderedDict({
        "bounding_box_3d": boxes_3d,
        "class_ids": np.int64(class_ids),
        "instance_ids": np.int64(instance_ids)
    })


def parse_annotations_3d_proto(annotation_file, json_category_id_to_contiguous_id):
    """Parse annotations from BoundingBox2DAnnotations structure.

    Parameters
    ----------
    annotations: str
        Path to JSON file containing annotations for 2D bounding boxes

    json_category_id_to_contiguous_id: dict
        Lookup from COCO style JSON id's to contiguous id's

    transformation: Pose
        Pose object that can be used to convert annotations to a new reference frame.

    Returns
    -------
    tuple holding:
        boxes: list of BoundingBox3D
            Tensor containing bounding boxes for this sample
            (pose.quat.qw, pose.quat.qx, pose.quat.qy, pose.quat.qz,
            pose.tvec.x, pose.tvec.y, pose.tvec.z, width, length, height)
            in absolute scale

        class_ids: np.int64 array
            Numpy array containing class ids (aligned with ``boxes``)

        instance_ids: dict
            Map from instance_id to tuple of (box, class_id)
    """

    # *CAVEAT*: `attributes` field is defined in proto, but not being read here.
    # TODO: read attributes (see above); change outputs of all function calls.

    with open(annotation_file) as _f:
        annotations = open_pbobject(annotation_file, BoundingBox3DAnnotations)
        boxes, class_ids, instance_ids = [], [], {}
        for i, ann in enumerate(list(annotations.annotations)):
            boxes.append(
                BoundingBox3D(
                    Pose.from_pose_proto(ann.box.pose), np.float32([ann.box.width, ann.box.length, ann.box.height]),
                    ann.num_points, ann.box.occlusion, ann.box.truncation
                )
            )
            class_ids.append(json_category_id_to_contiguous_id[ann.class_id])
            instance_ids[ann.instance_id] = (boxes[i], class_ids[i])
        return boxes, class_ids, instance_ids


#-------------------------- 2D Bounding Box Annotations ----------------------------------------#


def load_bounding_box_2d_annotations(annotations, annotations_dir, json_category_id_to_contiguous_id):
    """Load 2D bounding box annotations as an OrderedDict.

    Parameters
    ----------
    annotations: dict
        Dictionary mapping annotation keys to annotation files for corresponding datum.

    annotations_dir: str
        Path to the annotations directory of the datum to be queried.

    json_category_id_to_contiguous_id: dict
        Lookup from COCO style JSON id's to contiguous id's

    Returns
    -------
    data: OrderedDict

        "bounding_box_2d": list of BoundingBox3D
            3D Bounding boxes for this sample specified in this point cloud
            sensor's reference frame. (i.e. this provides the bounding box
            (B) in the sensor's (S) reference frame `box_SB`).

        "class_ids": np.ndarray dtype=np.int64
            Tensor containing class ids (aligned with ``bounding_box_3d``)

        "instance_ids": np.ndarray dtype=np.int64
            Tensor containing instance ids (aligned with ``bounding_box_3d``)
    """
    _, _, id_to_box_2d, _ = parse_annotations_2d_proto(
        os.path.join(annotations_dir, annotations["bounding_box_2d"]), json_category_id_to_contiguous_id
    )

    # Extract 3D annotations with class and instance ids
    boxes_2d, class_ids, instance_ids = [], [], []
    for instance_id in id_to_box_2d:
        box_2d, class_id_2d = id_to_box_2d[instance_id]
        boxes_2d.append(box_2d)
        class_ids.append(class_id_2d)
        instance_ids.append(instance_id)
    return OrderedDict({
        "bounding_box_2d": np.float32(boxes_2d),
        "class_ids": np.int64(class_ids),
        "instance_ids": np.int64(instance_ids)
    })


def parse_annotations_2d_proto(annotation_file, json_category_id_to_contiguous_id):
    """Parse annotations from BoundingBox2DAnnotations structure.

    Parameters
    ----------
    annotations: str
        Path to JSON file containing annotations for 2D bounding boxes

    json_category_id_to_contiguous_id: dict
        Lookup from COCO style JSON id's to contiguous id's

    Returns
    -------
    tuple holding:
        boxes: torch.FloatTensor
            Tensor containing bounding boxes for this sample
            (x, y, w, h) in absolute pixel coordinates

        class_ids: np.int64 array
            Numpy array containing class ids (aligned with ``boxes``)

        instance_ids: dict
            Map from instance_id to tuple of (box, class_id)

        attributes: list
            list of dict mapping attribute names to values.
    """
    with open(annotation_file) as _f:
        annotations = open_pbobject(annotation_file, BoundingBox2DAnnotations)
        boxes, class_ids, instance_ids, attributes = [], [], {}, []
        for i, ann in enumerate(list(annotations.annotations)):
            boxes.append(np.float32([ann.box.x, ann.box.y, ann.box.w, ann.box.h]))
            class_ids.append(json_category_id_to_contiguous_id[ann.class_id])
            instance_ids[ann.instance_id] = (boxes[i], class_ids[i])
            attributes.append(getattr(ann, 'attributes', {}))
        return np.float32(boxes), np.int64(class_ids), instance_ids, attributes


#-------------------------- Semantic Segmentation Annotations ----------------------------------------#


def load_semantic_segmentation_2d_annotations(annotations, annotations_dir, label_lookup_table, ignore_id):
    segmentation_label = parse_semantic_segmentation_2d_proto(
        os.path.join(annotations_dir, annotations["semantic_segmentation_2d"]), label_lookup_table, ignore_id
    )
    return OrderedDict({
        "semantic_segmentation_2d": segmentation_label
    })


def parse_semantic_segmentation_2d_proto(annotation_file, label_lookup_table, ignore_id):
    """Parse semantic segmentation 2d annotations from annotation file.

    Parameters
    ----------
    annotations: str
        Path to PNG file containing annotations for 2D semantic segmentation

    label_lookup_table: np.array
        A lookup table converting raw label into continuous training ids.

    ignore_id: int
        pixels labeled with "ignore_id" will be ignored during training and evaluation.

    Returns
    -------
    segmentation_label: np.array
        Dense 2D semantic segmentation label
    """
    segmentation_label = Image.open(annotation_file)
    segmentation_label = np.array(segmentation_label, dtype=np.uint8)

    # IGNORE_ID is a special case not remapped not part of the classes
    not_ignore = segmentation_label != ignore_id
    segmentation_label[not_ignore] = label_lookup_table[segmentation_label[not_ignore]]
    return segmentation_label


#-------------------------- Panoptic Segmentation 2D Annotation ----------------------#


def load_panoptic_segmentation_2d_annotations(annotations, annotations_dir, name_to_contiguous_id):
    instance_masks, class_names, instance_ids = parse_panoptic_segmentation_2d_proto(
        os.path.join(annotations_dir, annotations["instance_segmentation_2d"])
    )
    class_ids = [name_to_contiguous_id[class_name] for class_name in class_names]
    return OrderedDict({
        "panoptic_instance_masks": instance_masks,
        "panoptic_class_names": class_names,
        "panoptic_instance_ids": np.int64(instance_ids),
        "panoptic_class_ids": np.int64(class_ids)
    })

def parse_panoptic_segmentation_2d_proto(annotation_file):
    """Parse panoptic segmentation 2d annotations from file .

    Parameters
    ----------
    annotation_file: str
        Full path to panoptic image. `index_to_label` JSON is expected to live at the same path with '.json' ending

    Returns
    -------
    tuple holding:
        instance_masks: List[np.bool]
            (H, W) bool array for each instance in panoptic annotation

        class_names: List[str]
            Class name for each instance in panoptic annotation

        instance_ids: List[int]
            Instance IDs for each instance in panoptic annotation
    """
    panoptic_image = cv2.imread(annotation_file, cv2.IMREAD_UNCHANGED)
    with open('{}.json'.format(os.path.splitext(annotation_file)[0])) as _f:
        index_to_label = json.load(_f)

    instance_masks, class_names, instance_ids = [], [], []
    for class_name, labels in index_to_label.items():
        if isinstance(labels, list):
            for label in labels:
                instance_id = label['index']
                if instance_id <= 0:
                    raise ValueError('`index` field of a thing class is expected to be non-negative')

                # Mask for pixels belonging to this instance
                instance_masks.append(panoptic_image == instance_id)
                class_names.append(class_name)
                instance_ids.append(instance_id)
    return instance_masks, class_names, instance_ids
#-------------------------- Depth Annotations ----------------------------------------#


@diskcache(protocol='npz')
def get_depth_from_point_cloud(
    dataset, scene_idx, sample_idx_in_scene, cam_datum_idx_in_sample, pc_datum_idx_in_sample
):
    """Generate the depth map in the camera view using the provided point cloud
    datum within the sample.

    Parameters
    ----------
    dataset: dgp.dataset.BaseDataset
        Inherited base dataset to augment with depth data.

    scene_idx: int
        Index of the scene.

    sample_idx_in_scene: int
        Index of the sample within the scene at scene_idx.

    cam_datum_idx_in_sample: int
        Index of the camera datum within the sample.

    pc_datum_idx_in_sample: int
        Index of the point cloud datum within the sample.

    Returns
    -------
    depth: np.ndarray
        Depth map from the camera's viewpoint.
    """
    # Get point cloud datum and load it
    pc_datum = dataset.get_datum(scene_idx, sample_idx_in_scene, pc_datum_idx_in_sample)
    pc_datum_type = pc_datum.datum.WhichOneof('datum_oneof')
    assert pc_datum_type == 'point_cloud', 'Depth cannot be generated from {} '.format(pc_datum_type)
    pc_datum_data = dataset.get_point_cloud_from_datum(scene_idx, sample_idx_in_scene, pc_datum_idx_in_sample)
    X_W = pc_datum_data['pose'] * pc_datum_data['point_cloud']
    # Get target camera datum for projection
    cam_datum = dataset.get_datum(scene_idx, sample_idx_in_scene, cam_datum_idx_in_sample)
    cam_datum_type = cam_datum.datum.WhichOneof('datum_oneof')
    assert cam_datum_type == 'image', 'Depth cannot be projected onto {} '.format(cam_datum_type)
    cam_datum_data = dataset.get_image_from_datum(scene_idx, sample_idx_in_scene, cam_datum_idx_in_sample)
    p_WC = cam_datum_data['pose']
    camera = Camera(K=cam_datum_data['intrinsics'], p_cw=p_WC.inverse())
    (W, H) = cam_datum_data['rgb'].size[:2]
    return generate_depth_map(camera, X_W, (H, W))
