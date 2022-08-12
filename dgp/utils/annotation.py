# Copyright 2021 Toyota Research Institute.  All rights reserved.
import locale

from dgp.utils.cache import diskcache
from dgp.utils.camera import Camera, generate_depth_map
from dgp.utils.protobuf import open_pbobject


def is_empty_annotation(annotation_file, annotation_type):
    """Check if JSON style annotation files are empty

    Parameters
    ----------
    annotation_file: str
        Path to JSON file containing annotations for 2D/3D bounding boxes

    annotation_type: object
        Protobuf pb2 object we want to load into.

    Returns
    -------
    bool:
        True if empty annotation, otherwise False
    """
    with open(annotation_file, encoding=locale.getpreferredencoding()) as _f:
        annotations = open_pbobject(annotation_file, annotation_type)
        return len(list(annotations.annotations)) == 0


@diskcache(protocol='npz')
def get_depth_from_point_cloud(dataset, scene_idx, sample_idx_in_scene, cam_datum_name, pc_datum_name):
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

    cam_datum_name: str
        Name of camera datum within the sample.

    pc_datum_name: str
        Name of the point cloud datum within the sample.

    Returns
    -------
    depth: np.ndarray
        Depth map from the camera's viewpoint.
    """
    # Get point cloud datum and load it
    pc_datum = dataset.get_datum(scene_idx, sample_idx_in_scene, pc_datum_name)
    pc_datum_type = pc_datum.datum.WhichOneof('datum_oneof')
    assert pc_datum_type == 'point_cloud', 'Depth cannot be generated from {} {} {}'.format(
        pc_datum_type, pc_datum_name, pc_datum
    )
    pc_datum_data, _ = dataset.get_point_cloud_from_datum(scene_idx, sample_idx_in_scene, pc_datum_name)
    X_W = pc_datum_data['pose'] * pc_datum_data['point_cloud']

    # Get target camera datum for projection
    cam_datum = dataset.get_datum(scene_idx, sample_idx_in_scene, cam_datum_name)
    cam_datum_type = cam_datum.datum.WhichOneof('datum_oneof')
    assert cam_datum_type == 'image', 'Depth cannot be projected onto {} '.format(cam_datum_type)
    cam_datum_data, _ = dataset.get_image_from_datum(scene_idx, sample_idx_in_scene, cam_datum_name)
    p_WC = cam_datum_data['pose']
    camera = Camera(K=cam_datum_data['intrinsics'], p_cw=p_WC.inverse())
    (W, H) = cam_datum_data['rgb'].size[:2]
    return generate_depth_map(camera, X_W, (H, W))
