# Copyright 2021 Toyota Research Institute.  All rights reserved.
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

from dgp.constants import Vehicle
from dgp.datasets.agent_dataset import AgentDatasetLite
from dgp.utils.pose import Pose
from dgp.utils.structures.bounding_box_3d import BoundingBox3D
from dgp.utils.visualization_utils import visualize_agent_bev


def calc_warp_pose(pose_other, pose_target):
    """Transform local pose to a target pose
    Parameters
    ----------
    pose_other: Local pose
    pose_target: Target pose

    Returns
    -------
    Pose 
        Transformed pose
    """
    pose_target_w = pose_target.inverse()  # world to target
    pose_p2p1 = pose_target_w * pose_other  # local other to world, world to target -> local to target
    return pose_p2p1


def render_agent_bev(agent_dataset, ego_dimensions):
    """Render BEV of agent bounding boxes.
    Parameters
    ----------
    agent_dataset: Agent dataset 
    ego_dimensions: Ego vehicle dimensions

    Returns
    -------
    List[np.ndarray]
        Frames of BEV visualizations
    """
    ontology = agent_dataset.Agent_dataset_metadata.ontology_table.get('bounding_box_3d', None)
    class_colormap = ontology._contiguous_id_colormap
    id_to_name = ontology.contiguous_id_to_name

    tvec = np.array([ego_dimensions.vehicle_length / 2 - ego_dimensions.vehicle_applanix_origin_to_r_bumper, 0, 0])
    ego_box = BoundingBox3D(
        Pose(tvec=tvec),
        sizes=np.array([ego_dimensions.vehicle_width, ego_dimensions.vehicle_length, ego_dimensions.vehicle_height]),
        class_id=1,
        instance_id=0
    )

    # Drawing code, create a pallet
    pallet = list(sns.color_palette("hls", 32))
    pallet = [[np.int(255 * a), np.int(255 * b), np.int(255 * c)] for a, b, c in pallet]

    def get_random_color():
        idx = np.random.choice(len(pallet))
        return pallet[idx]

    trackid_to_color = {}
    paths = {}
    frames = []
    prior_pose = None
    max_path_len = 15

    for k in tqdm(range(0, len(agent_dataset))):
        context = agent_dataset[k]
        lidar = context[0]["datums"][-1]
        camera = context[0]["datums"][0]
        agents = context[0]['agents']
        cam_color = [(0, 255, 0)]
        agents.boxlist.append(ego_box)
        trackid_to_color[0] = (255, 255, 255)

        # core tracking color and path generation code
        if prior_pose is None:
            prior_pose = lidar['pose']

        warp_pose = calc_warp_pose(prior_pose, lidar['pose'])
        prior_pose = lidar['pose']

        new_colors = {box.instance_id: get_random_color() for box in agents if box.instance_id not in trackid_to_color}
        trackid_to_color.update(new_colors)
        updated = []
        # warp existing paths
        for instance_id in paths:
            # move the path into ego's local frame. We assume all prior path entrys are in the previous frame
            # this is not true if we skip a step because of occulision or missed detection... TODO: handle this somehow
            paths[instance_id] = [warp_pose * pose if pose is not None else None for pose in paths[instance_id]]

        # add new boxes to the path
        for box in agents:

            if box.instance_id not in paths:
                paths[box.instance_id] = []

            paths[box.instance_id].insert(0, box.pose)

            # keep track of what was updated so we can insert Nones if there is a miss
            updated.append(box.instance_id)

            if len(paths[box.instance_id]) > max_path_len:
                paths[box.instance_id].pop()

            box.attributes['path'] = paths[box.instance_id]

        # go through the non updated paths and append None
        for instance_id in paths:
            if instance_id not in updated:
                paths[instance_id].insert(0, None)

            if len(paths[instance_id]) > max_path_len:
                paths[instance_id].pop()

        cuboid_caption_fn = lambda x: ('Parked' if 'parked' in x.attributes else None, (255, 0, 0))

        marker_fn = lambda x: (cv2.MARKER_DIAMOND if 'parked' in x.attributes else None, (255, 0, 0))

        img = visualize_agent_bev(agents, [lidar], class_colormap, show_instance_id_on_bev=False, id_to_name = id_to_name, bev_font_scale = .5, bev_line_thickness = 2\
                        , instance_colormap = trackid_to_color, show_paths_on_bev=True,\
                        cuboid_caption_fn = cuboid_caption_fn, \
                        marker_fn = marker_fn,
                        camera_datums= [camera],
                        camera_colors = cam_color,
                        bev_metric_width=100,
                        bev_metric_height=int(3*100/4),
                        bev_pixels_per_meter = 10,
                        bev_center_offset_w=0
                        )

        frames.append(img)
    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=True
    )

    parser.add_argument('--agent-json', help='Agent JSON file')
    parser.add_argument('--scene-dataset-json', help='Scene Dataset JSON file')
    parser.add_argument(
        '--split', help='Dataset split', required=False, choices=['train', 'val', 'test'], default='train'
    )
    args, other_args = parser.parse_known_args()

    agent_dataset_frame = AgentDatasetLite(
        args.scene_dataset_json,
        args.agent_json,
        split=args.split,
        datum_names=['lidar', 'CAMERA_01'],
        requested_main_agent_classes=('Car', 'Person'),
        requested_feature_types=("parked_car", ),
        batch_per_agent=False
    )

    ego_vehicle = Vehicle("Lexus", 5.234, 1.900, 1.68, 1.164)
    bev_frames = render_agent_bev(agent_dataset_frame, ego_vehicle)

    a = [agent_dataset_frame.dataset_item_index[k][0] for k in range(len(agent_dataset_frame))]

    #Store gif per scene
    frame_num = 0
    for i in range(max(a) + 1):

        plt.figure(figsize=(20, 20))

        clip = ImageSequenceClip(bev_frames[frame_num:frame_num + a.count(i)], fps=10)
        clip.write_gif('test_scene' + str(i) + '.gif', fps=10)
        frame_num += a.count(i)
