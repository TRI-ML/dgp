import os
import unittest

import numpy as np

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.utils.accumulate import accumulate_points, points_in_cuboid
from dgp.utils.pose import Pose
from dgp.utils.structures.bounding_box_3d import BoundingBox3D
from tests import TEST_DATA_DIR


class TestAccumulation(unittest.TestCase):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")

    def test_accumulation_in_scene(self):

        # Load a scene without accumulation
        scenes_dataset_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
        dataset = SynchronizedSceneDataset(
            scenes_dataset_json,
            split='train',
            datum_names=['lidar'],
        )

        # Load the same scene with max context available accumulation context, this dataset has two scenes each with 3 samples
        dataset_acc = SynchronizedSceneDataset(
            scenes_dataset_json,
            split='train',
            datum_names=['lidar'],
            requested_annotations=['bounding_box_3d'],
            accumulation_context={'lidar': (2, 0)},
            transform_accumulated_box_points=True
        )

        # We should only have two samples (one per scene)
        assert len(dataset_acc) == 2

        # Verify that we have not lost any points by accumulating
        num_points = 0
        for i in range(3):
            context = dataset[i]
            num_points += len(context[0][-1]['point_cloud'])

        context_acc = dataset_acc[0]
        num_points_acc = len(context_acc[0][-1]['point_cloud'])

        assert num_points == num_points_acc

    def test_accumulation(self):
        """Test accumulation"""

        # Generate some samples
        scenes_dataset_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
        dataset = SynchronizedSceneDataset(
            scenes_dataset_json,
            split='train',
            datum_names=['lidar'],
        )

        assert len(dataset) >= 2

        point_datums = []
        for sample in dataset:
            point_datums.append(sample[0][0])

        p1, p2 = point_datums[0], point_datums[-1]

        p1_and_p2_in_p1 = accumulate_points([p1, p2], p1)
        assert len(p1_and_p2_in_p1['point_cloud']) == len(p1['point_cloud']) + len(p2['point_cloud'])

        p1_and_p2_in_p2 = accumulate_points([p1, p2], p2)
        assert len(p1_and_p2_in_p2['point_cloud']) == len(p1['point_cloud']) + len(p2['point_cloud'])

        # If we move the accumulated p1 frame points to p2, we should recover the accumulated p2 points
        p1_and_p2_in_p2_v2 = accumulate_points([p1_and_p2_in_p1], p2)
        assert np.allclose(p1_and_p2_in_p2_v2['point_cloud'], p1_and_p2_in_p2['point_cloud'])

        # If we accumulate a single point nothing should happen
        p1_v2 = accumulate_points([p1], p1)
        assert np.allclose(p1_v2['point_cloud'], p1['point_cloud'])

    def test_point_in_cuboid(self):
        """Test point in cuboid"""

        cuboid = BoundingBox3D(Pose(), sizes=np.float32([1, 2, 3]))
        query_points = np.mean(cuboid.corners, 0, keepdims=True)

        in_view = points_in_cuboid(query_points, cuboid)
        # Cuboid center should be inside the cube
        assert np.all(in_view)

        query_points = query_points + 5
        in_view = points_in_cuboid(query_points, cuboid)
        # This should not be in the cuboid
        assert not np.any(in_view)


if __name__ == "__main__":
    unittest.main()
