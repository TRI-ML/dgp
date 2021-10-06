# Copyright 2021 Toyota Research Institute. All rights reserved.
import os
import unittest

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
from dgp.utils.visualization_utils import visualize_bev
from tests import TEST_DATA_DIR


class TestVisualization(unittest.TestCase):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")

    def setUp(self):

        # Initialize synchronized dataset
        scenes_dataset_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene", "scene_dataset_v1.0.json")
        self.dataset = SynchronizedSceneDataset(
            scenes_dataset_json,
            split='train',
            datum_names=['camera_01', 'lidar'],
            backward_context=0,
            requested_annotations=("bounding_box_2d", "bounding_box_3d")
        )

    def test_visualize_bev(self):
        # Test the bev visualization
        context = self.dataset[0]
        lidar = context[0][-1]

        ontology = self.dataset.dataset_metadata.ontology_table.get('bounding_box_3d', None)
        class_colormap = ontology._contiguous_id_colormap
        id_to_name = ontology.contiguous_id_to_name

        w = 100
        h = int(3 * w / 4)
        bev_pixels_per_meter = 10

        img = visualize_bev([lidar],
                            class_colormap,
                            show_instance_id_on_bev=False,
                            id_to_name=id_to_name,
                            bev_font_scale=.5,
                            bev_line_thickness=2,
                            bev_metric_width=w,
                            bev_metric_height=h,
                            bev_pixels_per_meter=bev_pixels_per_meter,
                            bev_center_offset_w=25)

        assert img.shape == (h * bev_pixels_per_meter, w * bev_pixels_per_meter, 3)

        # TODO: save the image and compare to a pre-generated image


if __name__ == "__main__":
    unittest.main()
