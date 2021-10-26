import os
import unittest

from dgp.datasets.agent_dataset import AgentDatasetLite
from tests import TEST_DATA_DIR


class TestAgentDataset(unittest.TestCase):
    DGP_TEST_DATASET_DIR = os.path.join(TEST_DATA_DIR, "dgp")

    def setUp(self):
        self.test_scene_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene/scene_dataset_v1.0.json")

        self.agent_json = os.path.join(self.DGP_TEST_DATASET_DIR, "test_scene/agents_pcc_mini_v1.json")

    def test_prediction_agent_dataset_3d(self):
        #Test agent dataset loading

        dataset = AgentDatasetLite(
            self.test_scene_json,
            self.agent_json,
            split='train',
            datum_names=None,
            requested_main_agent_classes=('Car', 'Person'),
            requested_feature_types=("parked_car", ),
            batch_per_agent=True
        )
        assert len(dataset) == 110


if __name__ == "__main__":
    unittest.main()
