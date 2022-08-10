# Copyright 2021 Toyota Research Institute.  All rights reserved.
import os

import numpy as np

from dgp.annotations import Annotation
from dgp.utils.dataset_conversion import generate_uid_from_point_cloud


class DenseDepthAnnotation(Annotation):
    """Container for per-pixel depth annotation.

    Parameters
    ----------
    depth: np.ndarray
        2D numpy float array that stores per-pixel depth.
    """
    def __init__(self, depth):
        assert isinstance(depth, np.ndarray)
        assert depth.dtype in [np.float32, np.float64]
        super().__init__(None)
        self._depth = depth

    @property
    def depth(self):
        return self._depth

    @classmethod
    def load(cls, annotation_file, ontology=None):
        """Loads annotation from file into a canonical format for consumption in __getitem__ function in BaseDataset.

        Parameters
        ----------
        annotation_file: str
            Full path to NPZ file that stores 2D depth array.

        ontology: None
            Dummy ontology argument to meet the usage in `BaseDataset.load_annotation()`.
        """
        assert ontology is None, "'ontology' must be 'None' for {}.".format(cls.__name__)
        depth = np.load(annotation_file)['data']
        return cls(depth)

    @property
    def hexdigest(self):
        return generate_uid_from_point_cloud(self.depth)

    def save(self, save_dir):
        """Serialize annotation object if possible, and saved to specified directory.
        Annotations are saved in format <save_dir>/<sha>.<ext>

        Parameters
        ----------
        save_dir: str
            Path to directory to saved annotation

        Returns
        -------
        pointcloud_path: str
            Full path to the output NPZ file.
        """
        pointcloud_path = os.path.join(save_dir, '{}.npz'.format(self.hexdigest))
        np.savez_compressed(pointcloud_path, data=self.depth)
        return pointcloud_path

    def render(self):
        """TODO: Rendering function for per-pixel depth."""
