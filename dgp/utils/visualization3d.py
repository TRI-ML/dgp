# Copyright 2019 Toyota Research Institute.  All rights reserved.
"""3D Visualization tool.

Note: This file needs to be standalone until the known torch / open3d bug is
resolved.
"""
import numpy as np
import open3d as o3d


class Viz3d:
    """3D Visualization class for rendering points clouds and annotations.

    Parameters
    ----------
    headless: bool, default: False
        Run in headless mode.
    """
    def __init__(self, headless=False):
        """Initialize Viz3d class for rendering purposes.

        Parameters
        ----------
        headless: bool, default: False
            Run in headless mode.
        """
        assert headless is False, 'Headless mode is not yet supported.'
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window()
        self._items = []

    def set_camera(self, camera):
        """Set the renderer's view point and intrinsics.

        Note: Setting the intrinsics does not seem to work just yet, we need
        this to establish the correct field of view for the renderer.
        param.intrinsic.set_intrinsics(param.intrinsic.width, param.intrinsic.height,
                                       camera.fx, camera.fy,
                                       camera.cx, camera.cy)

        Parameters
        ----------
        camera: Camera
            Camera object whose intrinsics and extrinsics are used to set the
            renderer's viewport.
        """
        ctr = self._vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = camera.p_cw.inverse().matrix
        ctr.convert_from_pinhole_camera_parameters(param)

    def clear(self):
        """Remove all geometries from the viewer."""
        for item in self._items:
            self._vis.remove_geometry(item)

    @property
    def image_buffer(self):
        """Retrieve the viewer buffer as an image.

        Returns
        ----------
        buffer: np.ndarray
            Rendered image buffer (RGB order).
        """
        im_buffer = self._vis.capture_screen_float_buffer(do_render=True)
        return (np.asarray(im_buffer) * 255).astype(np.uint8)

    def draw_point_cloud(self, X, color=None):
        """Draw 3D colored point cloud.

        Parameters
        ----------
        X: np.ndarray
            Point cloud to render (N x 3)

        color: np.ndarray, default: None
            Point cloud color to be rendered with (N x 3)
        """
        assert X.shape[1] == 3, 'Point cloud needs to be in N x 3 format'
        if color is None:
            color = np.tile(np.float32([[0, 1, 0]]), (len(X), 1))
        assert X.shape == color.shape, 'Colors need to be in N x 3 format'
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(X)
        pc.colors = o3d.utility.Vector3dVector(color)
        self._vis.add_geometry(pc)
        self._items.append(pc)

    def draw_bounding_box_3d(self, bbox3d, color=None):
        """Draw 3D bounding box.

        Parameters
        ----------
        bbox3d: BoundingBox3D
            3D Bounding Box to be rendered.

        color: np.ndarray, default: None
            Bounding box color.
        """
        edges = bbox3d.edges
        if color is None:
            color = np.tile(np.float32([[0, 1, 0]]), (len(edges), 1))
        assert len(edges) == len(color), 'Colors need to be in N x 3 format'
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox3d.corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.colors = o3d.utility.Vector3dVector(color)
        self._vis.add_geometry(line_set)
        self._items.append(line_set)

    def render(self):
        """Main blocking rendering call."""
        self._vis.run()
        self._vis.destroy_window()
