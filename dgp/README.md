# DGP Command-Line Interface

[dgp/cli.py](cli.py) is the main CLI entrypoint for handling DGP datasets.

## Visualize DGP-compliant Scene and SceneDataset

DGP CLI subcommands `visualize-scenes` and `visualize-scene` can be used to
visualize DGP-compliant data.

- To visualize a split of a **[DGP SceneDataset](proto/dataset.proto#L127)**,
  run `python dgp/cli.py visualize-scenes`:

  Show the help message via:

  ```sh
  dgp$ python dgp/cli.py visualize-scenes --help
  ```

  Example command to visualize the images from `CAMERA_01, CAMERA_05, CAMERA_06`
  and point cloud from `LIDAR` along with ground_truth annotations
  `bounding_box_2d, bounding_box_3d` from `train` split of the toy dataset
  `tests/data/dgp/test_scene/scene_dataset_v1.0.json`, and store the resulting
  videos in `--dst-dir vis`. One can find the resulting 3D visualization videos
  in `vis/3d` and 2D visualization videos in `vis/2d`.

  ```sh
  dgp$ python dgp/cli.py visualize-scenes --scene-dataset-json tests/data/dgp/test_scene/scene_dataset_v1.0.json --split train --dst-dir vis -l LIDAR -c CAMERA_01 -c CAMERA_05 -c CAMERA_06 -a bounding_box_2d -a bounding_box_3d
  ```

  <p align="center">
    <img src="https://raw.githubusercontent.com/TRI-ML/dgp/master/docs/3d-viz.gif" alt="3d-viz"/>
  </p>

Add flag `render-pointcloud` to render projected pointcloud onto images:

```sh
dgp$ python dgp/cli.py visualize-scenes --scene-dataset-json tests/data/dgp/test_scene/scene_dataset_v1.0.json --split train --dst-dir vis -l LIDAR -c CAMERA_01 -c CAMERA_05 -c CAMERA_06 -a bounding_box_2d -a bounding_box_3d --render-pointcloud
```

<p align="center">
  <img src="https://raw.githubusercontent.com/TRI-ML/dgp/master/docs/3d-viz-proj.gif" alt="3d-viz-proj"/>
</p>

- To visualize a single **[DGP Scene](proto/scene.proto#L14)**, run
  `python dgp/cli.py visualize-scene`:

  Show the help message via:

  ```sh
  dgp$ python dgp/cli.py visualize-scene --help
  ```

  Example command to visualize the images from `CAMERA_01, CAMERA_05, CAMERA_06`
  and point cloud from `LIDAR` along with ground_truth annotations
  `bounding_box_2d, bounding_box_3d` from the toy Scene
  `tests/data/dgp/test_scene/scene_01/scene_a8dc5ed1da0923563f85ea129f0e0a83e7fe1867.json`,
  and store the resulting videos in `--dst-dir vis`. One can find the resulting
  3D visualization video in `vis/3d` and 2D visualization video in `vis/2d`.

  ```sh
  dgp$ python dgp/cli.py visualize-scene --scene-json tests/data/dgp/test_scene/scene_01/scene_a8dc5ed1da0923563f85ea129f0e0a83e7fe1867.json --dst-dir vis -l LIDAR -c CAMERA_01 -c CAMERA_05 -c CAMERA_06 -a bounding_box_2d -a bounding_box_3d
  ```

## Coming soon: Retrieve information about an ML dataset in the DGP

DGP CLI provides information about a dataset, including the remote location (S3
url) of the dataset, its raw dataset url, the set of available annotation types
contained in the dataset, etc. For more information, see relevant metadata
stored with a dataset artifact in [DatasetMetadata](proto/dataset.proto) and
[DatasetArtifacts](proto/artifacts.proto).

```sh
dgp$ python dgp/cli.py info --scene-dataset-json <scene-dataset-json>
```

## Coming soon: Validate a dataset

DGP CLI provides a simplified mechanism for validating newly created datasets,
ensuring that the dataset schema is maintained and valid. This is done via:

```sh
dgp$ python dgp/cli.py validate --scene-dataset-json <scene-dataset-json>
```
