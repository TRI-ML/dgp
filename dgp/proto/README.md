TRI Dataset Governance Policy (DGP) Schema
==========
DGP-compliant datasets follow the structure and schema defined in this directory. Top-level
data containers are listed below:

* SceneDataset: A SceneDataset contains `DatasetMetadata` and a collection of `Scene`
assigned to different `split` for training, evaluation and inference.
* Scene: A Scene consists of consecutive `Sample` extracted at a fixed frequency in
a robot session. Normally we extract 10~20 seconds Scenes at 10Hz from raw driving logs.
* Sample: A sample is a container that encapsulates time-synchronized sensory `Datum`
(images, point clouds, GPS/IMU etc) and calibrations.
* Datum: A Datum encapsulates sensory data (image, point cloud, GPS/IMU etc),
along with their associated annotations.

## Protobuf Directory Structure

All the protobuf schemas are defined under this [proto/](./)
directory. Schema for the following types are specified their
corresponding `proto` files.
* Annotations and Annotation Types: [`annotations.proto`](./annotations.proto)
* Dataset Artifacts: [`artifacts.proto`](./artifacts.proto)
* Dataset Provenance / Tracking: [`identifiers.proto`](./identifiers.proto)
* Dataset: [`dataset.proto`](./dataset.proto)
* Image container: [`image.proto`](./image.proto)
* PointCloud container: [`point_cloud.proto`](./point_cloud.proto)
* Remote Storage: [`remote.proto`](./remote.proto)

### SceneDataset schema graph
![SceneDataset schema](../../docs/scene-dataset-schema.jpg?raw=true "SceneDataset schema")

## DGP SceneDataset Structure
Scenes are stored in the DGP under the following structure:
```
<dataset_root_dir>
├── <scene_name>
│   ├── point_cloud // datum
│   │   └── <lidar_name>
│   │       ├── <posix_timestamp_us>.npz
│   │       └── ...
│   │   └── ...
│   ├── rgb // datum
│   │   └── <camera_name>
│   │       ├── <posix_timestamp_us>.png
│   │       └── ...
│   │   └── ...
│   ├── bounding_box_2d // annotation
│   │   └── <camera_name>
│   │       ├── <annotation_hash>.json
│   │       └── ...
│   │   └── ...
│   ├── bounding_box_3d // annotation
│   │   └── <lidar_name>
│   │       ├── <annotation_hash>.json
│   │       └── ...
│   │   └── ...
│   ├── calibration // calibration
│   │   └── <calibration_hash>.json
│   │   └── ...
│   ├── ontology // ontology
│   │   └── <ontology_hash>.json
|   └── ...
│   └── scene_<scene_hash>.json
├── <scene_name>
└── ...
└── scene_dataset_v<version>.json
```