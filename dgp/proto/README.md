# TRI Dataset Governance Policy (DGP) Schema

DGP-compliant datasets follow the structure and schema defined in this
directory. Top-level data containers are listed below:

## SceneDataset data structure description

- SceneDataset: A SceneDataset contains `DatasetMetadata` and a collection of
  `Scene` assigned to different `split` for training, evaluation and inference.
- Scene: A Scene consists of consecutive `Sample` extracted at a fixed frequency
  in a robot session. Normally we extract 10~20 seconds Scenes at 10Hz from raw
  driving logs.
- Sample: A sample is a container that encapsulates time-synchronized sensory
  `Datum` (images, point clouds, radar point clouds, etc) and calibrations.
- Datum: A Datum encapsulates sensory data (image, point cloud, radar point
  cloud, etc), along with their associated annotations.

## AgentDataset data structure description

- AgentDataset: Dataset for agent-centric prediction or planning use cases,but
  guaranteeing trajectory of main agent is present in any fetched sample.
- AgentSnapshot: An Agent in a Scene can be represented by either an
  AgentSnapshot2D or AgentSnapshot3D. If it is AgentSnaphot3D, then the agent is
  represented as a BoundingBox3D along with other fields such as features,
  instance id's, etc. In case it is represented as AgentSnapshot2D, then a
  BoundingBox2D is used.
- AgentSlice: Encapsulates all Agents in a Sample.
- AgentTrack: The track of a single Agent in the Scene.
- AgentGroup: Encapsulates all Agents in a Scene.
- Feature: Agent's requested feature type can include agent_2d, agent_3d,
  ego_intention, corridor, intersection or parked car.
- Ontology: An Ontology represents a set of unique objects such as Vehicle,
  Truck, Pedestrian, etc.
- Feature Ontology: A Feature Ontology represents a set of unique feature fields
  such as Speed, Parking attribute, etc.

## Protobuf Directory Structure

All the protobuf schemas are defined under this [proto/](./) directory. Schema
for the following types are specified their corresponding `proto` files.

- Agent: [`agent.proto`](./agent.proto)
- Annotations and Annotation Types: [`annotations.proto`](./annotations.proto)
- Dataset Artifacts: [`artifacts.proto`](./artifacts.proto)
- Dataset Provenance / Tracking: [`identifiers.proto`](./identifiers.proto)
- Dataset: [`dataset.proto`](./dataset.proto)
- Features: [`features.proto`](./features.proto)
- Image container: [`image.proto`](./image.proto)
- Ontology: [`ontology.proto`](./ontology.proto)
- PointCloud container: [`point_cloud.proto`](./point_cloud.proto)
- Radar PointCloud container:
  [`radar_point_cloud.proto`](./radar_point_cloud.proto)
- Remote Storage: [`remote.proto`](./remote.proto)

### SceneDataset schema graph

![SceneDataset schema](https://raw.githubusercontent.com/TRI-ML/dgp/master/docs/scene-dataset-schema.jpg?raw=true "SceneDataset schema")

### DGP SceneDataset Structure

Scenes are stored in the DGP under the following structure:

```filelist
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

### DGP AgentDataset Structure

Agents are stored in the DGP under the following structure:

```text
📦<dataset_root_dir>
 ┣ 📂<scene_name>
 ┃ ┣ 📂agent
 ┃ ┃ ┣ 📜agent_tracks_<agent_hash>.json
 ┃ ┃ ┗ 📜agents_slices_<agent_hash>.json
 ┃ ┣ 📂bounding_box_2d
 ┃ ┃ ┣ 📂<camera_name>
 ┃ ┃ ┃ ┣ 📜<annotation_hash>.json
 ┃ ┃ ┃ ┗ ..
 ┃ ┃ ┗ ..
 ┃ ┣ 📂bounding_box_3d
 ┃ ┃ ┣ 📂<camera_name>
 ┃ ┃ ┃ ┗ 📜<annotation_hash>.json
 ┃ ┃ ┃ ┗ ..
 ┃ ┃ ┗ ..
 ┃ ┃ ┗ 📂<lidar_name>
 ┃ ┃ ┃ ┗ 📜<annotation_hash>.json
 ┃ ┃ ┃ ┗ ..
 ┃ ┣ 📂calibration
 ┃ ┃ ┗ 📜<calibration_hash>.json
 ┃ ┣ 📂feature_ontology
 ┃ ┃ ┗ 📜<feature_ontology_hash>.json
 ┃ ┣ 📂ontology
 ┃ ┃ ┗ 📜<ontology_hash>.json
 ┃ ┣ 📂point_cloud
 ┃ ┃ ┗ 📂<lidar_name>
 ┃ ┃ ┃ ┣ 📜<posix_timestamp_us>.npz
 ┃ ┃ ┃ ┗ ..
 ┃ ┣ 📂rgb
 ┃ ┃ ┣ 📂<camera_name>
 ┃ ┃ ┃ ┣ 📜<posix_timestamp_us>.jpg
 ┃ ┃ ┃ ┗ ..
 ┃ ┃ ┗ ..
 ┃ ┣ 📜agents_<agent_hash>.json
 ┃ ┗ 📜scene_<scene_hash>.json
 ┣ 📜agents_v<version>.json
 ┗ 📜scene_dataset_v<version>.json
```

### Autolabel Structure

Autolabes should be stored within their parent scene directory under
`<parent scene dir>/autolabels/<autolabel model>` folder. Autolabels may also be
stored outside of the parent scene folder, in this case they must be stored with
the following directory structure:
`<autolabel root>/<parent scene dir basename>/autolabels/<autolabel model>`.

In both cases `<autolabel model>` should be unique a string denoting which model
or process generated the autolabels.
