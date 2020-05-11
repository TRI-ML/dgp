[<img src="/docs/tri-logo.jpeg" width="25%">](https://www.tri.global/)

TRI Dataset Governance Policy
==========
To ensure the traceability, reproducibility and standardization for
all ML datasets and models generated and consumed within TRI, we developed the
Dataset-Governance-Policy (DGP) that codifies the schema and
maintenance of all TRI's Autonomous Vehicle (AV) datasets.


## Components
- [Schema](dgp/proto/README.md): [Protobuf](https://developers.google.com/protocol-buffers)-based schemas for raw data, annotations
  and dataset management.
- [DataLoaders](dgp/datasets): Universal PyTorch DatasetClass to load all DGP-compliant datasets.
- [Visualizer](dgp/scripts/visualizer.py): Simple web-based visualizer for viewing annotations.
- [CLI](dgp/README.md): Main CLI for handling DGP datasets.

## Getting Started
Getting started is as simple as initializing a dataset-class with the
relevant dataset JSON, raw data sensor names, annotation types, and
split information. Below, we show a few examples of initializing a
Pytorch dataset for multi-modal learning from 2D bounding boxes, and
3D bounding boxes.
```python
from dgp.datasets import SynchronizedSceneDataset

# Load synchronized pairs of camera and lidar frames, with 2d and 3d
# bounding box annotations.
dataset = SynchronizedSceneDataset('<dataset_name>_v0.0.json',
    datum_names=('camera_01', 'lidar'),
    requested_annotations=('bounding_box_2d', 'bounding_box_3d'),
    split='train')
```

## Examples
A list of starter scripts are provided in the [examples](examples/)
directory.
- [examples/load_dataset.py](examples/load_dataset.py): Simple example
  script to load a multi-modal dataset based on the **Getting
  Started** section above.

## Build and run tests
You can build the base docker image and run the tests within docker
via:
```sh
make docker-build
make docker-run-tests
```

## Run Visualizer
Run streamlit-based interactive visualizer
via:
```sh
make docker-start-visualizer
```
