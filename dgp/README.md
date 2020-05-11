# DGP CLI

[dgp/cli.py](cli.py) is the main CLI entrypoint for handling DGP datasets.

1. **Retrieve information about an ML dataset in the DGP.**

The DGP CLI also provides information about a dataset, including
the remote location (S3 url) of the dataset, its raw dataset url, the
set of available annotation types contained in the dataset, etc. For
more information, see relevant metadata stored with a dataset artifact
in [DatasetMetadata](proto/dataset.proto) and [DatasetArtifacts](proto/artifacts.proto).
```sh
python dgp/cli.py info --scene-dataset-json <scene-dataset-json>
```

2. **SOON: Validate a dataset.**

The DGP CLI provides a simplified mechanism for validating newly
created datasets, ensuring that the dataset schema is maintained and
valid. This is done via:
```sh
python dgp/cli.py validate --scene-dataset-json <scene-dataset-json>
```