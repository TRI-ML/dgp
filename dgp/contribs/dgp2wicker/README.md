# DGP SynchronizedScene to Wicker Conversion

This adds support for using DGP data in
[wicker](https://github.com/woven-planet/wicker)

Specifically this saves the output of SynchronizedScene to wicker

---

### Install

```bash
cd dgp/contribs/dgp2wicker
pip install --editable .
```

or, use the included docker. Note: the s3 location of the wicker datasets is
specified in a required wicker config file, please see Wicker documentaiton for
more details. An example sample_wickerconfig.json is included in the docker,
this can be modified with the s3 bucket path and will work with the docker.

```bash
cd dgp/contribs/dgp2wicker
make docker-build
```

### Example

#### Save dataset to wicker

```bash
$dgp2wicker ingest \
--scene-dataset-json <path to scene dataset json in s3 or local> \
--wicker-dataset-name test_dataset \
--wicker-dataset-version 0.0.1 \
--datum-names camera_01,camera_02,lidar \
--requested-annotations bounding_box_3d,semantic_segmentation_2d \
--only-annotated-datums
```

#### Read dataset from wicker

```python
from dgp2wicker.dataset import DGPS3Dataset, compute_columns

columns = compute_columns(datum_names = ['camera_01','camera_02','lidar',],\
                          datum_types = ['image','image','point_cloud',], \
                          requested_annotations=['bounding_box_3d','semantic_segmentation_2d','depth',], \
                          cuboid_datum = 'lidar',)

dataset = DGPS3Dataset(dataset_name = 'test_dataset',\
                       dataset_version = '0.0.1', \
                       dataset_partition_name='train', \
                       columns_to_load = columns,)

context = dataset[0]
```

---

### Supported datums/annotations

datums:

- [x] image
- [x] point_cloud
- [x] radar_point_cloud
- [ ] file_datum
- [ ] agent

annotations:

- [x] bounding_box_2d
- [x] bounding_box_3d
- [x] depth
- [x] semantic_segmentation_2d
- [x] instance_segmentation_2d
- [ ] key_point_2d
- [ ] key_line_2d
