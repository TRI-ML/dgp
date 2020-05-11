# Copyright 2019 Toyota Research Institute.  All rights reserved.
"""streamlit-based visualizer. Run `streamlit run visualizer.py` to start the visualizer."""

import glob
import os

import numpy as np
import streamlit as st
from dgp.datasets.base_dataset import DatasetMetadata
from dgp.datasets.pd_dataset import ParallelDomainSceneDataset
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset
# Required for Parallel Domain scene metadata
import dgp.contribs.pd.metadata_pb2 # pylint: disable=unused-import
from dgp.scripts.visualize_dataset import (render_bev,
                                           render_pointcloud_and_box_onto_rgb)


def display_dataset_info(dataset, dataset_path):
    """Display dataset information on the sidebar.

    Parameters
    ----------
    dataset: Dataset
        DGP Dataset object.

    dataset_path: str
        Path to datase.json or root directory of scenes.

    """
    # TODO: display more dataset info
    st.sidebar.markdown("# Dataset Info")
    if dataset.dataset_metadata.metadata:
        st.sidebar.markdown("* Name: `{}`".format(dataset.dataset_metadata.metadata.name))
        st.sidebar.markdown("* Version: `{}`".format(dataset.dataset_metadata.metadata.version))
    st.sidebar.markdown("* Path: `{}`".format(dataset_path))
    st.sidebar.markdown("* Split: `{}`".format(dataset.split))


def display_metadata(metadata):
    """Display metadata information on the sidebar.

    Parameters
    ----------
    metadata: OrderedDict
        Metadata of a datum, sample or scene.

    """
    st.sidebar.markdown("# Selected Data Metadata")
    # TODO: display more sample info
    TO_DISPLAY = ('scene_index', 'sample_index_in_scene', 'log_id', 'timestamp', 'scene_name', 'scene_description')
    for item in TO_DISPLAY:
        if item in metadata:
            st.sidebar.markdown("{}: `{}`".format(item, metadata[item]))


def display_image(image, header=None, description=None):
    """Display image and description.

    Parameters
    ----------
    image: np.array
        RGB image.

    header: str
        Header of the image.

    description: str
        description of the image.

    """
    # Draw the header and image.
    if header:
        st.subheader(header)
    if description:
        st.markdown(description)
    st.image(image.astype(np.uint8), use_column_width=True)


def scene_selector_ui(dataset):
    """Interactive sidebar UI enable users to query scene, sample, datum in a dataset.

    Parameters
    ----------
    dataset: Dataset
        DGP Dataset object.

    Returns
    -------
    selected_data: OrderedDict
        Selected data.

    selected_metadata: dict
        Metadata of the selected data.

    """
    st.sidebar.markdown("# Scene")

    scene_idx = st.sidebar.slider("Choose a scene (index)", 0, len(dataset.scenes) - 1, 0) \
        if len(dataset.scenes) > 1 else 0
    st.sidebar.markdown("Scene name: `{}`".format(dataset.scenes[scene_idx].scene.name))
    scene = dataset.get_scene(scene_idx)

    st.sidebar.markdown("# Sample")
    sample_idx = st.sidebar.slider("Choose a sample (index)", 0, len(scene.samples) - 1) \
        if len(scene.samples) > 1 else 0

    datum_name_to_datum_index = dataset.get_lookup_from_datum_name_to_datum_index_in_sample(scene_idx, sample_idx)
    idx = dataset.dataset_item_index.index((scene_idx, sample_idx, list(datum_name_to_datum_index.values())))

    data = dataset[idx]
    # TODO: support sample level metadata
    # Fow now we assume only the first sample(indexed at 0) has metadata (scene-level metadata).
    metadata = dataset.metadata_index[(scene_idx, 0)]

    return data, metadata


def run_synchronized_visualizer(dataset_path, dataset_split="train"):
    """App to visualize SynchronizedDataset.

    Parameters
    ----------
    dataset_path: str
        Path to dataset.json

    dataset_split: str, default: train
         Split of dataset to read (train | val | test | train_overfit).

    """
    @st.cache(ignore_hash=True)
    def _load_dataset(path, split):
        return SynchronizedSceneDataset(
            # TODO: add a interactive checkbox to enable users to select datums.
            scene_dataset_json=path,
            split=split,
            requested_annotations=("bounding_box_3d", ),
            only_annotated_datums=True
        )

    dataset = _load_dataset(dataset_path, dataset_split)

    selected_data, selected_metadata = scene_selector_ui(dataset)

    lidar_datum_name_prefix = st.sidebar.text_input("Enter camera datum name prefix", "LIDAR")
    d_lidar = [d for d in selected_data if d["datum_name"].startswith(lidar_datum_name_prefix)]
    if not d_lidar:
        st.warning("No LiDAR datum was found. Please check if `lidar_datum_name_prefix` is correct.")
    X_W, bev = render_bev(d_lidar)

    # TODO: unify camera and lidar datum name prefix in the dataset.
    camera_datum_name_prefix = st.sidebar.text_input("Enter camera datum name prefix", "CAM")
    d_cam = [d for d in selected_data if d["datum_name"].startswith(camera_datum_name_prefix)]
    if not d_cam:
        st.warning("No camera datum was found. Please check if `camera_datum_name_prefix` is correct.")
    images_2d, images_3d = render_pointcloud_and_box_onto_rgb(d_cam, X_W)

    display_metadata(selected_metadata)
    if bev:
        display_image(bev.data, "", "BEV")
    for _datum_name, _rgb in images_2d.items():
        st.subheader("datum_name: `{}`".format(_datum_name))
        display_image(_rgb, "", "RGB with bounding boxes")
        display_image(images_3d[_datum_name], "", "RGB with bounding boxes and LiDAR projection")


# TODO: refactor `run_synchronized_visualizer` and `run_parallel_domain_visualizer` into single `run_visualizer`.
def run_parallel_domain_visualizer(dataset_path, dataset_split="train"):
    """App to visualize SynchronizedDataset.

    Parameters
    ----------
    dataset_path: str
        Path to dataset.json

    dataset_split: str, default: train
         Split of dataset to read (train | val | test | train_overfit).

    """
    def _load_dataset(path, split):
        return ParallelDomainSceneDataset(
            # TODO: add a interactive checkbox to enable users to select datums.
            scene_dataset_json=path,
            split=split,
            datum_names=[
                "camera_01", "camera_04", "camera_05", "camera_06", "camera_07", "camera_08", "camera_09", "lidar"
            ],
            requested_annotations=("bounding_box_3d", ),
        )

    dataset = _load_dataset(dataset_path, dataset_split)

    selected_data, selected_metadata = scene_selector_ui(dataset)

    lidar_datum_name_prefix = st.sidebar.text_input("Enter camera datum name prefix", "lidar")
    d_lidar = [d for d in selected_data if d["datum_name"].startswith(lidar_datum_name_prefix)]
    if not d_lidar:
        st.warning("No LiDAR datum was found. Please check if `lidar_datum_name_prefix` is correct.")
    X_W, bev = render_bev(d_lidar)

    # TODO: unify camera and lidar datum name prefix in the dataset.
    camera_datum_name_prefix = st.sidebar.text_input("Enter camera datum name prefix", "cam")
    d_cam = [d for d in selected_data if d["datum_name"].startswith(camera_datum_name_prefix)]
    if not d_cam:
        st.warning("No camera datum was found. Please check if `camera_datum_name_prefix` is correct.")
    images_2d, images_3d = render_pointcloud_and_box_onto_rgb(d_cam, X_W)

    display_metadata(selected_metadata)
    if bev:
        display_image(bev.data, "", "BEV")
    for _datum_name, _rgb in images_2d.items():
        st.subheader("datum_name: `{}`".format(_datum_name))
        display_image(_rgb, "", "RGB with bounding boxes")
        display_image(images_3d[_datum_name], "", "RGB with bounding boxes and LiDAR projection")


@st.cache
def get_dataset_split(dataset_path):
    """Get a list of splits in the dataset.json.

    Parameters
    ----------
    dataset_path: str
        Full path to the dataset json holding dataset metadata, ontology, and image and annotation paths.

    Returns
    -------
    dataset_splits: list of str
        List of dataset splits (train | val | test | train_overfit).
    """
    return DatasetMetadata.get_dataset_splits(dataset_path)


def main():
    with open(os.path.join(os.path.abspath(os.path.join(__file__, *3 * [os.path.pardir])), "README.md"), "r") as _f:
        readme_text = st.markdown(_f.read())

    st.sidebar.title("What to visualize?")
    dataset_path = st.sidebar.text_input("Enter the dataset root directory", "/mnt/fsx/dgp/")
    dataset_split = "train"

    dataset_json_list = glob.glob(os.path.join(dataset_path, "*.json"))
    if dataset_json_list:
        dataset_version = st.sidebar.selectbox(
            "Choose a version (dataset.json)", [os.path.basename(_x) for _x in dataset_json_list]
        )
        dataset_path = os.path.join(dataset_path, dataset_version)
        splits = get_dataset_split(dataset_path)
        dataset_split = st.sidebar.selectbox("Choose a dataset split to read", splits)

    app_mode = st.sidebar.selectbox(
        "Choose the mode", [
            "Show instructions", "Show the visualizer source code", "Run Parallel Domain Visualizer",
            "Run Synchronized Visualizer"
        ]
    )

    # TODO: automatically detect type of the dataset and display a list of available visualization methods.
    if app_mode == "Show instructions":
        st.sidebar.success('To continue enter the "dataset path" and suitable "visualizer"')
    elif app_mode == "Show the visualizer source code":
        readme_text.empty()
        with open(os.path.abspath(__file__), "r") as _f:
            st.code(_f.read())
    elif app_mode == "Run Synchronized Visualizer":
        readme_text.empty()
        run_synchronized_visualizer(dataset_path, dataset_split)
    elif app_mode == "Run Parallel Domain Visualizer":
        readme_text.empty()
        run_parallel_domain_visualizer(dataset_path, dataset_split)


if __name__ == "__main__":
    main()
