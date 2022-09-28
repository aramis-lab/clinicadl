# coding: utf8
import json
import os
import shutil
from os.path import exists, join

import pytest

from clinicadl import MapsManager


@pytest.fixture(
    params=[
        "predict_image_classification",
        "predict_roi_regression",
        "predict_slice_classification",
        "predict_patch_regression",
        "predict_patch_multi_classification",
        "predict_roi_reconstruction",
    ]
)
def predict_commands(request):

    # root = "/network/lustre/iss02/aramis/projects/clinicadl/data/"
    root = "/mnt/data/data_CI"
    if request.param == "predict_image_classification":
        model_folder = join(
            root, "predict/in/maps_image_cnn"
        )  # data/models/maps_image/"
        modes = ["image"]
        use_labels = True
        caps_folder = join(root, "predict/in/caps_image")
    elif request.param == "predict_slice_classification":
        model_folder = join(
            root, "predict/in/maps_slice_cnn"
        )  # "data/models/maps_slice/"
        modes = ["image", "slice"]
        use_labels = True
        caps_folder = join(root, "predict/in/caps_slice")
    elif request.param == "predict_patch_regression":
        model_folder = join(
            root, "predict/in/maps_patch_cnn"
        )  # "data/models/maps_patch/"
        modes = ["image", "patch"]
        use_labels = False
        caps_folder = join(root, "predict/in/caps_patch")
    elif request.param == "predict_roi_regression":
        model_folder = join(root, "predict/in/maps_roi_cnn")  # "data/models/maps_roi/"
        modes = ["image", "roi"]
        use_labels = False
        caps_folder = join(root, "predict/in/caps_roi")
    elif request.param == "predict_patch_multi_classification":
        model_folder = join(
            root, "predict/in/maps_patch_multi_cnn"
        )  # "data/models/maps_roi_multi/"
        modes = ["image", "patch"]
        use_labels = False
        caps_folder = join(root, "predict/in/caps_patch")
    elif request.param == "predict_roi_reconstruction":
        model_folder = join(
            root, "predict/in/maps_roi_ae"
        )  # "data/models/maps_roi_ae/"
        modes = ["roi"]
        use_labels = False
        caps_folder = join(root, "predict/in/caps_roi")
    else:
        raise NotImplementedError(f"Test {request.param} is not implemented.")

    return model_folder, use_labels, modes, caps_folder


def test_predict(predict_commands):
    model_folder, use_labels, modes, caps_folder = predict_commands
    out_dir = join(model_folder, "split-0/best-loss/test-RANDOM")

    if exists(out_dir):
        shutil.rmtree(out_dir)

    # Correction of JSON file for ROI
    if "roi" in modes:
        json_path = join(model_folder, "maps.json")
        with open(json_path, "r") as f:
            parameters = json.load(f)
        parameters["roi_list"] = ["leftHippocampusBox", "rightHippocampusBox"]
        json_data = json.dumps(parameters, skipkeys=True, indent=4)
        with open(json_path, "w") as f:
            f.write(json_data)

    maps_manager = MapsManager(model_folder, verbose="debug")
    maps_manager.predict(
        data_group="test-RANDOM",
        caps_directory="/network/lustre/iss02/aramis/projects/clinicadl/data/predict/in/caps_random",
        tsv_path="/network/lustre/iss02/aramis/projects/clinicadl/data/predict/in/caps_random/data.tsv",
        gpu=False,
        use_labels=use_labels,
        overwrite=True,
        diagnoses=["CN"],
    )

    for mode in modes:
        maps_manager.get_prediction(data_group="test-RANDOM", mode=mode)
        if use_labels:
            maps_manager.get_metrics(data_group="test-RANDOM", mode=mode)
