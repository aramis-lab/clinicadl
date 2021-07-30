# coding: utf8
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
        "predict_roi_multi_classification",
        "predict_roi_reconstruction",
    ]
)
def predict_commands(request):
    if request.param == "predict_image_classification":
        model_folder = "data/models/maps_image/"
        modes = ["image"]
        use_labels = True
    elif request.param == "predict_slice_classification":
        model_folder = "data/models/maps_slice/"
        modes = ["image", "slice"]
        use_labels = True
    elif request.param == "predict_patch_regression":
        model_folder = "data/models/maps_patch/"
        modes = ["image", "patch"]
        use_labels = False
    elif request.param == "predict_roi_regression":
        model_folder = "data/models/maps_roi/"
        modes = ["image", "roi"]
        use_labels = False
    elif request.param == "predict_roi_multi_classification":
        model_folder = "data/models/maps_roi_multi/"
        modes = ["image", "roi"]
        use_labels = False
    elif request.param == "predict_roi_reconstruction":
        model_folder = "data/models/maps_roi_ae/"
        modes = ["roi"]
        use_labels = False
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return model_folder, use_labels, modes


def test_predict(predict_commands):
    model_folder, use_labels, modes = predict_commands
    out_dir = join(model_folder, "fold-0/best-loss/test-RANDOM")

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    maps_manager = MapsManager(model_folder, verbose="debug")
    maps_manager.predict(
        data_group="test-RANDOM",
        caps_directory="data/dataset/OasisCaps_example",
        tsv_path="data/dataset/OasisCaps_example/data.tsv",
        use_labels=use_labels,
        overwrite=True,
    )

    for mode in modes:
        maps_manager.get_prediction(data_group="test-RANDOM", mode=mode)
        if use_labels:
            maps_manager.get_metrics(data_group="test-RANDOM", mode=mode)
