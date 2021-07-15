# coding: utf8
import os
import shutil
from os.path import exists, join

import pytest


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
    out_dir = "fold-0/best-loss/test-RANDOM"
    if request.param == "predict_image_classification":
        model_folder = "data/models/maps_image/"
        test_input = [
            "predict",
            "data/dataset/OasisCaps_example",
            "data/dataset/OasisCaps_example/data.tsv",
            model_folder,
            "test-RANDOM",
            "-cpu",
            "--selection_metrics loss",
        ]
    elif request.param == "predict_slice_classification":
        model_folder = "data/models/maps_slice/"
        test_input = [
            "predict",
            "data/dataset/OasisCaps_example",
            "data/dataset/OasisCaps_example/data.tsv",
            model_folder,
            "test-RANDOM",
            "-cpu",
            "--selection_metrics loss",
        ]
    elif request.param == "predict_patch_regression":
        model_folder = "data/models/maps_patch/"
        test_input = [
            "predict",
            "data/dataset/OasisCaps_example",
            "data/dataset/OasisCaps_example/data.tsv",
            model_folder,
            "test-RANDOM",
            "-cpu",
            "-nl",
            "--selection_metrics loss",
        ]
    elif request.param == "predict_roi_regression":
        model_folder = "data/models/maps_roi/"
        test_input = [
            "predict",
            "data/dataset/OasisCaps_example",
            "data/dataset/OasisCaps_example/data.tsv",
            model_folder,
            "test-RANDOM",
            "-cpu",
            "-nl",
            "--selection_metrics loss",
        ]
    elif request.param == "predict_roi_multi_classification":
        model_folder = "data/models/maps_roi_multi/"
        test_input = [
            "predict",
            "data/dataset/OasisCaps_example",
            "data/dataset/OasisCaps_example/data.tsv",
            model_folder,
            "test-RANDOM",
            "-cpu",
            "--selection_metrics loss",
        ]
    elif request.param == "predict_roi_reconstruction":
        model_folder = "data/models/maps_roi_ae/"
        test_input = [
            "predict",
            "data/dataset/OasisCaps_example",
            "data/dataset/OasisCaps_example/data.tsv",
            model_folder,
            "test-RANDOM",
            "-cpu",
            "--selection_metrics loss",
        ]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    output_files = join(model_folder, out_dir)
    return test_input, output_files


def test_predict(predict_commands):
    test_input = predict_commands[0]
    out_dir = predict_commands[1]

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    flag_error = not os.system("clinicadl " + " ".join(test_input))

    assert flag_error
    assert exists(out_dir)
