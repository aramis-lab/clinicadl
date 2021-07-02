# coding: utf8
import os
import shutil
from os.path import exists, join

import pytest


@pytest.fixture(
    params=[
        "classify_image",
        "classify_roi",
        "classify_slice",
        "classify_patch",
        "classify_roi_multi",
        "classify_roi_ae",
    ]
)
def classify_commands(request):
    out_dir = "fold-0/best-loss/test-RANDOM"
    if request.param == "classify_image":
        model_folder = "data/models/maps_image/"
        test_input = [
            "classify",
            "data/dataset/OasisCaps_example",
            "data/dataset/OasisCaps_example/data.tsv",
            model_folder,
            "test-RANDOM",
            "-cpu",
            "--selection_metrics loss",
        ]
    elif request.param == "classify_slice":
        model_folder = "data/models/maps_slice/"
        test_input = [
            "classify",
            "data/dataset/OasisCaps_example",
            "data/dataset/OasisCaps_example/data.tsv",
            model_folder,
            "test-RANDOM",
            "-cpu",
            "--selection_metrics loss",
        ]
    elif request.param == "classify_patch":
        model_folder = "data/models/maps_patch/"
        test_input = [
            "classify",
            "data/dataset/OasisCaps_example",
            "data/dataset/OasisCaps_example/data.tsv",
            model_folder,
            "test-RANDOM",
            "-cpu",
            "-nl",
            "--selection_metrics loss",
        ]
    elif request.param == "classify_roi":
        model_folder = "data/models/maps_roi/"
        test_input = [
            "classify",
            "data/dataset/OasisCaps_example",
            "data/dataset/OasisCaps_example/data.tsv",
            model_folder,
            "test-RANDOM",
            "-cpu",
            "-nl",
            "--selection_metrics loss",
        ]
    elif request.param == "classify_roi_multi":
        model_folder = "data/models/maps_roi_multi/"
        test_input = [
            "classify",
            "data/dataset/OasisCaps_example",
            "data/dataset/OasisCaps_example/data.tsv",
            model_folder,
            "test-RANDOM",
            "-cpu",
            "--selection_metrics loss",
        ]
    elif request.param == "classify_roi_ae":
        model_folder = "data/models/maps_roi_ae/"
        test_input = [
            "classify",
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


def test_classify(classify_commands):
    test_input = classify_commands[0]
    out_dir = classify_commands[1]

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    flag_error = not os.system("clinicadl " + " ".join(test_input))

    assert flag_error
    assert exists(out_dir)
