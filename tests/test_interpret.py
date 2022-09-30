# coding: utf8

import os
import shutil
from os.path import join

import pytest

from clinicadl import MapsManager

# root = "/network/lustre/iss02/aramis/projects/clinicadl/data"


@pytest.fixture(params=["classification", "regression"])
def cli_commands(request):
    if request.param == "classification":
        cnn_input = [
            "train",
            "classification",
            "interpret/in/caps_image",
            "t1-linear_mode-image.json",
            "interpret/in/labels_list",
            "interpret/out/maps",
            "--architecture Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]

    elif request.param == "regression":
        cnn_input = [
            "train",
            "regression",
            "interpret/in/caps_patch",
            "t1-linear_mode-patch.json",
            "interpret/in/labels_list",
            "interpret/out/maps",
            "--architecture Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
    else:
        raise NotImplementedError(f"Test {request.param} is not implemented.")

    return cnn_input


def test_interpret(cli_commands):
    from clinicadl.interpret.gradients import method_dict

    maps_path = "interpret/out/maps"
    cnn_input = cli_commands
    if os.path.exists(maps_path):
        shutil.rmtree(maps_path)

    train_error = not os.system("clinicadl " + " ".join(cnn_input))
    assert train_error
    maps_manager = MapsManager(maps_path, verbose="debug")
    for method in method_dict.keys():
        maps_manager.interpret("train", f"test-{method}", method)
        interpret_map = maps_manager.get_interpretation("train", f"test-{method}")
    shutil.rmtree(maps_path)
