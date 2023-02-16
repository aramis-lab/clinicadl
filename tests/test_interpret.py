# coding: utf8

import os
import shutil

import pytest

from clinicadl import MapsManager


@pytest.fixture(params=["classification", "regression"])
def cli_commands(request):

    if request.param == "classification":
        cnn_input = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_image.json",
            "data/labels_list",
            "results",
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
            "data/dataset/random_example",
            "extract_patch.json",
            "data/labels_list",
            "results",
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

    cnn_input = cli_commands
    if os.path.exists("results"):
        shutil.rmtree("results")

    train_error = not os.system("clinicadl " + " ".join(cnn_input))
    assert train_error
    maps_manager = MapsManager("results", verbose="debug")
    for method in method_dict.keys():
        maps_manager.interpret("train", f"test-{method}", method)
        interpret_map = maps_manager.get_interpretation("train", f"test-{method}")
    shutil.rmtree("results")
