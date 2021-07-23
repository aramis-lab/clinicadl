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
            "image",
            "classification",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "--model Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--folds",
            "0",
        ]

    elif request.param == "regression":
        cnn_input = [
            "train",
            "image",
            "regression",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "--model Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--folds",
            "0",
        ]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return cnn_input


def test_interpret(cli_commands):
    cnn_input = cli_commands
    if os.path.exists("results"):
        shutil.rmtree("results")

    train_error = not os.system("clinicadl " + " ".join(cnn_input))
    maps_manager = MapsManager("results", verbose="debug")
    maps_manager.interpret("train", "test")
    interpret_map = maps_manager.get_interpretation("train", "test")
    assert train_error
    shutil.rmtree("results")
