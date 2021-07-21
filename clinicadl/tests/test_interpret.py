# coding: utf8

import os
import shutil

import pytest


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
        interpret_input = ["interpret", "results", "group-test"]

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
        interpret_input = ["interpret", "results", "individual-test"]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return cnn_input, interpret_input


def test_interpret(cli_commands):
    cnn_input, interpret_input = cli_commands
    if os.path.exists("results"):
        shutil.rmtree("results")

    train_error = not os.system("clinicadl " + " ".join(cnn_input))
    interpret_error = not os.system("clinicadl " + " ".join(interpret_input))
    interpret_flag = os.path.exists(
        os.path.join("results", "fold-0", "best-loss", "interpretation")
    )
    assert train_error
    assert interpret_error
    assert interpret_flag
    shutil.rmtree("results")
