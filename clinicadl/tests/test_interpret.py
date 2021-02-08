# coding: utf8

import os
import shutil

import pytest


@pytest.fixture(params=["group_image", "individual_image"])
def cli_commands(request):

    if request.param == "group_image":
        # fmt: off
        cnn_input = [
            "train",
            "image",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "Conv5_FC3",
            "--epochs", "1",
            "--n_splits", "2",
            "--split", "0"
        ]
        # fmt: on
        interpret_input = ["interpret", "group", "results", "group-test"]

    elif request.param == "individual_image":
        # fmt: off
        cnn_input = [
            "train",
            "image",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "Conv5_FC3",
            "--epochs", "1",
            "--n_splits", "2",
            "--split", "0"
        ]
        # fmt: on
        interpret_input = ["interpret", "individual", "results", "individual-test"]
    else:
        raise NotImplementedError(f"Test {request.param} is not implemented.")

    return cnn_input, interpret_input


def test_interpret(cli_commands):
    cnn_input, interpret_input = cli_commands
    train_error = not os.system("clinicadl " + " ".join(cnn_input))
    interpret_error = not os.system("clinicadl " + " ".join(interpret_input))
    interpret_flag = os.path.exists(os.path.join("results", "fold-0", "gradients"))
    assert train_error
    assert interpret_error
    assert interpret_flag
    shutil.rmtree("results")
