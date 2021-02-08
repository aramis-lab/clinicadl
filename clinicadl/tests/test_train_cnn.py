# coding: utf8

import os
import shutil

import pytest


@pytest.fixture(
    params=[
        "train_slice_cnn",
        "train_image_cnn",
        "train_patch_cnn",
        "train_patch_multicnn",
        "train_roi_cnn",
        "train_roi_multicnn",
    ]
)
def cli_commands(request):

    if request.param == "train_slice_cnn":
        # fmt: off
        test_input = [
            "train",
            "slice",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "resnet18",
            "--epochs", "1",
            "--n_splits", "2",
            "--split", "0"
        ]
        # fmt: on
    elif request.param == "train_image_cnn":
        # fmt: off
        test_input = [
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
    elif request.param == "train_patch_cnn":
        # fmt: off
        test_input = [
            "train",
            "patch",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "Conv4_FC3",
            "--epochs", "1",
            "--n_splits", "2",
            "--split", "0"
        ]
        # fmt: on
    elif request.param == "train_patch_multicnn":
        # fmt: off
        test_input = [
            "train",
            "patch",
            "multicnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "Conv4_FC3",
            "--epochs", "1",
            "--n_splits", "2",
            "--split", "0"
        ]
        # fmt: on
    elif request.param == "train_roi_cnn":
        # fmt: off
        test_input = [
            "train",
            "roi",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "Conv4_FC3",
            "--epochs", "1",
            "--n_splits", "2",
            "--split", "0"
        ]
        # fmt: on
    elif request.param == "train_roi_multicnn":
        # fmt: off
        test_input = [
            "train",
            "roi",
            "multicnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "Conv4_FC3",
            "--epochs", "1",
            "--n_splits", "2",
            "--split", "0"
        ]
        # fmt: on
    else:
        raise NotImplementedError(f"Test {request.param} is not implemented.")

    return test_input


def test_train(cli_commands):
    test_input = cli_commands
    flag_error = not os.system("clinicadl " + " ".join(test_input))
    performances_flag = os.path.exists(
        os.path.join("results", "fold-0", "cnn_classification")
    )
    assert flag_error
    assert performances_flag
    shutil.rmtree("results")
