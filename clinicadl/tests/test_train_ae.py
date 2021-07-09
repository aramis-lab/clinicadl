# coding: utf8

import os
import shutil

import pytest


@pytest.fixture(
    params=[
        "train_image_ae",
        "train_patch_ae",
        "train_roi_ae",
    ]
)
def cli_commands(request):
    if request.param == "train_image_ae":
        test_input = [
            "train",
            "image",
            "reconstruction",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "AE_Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--folds",
            "0",
        ]
    elif request.param == "train_patch_ae":
        test_input = [
            "train",
            "patch",
            "reconstruction",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "AE_Conv4_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--folds",
            "0",
        ]
    elif request.param == "train_roi_ae":
        test_input = [
            "train",
            "roi",
            "reconstruction",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "AE_Conv4_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--folds",
            "0",
        ]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return test_input


def test_train(cli_commands):
    if os.path.exists("results"):
        shutil.rmtree("results")

    test_input = cli_commands
    if os.path.exists("results"):
        shutil.rmtree("results")
    flag_error = not os.system("clinicadl " + " ".join(test_input))
    assert flag_error
    shutil.rmtree("results")
