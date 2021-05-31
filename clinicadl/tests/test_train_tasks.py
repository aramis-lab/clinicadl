# coding: utf8

import os
import shutil

import pytest


@pytest.fixture(
    params=[
        "train_sex_classification",
        "train_age_regression",
        "train_multi_age_regression",
    ]
)
def cli_commands(request):

    if request.param == "train_sex_classification":
        test_input = [
            "train",
            "roi",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "Conv4_FC3",
            "--epochs 1",
            "--n_splits 2",
            "--split 0",
            "--network_task classification",
            "--label sex",
        ]
    elif request.param == "train_age_regression":
        test_input = [
            "train",
            "roi",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "Conv4_FC3",
            "--epochs 1",
            "--n_splits 2",
            "--split 0",
            "--network_task regression",
            "--label age",
        ]

    elif request.param == "train_multi_age_regression":
        test_input = [
            "train",
            "roi",
            "multicnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            "Conv4_FC3",
            "--epochs 1",
            "--n_splits 2",
            "--split 0",
            "--network_task regression",
            "--label age",
        ]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

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
