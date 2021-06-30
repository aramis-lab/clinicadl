# coding: utf8

import os
import shutil

import pytest

output_dir = "results"


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
        test_input = [
            "train",
            "slice",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            output_dir,
            "resnet18",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
    elif request.param == "train_image_cnn":
        test_input = [
            "train",
            "image",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            output_dir,
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
    elif request.param == "train_patch_cnn":
        test_input = [
            "train",
            "patch",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            output_dir,
            "Conv4_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
    elif request.param == "train_patch_multicnn":
        test_input = [
            "train",
            "patch",
            "multicnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            output_dir,
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
    elif request.param == "train_roi_cnn":
        test_input = [
            "train",
            "roi",
            "cnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            "results",
            output_dir,
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
    elif request.param == "train_roi_multicnn":
        test_input = [
            "train",
            "roi",
            "multicnn",
            "data/dataset/random_example",
            "t1-linear",
            "data/labels_list",
            output_dir,
            "Conv4_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return test_input


def test_train(cli_commands):
    test_input = cli_commands
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    flag_error = not os.system("clinicadl " + " ".join(test_input))
    performances_flag = os.path.exists(
        os.path.join("results", "fold-0", "cnn_classification")
    )
    assert flag_error
    assert performances_flag
    shutil.rmtree(output_dir)
