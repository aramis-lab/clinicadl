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
    split = "0"
    if request.param == "train_slice_cnn":
        test_input = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629294320.json",
            "data/labels_list",
            output_dir,
            "--architecture",
            "Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            split,
        ]
    elif request.param == "train_image_cnn":
        split = "1"
        test_input = [
            "train",
            "regression",
            "data/dataset/random_example",
            "extract_1629205602.json",
            "data/labels_list",
            output_dir,
            "--architecture",
            "Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            split,
        ]
    elif request.param == "train_patch_cnn":
        split = "1"
        test_input = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629271314.json",
            "data/labels_list",
            output_dir,
            "--architecture",
            "Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            split,
        ]
    elif request.param == "train_patch_multicnn":
        test_input = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629271314.json",
            "data/labels_list",
            output_dir,
            "--architecture",
            "Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            split,
            "--multi_network",
        ]
    elif request.param == "train_roi_cnn":
        test_input = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629205602.json",
            "data/labels_list",
            output_dir,
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            split,
        ]
    elif request.param == "train_roi_multicnn":
        test_input = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629271314.json",
            "data/labels_list",
            output_dir,
            "--architecture",
            "Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            split,
            "--multi_network",
        ]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return test_input, split


def test_train(cli_commands):
    test_input, split = cli_commands
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    flag_error = not os.system("clinicadl " + " ".join(test_input))
    performances_flag = os.path.exists(
        os.path.join("results", f"fold-{split}", "best-loss", "train")
    )
    assert flag_error
    assert performances_flag
    shutil.rmtree(output_dir)
