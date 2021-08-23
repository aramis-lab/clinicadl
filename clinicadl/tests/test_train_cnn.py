# coding: utf8

import json
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
        mode = "slice"
        test_input = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629294320.json",
            "data/labels_list",
            output_dir,
            "-c",
            "data/train_config.toml",
        ]
    elif request.param == "train_image_cnn":
        mode = "image"
        split = "1"
        test_input = [
            "train",
            "regression",
            "data/dataset/random_example",
            "extract_1629205602.json",
            "data/labels_list",
            output_dir,
            "-c",
            "data/train_config.toml",
            "--split",
            split,
        ]
    elif request.param == "train_patch_cnn":
        mode = "patch"
        split = "1"
        test_input = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629271314.json",
            "data/labels_list",
            output_dir,
            "-c",
            "data/train_config.toml",
            "--split",
            split,
        ]
    elif request.param == "train_patch_multicnn":
        mode = "patch"
        test_input = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629271314.json",
            "data/labels_list",
            output_dir,
            "-c",
            "data/train_config.toml",
            "--multi_network",
        ]
    elif request.param == "train_roi_cnn":
        mode = "roi"
        test_input = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629458899.json",
            "data/labels_list",
            output_dir,
            "-c",
            "data/train_config.toml",
        ]
    elif request.param == "train_roi_multicnn":
        mode = "roi"
        test_input = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629458899.json",
            "data/labels_list",
            output_dir,
            "-c",
            "data/train_config.toml",
        ]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return test_input, split, mode


def test_train(cli_commands):
    test_input, split, mode = cli_commands
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    flag_error = not os.system("clinicadl " + " ".join(test_input))
    assert flag_error
    performances_flag = os.path.exists(
        os.path.join("results", f"fold-{split}", "best-loss", "train")
    )
    assert performances_flag
    with open(os.path.join("results", "maps.json"), "r") as f:
        json_data = json.load(f)
    assert json_data["mode"] == mode

    shutil.rmtree(output_dir)
