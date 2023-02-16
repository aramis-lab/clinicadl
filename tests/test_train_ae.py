# coding: utf8

import json
import os
import shutil

import pytest


@pytest.fixture(
    params=[
        "train_image_ae",
        "train_patch_ae",
        "train_roi_ae",
        "train_slice_ae",
    ]
)
def cli_commands(request):
    if request.param == "train_image_ae":
        mode = "image"
        test_input = [
            "train",
            "reconstruction",
            "data/dataset/random_example",
            "extract_image.json",
            "data/labels_list",
            "results",
            "-c",
            "data/train_config.toml",
        ]
    elif request.param == "train_patch_ae":
        mode = "patch"
        test_input = [
            "train",
            "reconstruction",
            "data/dataset/random_example",
            "extract_patch.json",
            "data/labels_list",
            "results",
            "-c",
            "data/train_config.toml",
        ]
    elif request.param == "train_roi_ae":
        mode = "roi"
        test_input = [
            "train",
            "reconstruction",
            "data/dataset/random_example",
            "extract_roi.json",
            "data/labels_list",
            "results",
            "-c",
            "data/train_config.toml",
        ]
    elif request.param == "train_slice_ae":
        mode = "slice"
        test_input = [
            "train",
            "reconstruction",
            "data/dataset/random_example",
            "extract_slice.json",
            "data/labels_list",
            "results",
            "-c",
            "data/train_config.toml",
        ]
    else:
        raise NotImplementedError(f"Test {request.param} is not implemented.")

    return test_input, mode


def test_train(cli_commands):
    if os.path.exists("results"):
        shutil.rmtree("results")

    test_input, mode = cli_commands
    if os.path.exists("results"):
        shutil.rmtree("results")
    flag_error = not os.system("clinicadl " + " ".join(test_input))
    assert flag_error
    with open(os.path.join("results", "maps.json"), "r") as f:
        json_data = json.load(f)
    assert json_data["mode"] == mode

    shutil.rmtree("results")
