# coding: utf8

import json
import os
import shutil
from os.path import join

import pytest

# root = "/network/lustre/iss02/aramis/projects/clinicadl/data"
root = "/mnt/data/data_CI"


@pytest.fixture(
    params=[
        "train_image_ae",
        "train_patch_ae",
        "train_roi_ae",
        "train_slice_ae",
    ]
)
def cli_commands(request):
    out_path = join(root, "train/out")
    labels_path = join(root, "train/in/labels_list")
    config_path = join(root, "train/in/train_config.toml")
    if request.param == "train_image_ae":
        mode = "image"
        test_input = [
            "train",
            "reconstruction",
            join(root, "train/in/caps_image"),
            "t1-linear_mode-image.json",
            labels_path,
            out_path,
            "-c",
            config_path,
        ]
    elif request.param == "train_patch_ae":
        mode = "patch"
        test_input = [
            "train",
            "reconstruction",
            join(root, "train/in/caps_patch"),
            "t1-linear_mode-patch.json",
            labels_path,
            out_path,
            "-c",
            config_path,
        ]
    elif request.param == "train_roi_ae":
        mode = "roi"
        test_input = [
            "train",
            "reconstruction",
            join(root, "train/in/caps_roi"),
            "t1-linear_mode-roi.json",
            labels_path,
            out_path,
            "-c",
            config_path,
        ]
    elif request.param == "train_slice_ae":
        mode = "slice"
        test_input = [
            "train",
            "reconstruction",
            join(root, "train/in/caps_slice"),
            "t1-linear_mode-slice.json",
            labels_path,
            out_path,
            "-c",
            config_path,
        ]
    else:
        raise NotImplementedError(f"Test {request.param} is not implemented.")

    return test_input, mode


def test_train(cli_commands):

    out_path = join(root, "train/out")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    test_input, mode = cli_commands
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    flag_error = not os.system("clinicadl " + " ".join(test_input))
    assert flag_error
    with open(os.path.join(out_path, "maps.json"), "r") as f:
        json_data = json.load(f)
    assert json_data["mode"] == mode

    shutil.rmtree(out_path)
