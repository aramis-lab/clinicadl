# coding: utf8

import json
import os
import shutil
from os.path import join

import pytest

# root = "/network/lustre/iss02/aramis/projects/clinicadl/data"

output_dir = "train/out"


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
def test_name(request):
    return request.param


def cli_commands(cmdopt, tmp_path, test_name):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "train" / "in"
    ref_dir = base_dir / "train" / "ref"
    tmp_out_dir = tmp_path / "train" / "out"
    tmp_out_dir.mkdir(parents=True)

    labels_path = join(input_dir, "labels_list")
    config_path = join(input_dir, "train_config.toml")
    split = "0"

    if test_name == "train_slice_cnn":
        mode = "slice"
        test_input = [
            "train",
            "classification",
            "train/in/caps_slice",
            "t1-linear_mode-slice.json",
            labels_path,
            output_dir,
            "-c",
            config_path,
        ]
    elif test_name == "train_image_cnn":
        mode = "image"
        split = "1"
        test_input = [
            "train",
            "regression",
            join(input_dir, "caps_image"),
            "t1-linear_mode-image.json",
            labels_path,
            output_dir,
            "-c",
            config_path,
            "-s",
            split,
        ]
    elif test_name == "train_patch_cnn":
        mode = "patch"
        split = "0"
        test_input = [
            "train",
            "classification",
            "train/in/caps_patch",
            "t1-linear_mode-patch.json",
            labels_path,
            output_dir,
            "-c",
            config_path,
            "--split",
            split,
        ]
    elif test_name == "train_patch_multicnn":
        mode = "patch"
        test_input = [
            "train",
            "classification",
            "train/in/caps_patch",
            "t1-linear_mode-patch.json",
            labels_path,
            output_dir,
            "-c",
            config_path,
            "--multi_network",
        ]
    elif test_name == "train_roi_cnn":
        mode = "roi"
        test_input = [
            "train",
            "classification",
            "train/in/caps_roi",
            "t1-linear_mode-roi.json",
            labels_path,
            output_dir,
            "-c",
            config_path,
        ]
    elif test_name == "train_roi_multicnn":
        mode = "roi"
        test_input = [
            "train",
            "classification",
            "train/in/caps_roi",
            "t1-linear_mode-roi.json",
            labels_path,
            output_dir,
            "-c",
            config_path,
        ]
    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    return test_input, split, mode


def test_train(cli_commands):

    test_input, split, mode = cli_commands
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    flag_error = not os.system("clinicadl " + " ".join(test_input))
    assert flag_error
    performances_flag = os.path.exists(
        os.path.join(output_dir, f"split-{split}", "best-loss", "train")
    )
    assert performances_flag
    with open(os.path.join(output_dir, "maps.json"), "r") as f:
        json_data = json.load(f)
    assert json_data["mode"] == mode

    shutil.rmtree(output_dir)
