# coding: utf8

import json
import os
import shutil
from os.path import join
from pathlib import Path

import pytest

from tests.testing_tools import compare_folders


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


def test_train_cnn(cmdopt, tmp_path, test_name):
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
            join(str(input_dir), "caps_slice"),
            "t1-linear_crop-True_mode-slice.json",
            labels_path,
            str(tmp_out_dir),
            "-c",
            config_path,
        ]
    elif test_name == "train_image_cnn":
        mode = "image"
        split = "1"
        test_input = [
            "train",
            "regression",
            join(str(input_dir), "caps_image"),
            "t1-linear_crop-True_mode-image.json",
            labels_path,
            str(tmp_out_dir),
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
            join(str(input_dir), "caps_patch"),
            "t1-linear_crop-True_mode-patch.json",
            labels_path,
            str(tmp_out_dir),
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
            join(str(input_dir), "caps_patch"),
            "t1-linear_crop-True_mode-patch.json",
            labels_path,
            str(tmp_out_dir),
            "-c",
            config_path,
            "--multi_network",
        ]
    elif test_name == "train_roi_cnn":
        mode = "roi"
        test_input = [
            "train",
            "classification",
            join(str(input_dir), "caps_roi"),
            "t1-linear_crop-True_mode-roi.json",
            labels_path,
            str(tmp_out_dir),
            "-c",
            config_path,
        ]
    elif test_name == "train_roi_multicnn":
        mode = "roi"
        test_input = [
            "train",
            "classification",
            join(str(input_dir), "caps_roi"),
            "t1-linear_crop-True_mode-roi.json",
            labels_path,
            str(tmp_out_dir),
            "-c",
            config_path,
        ]
    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    if os.path.exists(str(tmp_out_dir)):
        shutil.rmtree(str(tmp_out_dir))

    flag_error = not os.system("clinicadl " + " ".join(test_input))
    assert flag_error

    performances_flag = os.path.exists(
        os.path.join(str(tmp_out_dir), f"split-{split}", "best-loss", "train")
    )
    assert performances_flag

    with open(os.path.join(str(tmp_out_dir), "maps.json"), "r") as out:
        json_data_out = json.load(out)
    with open(
        os.path.join(str(ref_dir / ("maps_" + test_name)), "maps.json"), "r"
    ) as ref:
        json_data_ref = json.load(ref)

    assert json_data_out == json_data_ref  # ["mode"] == mode

    assert compare_folders(
        str(tmp_out_dir / "groups"),
        str(ref_dir / ("maps_" + test_name) / "groups"),
        tmp_path,
    )
    assert compare_folders(
        str(tmp_out_dir / "split-0" / "best-loss"),
        str(ref_dir / ("maps_" + test_name) / "split-0" / "best-loss"),
        tmp_path,
    )

    shutil.rmtree(str(tmp_out_dir))
