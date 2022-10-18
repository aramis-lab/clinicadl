# coding: utf8

import json
import os
import shutil
from os.path import join
from pathlib import Path

import pytest

from tests.testing_tools import clean_folder


@pytest.fixture(
    params=[
        "train_image_ae",
        "train_patch_ae",
        "train_roi_ae",
        "train_slice_ae",
    ]
)
def test_name(request):
    return request.param


def test_train_ae(cmdopt, tmp_path, test_name):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "train" / "in"
    ref_dir = base_dir / "train" / "ref"
    tmp_out_dir = tmp_path / "train" / "out"
    tmp_out_dir.mkdir(parents=True)

    clean_folder(tmp_out_dir, recreate=True)

    labels_path = str(input_dir / "labels_list")
    config_path = str(input_dir / "train_config.toml")
    if test_name == "train_image_ae":
        mode = "image"
        test_input = [
            "train",
            "reconstruction",
            str(input_dir / "caps_image"),
            "t1-linear_mode-image.json",
            labels_path,
            str(tmp_out_dir),
            "-c",
            config_path,
        ]
    elif test_name == "train_patch_ae":
        mode = "patch"
        test_input = [
            "train",
            "reconstruction",
            str(input_dir / "caps_patch"),
            "t1-linear_mode-patch.json",
            labels_path,
            str(tmp_out_dir),
            "-c",
            config_path,
        ]
    elif test_name == "train_roi_ae":
        mode = "roi"
        test_input = [
            "train",
            "reconstruction",
            str(input_dir / "caps_roi"),
            "t1-linear_mode-roi.json",
            labels_path,
            str(tmp_out_dir),
            "-c",
            config_path,
        ]
    elif test_name == "train_slice_ae":
        mode = "slice"
        test_input = [
            "train",
            "reconstruction",
            str(input_dir / "caps_slice"),
            "t1-linear_mode-slice.json",
            labels_path,
            str(tmp_out_dir),
            "-c",
            config_path,
        ]
    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    out_path = tmp_path / "train" / "out"
    if os.path.exists(tmp_out_dir):
        shutil.rmtree(tmp_out_dir)

    flag_error = not os.system("clinicadl " + " ".join(test_input))
    assert flag_error

    with open(tmp_out_dir / "maps.json", "r") as f:
        json_data = json.load(f)
    assert json_data["mode"] == mode
