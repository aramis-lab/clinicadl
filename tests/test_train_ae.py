# coding: utf8

import json
import os
import shutil
from pathlib import Path

import pytest

from .testing_tools import clean_folder, compare_folders, modify_maps


@pytest.fixture(
    params=[
        "image_ae",
        "patch_multi_ae",
        "roi_ae",
        "slice_ae",
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

    labels_path = str(input_dir / "labels_list" / "2_fold")
    config_path = str(input_dir / "train_config.toml")
    split = 0

    if test_name == "image_ae":
        split = 1
        test_input = [
            "train",
            "reconstruction",
            str(input_dir / "caps_image"),
            "t1-linear_crop-True_mode-image.json",
            labels_path,
            str(tmp_out_dir),
            "-c",
            config_path,
            "--split",
            str(split),
        ]
    elif test_name == "patch_multi_ae":
        test_input = [
            "train",
            "reconstruction",
            str(input_dir / "caps_patch"),
            "t1-linear_crop-True_mode-patch.json",
            labels_path,
            str(tmp_out_dir),
            "-c",
            config_path,
            "--multi_network",
        ]
    elif test_name == "roi_ae":
        test_input = [
            "train",
            "reconstruction",
            str(input_dir / "caps_roi"),
            "t1-linear_crop-True_mode-roi.json",
            labels_path,
            str(tmp_out_dir),
            "-c",
            config_path,
        ]
    elif test_name == "slice_ae":
        test_input = [
            "train",
            "reconstruction",
            str(input_dir / "caps_slice"),
            "t1-linear_crop-True_mode-slice.json",
            labels_path,
            str(tmp_out_dir),
            "-c",
            config_path,
        ]
    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    if cmdopt["no-gpu"]:
        test_input.append("--no-gpu")

    if tmp_out_dir.is_dir():
        shutil.rmtree(tmp_out_dir)

    flag_error = not os.system("clinicadl " + " ".join(test_input))
    assert flag_error

    with open(tmp_out_dir / "maps.json", "r") as out:
        json_data_out = json.load(out)
    with open(ref_dir / ("maps_" + test_name) / "maps.json", "r") as ref:
        json_data_ref = json.load(ref)

    if cmdopt["no-gpu"] or cmdopt["adapt-base-dir"]:
        json_data_ref = modify_maps(
            maps=json_data_ref,
            base_dir=base_dir,
            no_gpu=cmdopt["no-gpu"],
            adapt_base_dir=cmdopt["adapt-base-dir"],
            ssda=True,
        )
    assert json_data_out == json_data_ref  # ["mode"] == mode

    assert compare_folders(
        tmp_out_dir / "groups",
        ref_dir / ("maps_" + test_name) / "groups",
        tmp_path,
    )
    assert compare_folders(
        tmp_out_dir / f"split-{split}" / "best-loss",
        ref_dir / ("maps_" + test_name) / f"split-{split}" / "best-loss",
        tmp_path,
    )
