# coding: utf8

import json
import os
import shutil
from pathlib import Path

import pytest

from clinicadl import MapsManager
from tests.testing_tools import clean_folder, compare_folders


@pytest.fixture(params=["classification", "regression"])
def test_name(request):
    return request.param


def test_interpret(cmdopt, tmp_path, test_name):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "interpret" / "in"
    ref_dir = base_dir / "interpret" / "ref"
    tmp_out_dir = ref_dir  # = tmp_path / "interpret" / "out"
    clean_folder(tmp_out_dir)
    # tmp_out_dir.mkdir(parents=True)

    labels_dir_str = str(input_dir / "labels_list")
    maps_tmp_out_dir = str(tmp_out_dir / "maps")
    if test_name == "classification":
        cnn_input = [
            "train",
            "classification",
            str(input_dir / "caps_image"),
            "t1-linear_mode-image.json",
            labels_dir_str,
            maps_tmp_out_dir,
            "--architecture Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]

    elif test_name == "regression":
        cnn_input = [
            "train",
            "regression",
            str(input_dir / "caps_patch"),
            "t1-linear_mode-patch.json",
            labels_dir_str,
            maps_tmp_out_dir,
            "--architecture Conv5_FC3",
            "--epochs",
            "1",
            "--n_splits",
            "2",
            "--split",
            "0",
        ]
    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    run_interpret(cnn_input, tmp_out_dir, ref_dir)


def run_interpret(cnn_input, tmp_out_dir, ref_dir):
    from clinicadl.interpret.gradients import method_dict

    maps_path = tmp_out_dir / "maps"
    if os.path.exists(maps_path):
        shutil.rmtree(maps_path)

    train_error = not os.system("clinicadl " + " ".join(cnn_input))
    assert train_error
    maps_manager = MapsManager(maps_path, verbose="debug")
    for method in method_dict.keys():
        maps_manager.interpret("train", f"test-{method}", method)
        interpret_map = maps_manager.get_interpretation("train", f"test-{method}")

    with open(maps_path / "maps.json", "r") as out:
        json_data_out = json.load(out)
    with open(ref_dir / "maps_image" / "maps.json", "r") as ref:
        json_data_ref = json.load(ref)

    json_data_out[
        "transfer_path"
    ] = "/mnt/data/clinicadl_data_ci/data_ci/transferLearning/in/maps_roi_ae"
    json_data_out["gpu"] = "False"
    json_data_out[
        "caps_dirctory"
    ] = "/mnt/data/clinicadl_data_ci/data_ci/transferLearning/in/caps_roi"
    json_data_ref[
        "caps_dirctory"
    ] = "/mnt/data/clinicadl_data_ci/data_ci/transferLearning/in/caps_roi"
    assert json_data_out == json_data_ref  # ["mode"] == mode
