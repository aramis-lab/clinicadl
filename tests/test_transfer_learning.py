import json
import os
import shutil
from os.path import join
from pathlib import Path

import pytest

from tests.testing_tools import compare_folders


# Everything is tested on roi except for cnn --> multicnn (patch) as multicnn is not implemented for roi.
@pytest.fixture(
    params=[
        "transfer_ae_ae",
        "transfer_ae_cnn",
        "transfer_cnn_cnn",
        "transfer_cnn_multicnn",
    ]
)
def test_name(request):
    return request.param


def test_transfer_learning(cmdopt, tmp_path, test_name):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "transferLearning" / "in"
    ref_dir = base_dir / "transferLearning" / "ref"
    tmp_out_dir = tmp_path / "transferLearning" / "out"
    tmp_target_dir = tmp_path / "transferLearning" / "target"
    tmp_out_dir.mkdir(parents=True)

    caps_roi_path = input_dir / "caps_roi"
    extract_roi_str = "t1-linear_mode-roi.json"
    labels_path = input_dir / "labels_list" / "2_fold"
    config_path = input_dir / "train_config.toml"
    if test_name == "transfer_ae_ae":
        source_task = [
            "train",
            "reconstruction",
            str(caps_roi_path),
            extract_roi_str,
            str(labels_path),
            str(tmp_out_dir),
            "-c",
            str(config_path),
        ]
        target_task = [
            "train",
            "reconstruction",
            str(caps_roi_path),
            extract_roi_str,
            str(labels_path),
            str(tmp_target_dir),
            "-c",
            str(config_path),
            "--transfer_path",
            str(tmp_out_dir),
        ]
        name = "aeTOae"
    elif test_name == "transfer_ae_cnn":
        source_task = [
            "train",
            "reconstruction",
            str(caps_roi_path),
            extract_roi_str,
            str(labels_path),
            str(tmp_out_dir),
            "-c",
            str(config_path),
        ]
        target_task = [
            "train",
            "classification",
            str(caps_roi_path),
            extract_roi_str,
            str(labels_path),
            str(tmp_target_dir),
            "-c",
            str(config_path),
            "--transfer_path",
            str(tmp_out_dir),
        ]
        name = "aeTOcnn"
    elif test_name == "transfer_cnn_cnn":
        source_task = [
            "train",
            "classification",
            str(caps_roi_path),
            extract_roi_str,
            str(labels_path),
            str(tmp_out_dir),
            "-c",
            str(config_path),
        ]
        target_task = [
            "train",
            "classification",
            str(caps_roi_path),
            extract_roi_str,
            str(labels_path),
            str(tmp_target_dir),
            "-c",
            str(config_path),
            "--transfer_path",
            str(tmp_out_dir),
        ]
        name = "cnnTOcnn"
    elif test_name == "transfer_cnn_multicnn":
        source_task = [
            "train",
            "classification",
            str(caps_roi_path),
            extract_roi_str,
            str(labels_path),
            str(tmp_out_dir),
            "-c",
            str(config_path),
        ]
        target_task = [
            "train",
            "classification",
            str(caps_roi_path),
            extract_roi_str,
            str(labels_path),
            str(tmp_target_dir),
            "-c",
            str(config_path),
            "--transfer_path",
            str(tmp_out_dir),
        ]
        name = "cnnTOcnn"
    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    if tmp_out_dir.exists():
        shutil.rmtree(tmp_out_dir)
    if tmp_target_dir.exists():
        shutil.rmtree(tmp_target_dir)

    flag_source = not os.system("clinicadl -vvv " + " ".join(source_task))
    flag_target = not os.system("clinicadl -vvv  " + " ".join(target_task))
    assert flag_source
    assert flag_target

    with open(tmp_target_dir / "maps.json", "r") as out:
        json_data_out = json.load(out)
    with open(ref_dir / ("maps_roi_" + name) / "maps.json", "r") as ref:
        json_data_ref = json.load(ref)

    json_data_ref["transfer_path"] = json_data_out["transfer_path"]
    json_data_ref["gpu"] = json_data_out["gpu"]
    json_data_ref["caps_directory"] = json_data_out["caps_directory"]
    assert json_data_out == json_data_ref  # ["mode"] == mode

    assert compare_folders(
        tmp_target_dir / "groups",
        ref_dir / ("maps_roi_" + name) / "groups",
        tmp_path,
    )
    assert compare_folders(
        tmp_target_dir / "split-0" / "best-loss",
        ref_dir / ("maps_roi_" + name) / "split-0" / "best-loss",
        tmp_path,
    )
