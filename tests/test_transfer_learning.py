import json
import os
import shutil
from pathlib import Path

import pytest

from .testing_tools import compare_folders, modify_maps


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
            str(tmp_out_dir / "maps_roi_ae"),
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
            str(tmp_out_dir / "maps_roi_ae"),
        ]
        name = "aeTOae"
    elif test_name == "transfer_ae_cnn":
        source_task = [
            "train",
            "reconstruction",
            str(caps_roi_path),
            extract_roi_str,
            str(labels_path),
            str(tmp_out_dir / "maps_roi_ae"),
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
            str(tmp_out_dir / "maps_roi_ae"),
        ]
        name = "aeTOcnn"
    elif test_name == "transfer_cnn_cnn":
        source_task = [
            "train",
            "classification",
            str(caps_roi_path),
            extract_roi_str,
            str(labels_path),
            str(tmp_out_dir / "maps_roi_cnn"),
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
            str(tmp_out_dir / "maps_roi_cnn"),
        ]
        name = "cnnTOcnn"
    elif test_name == "transfer_cnn_multicnn":
        source_task = [
            "train",
            "classification",
            str(caps_roi_path),
            extract_roi_str,
            str(labels_path),
            str(tmp_out_dir / "maps_roi_cnn"),
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
            str(tmp_out_dir / "maps_roi_cnn"),
            "--multi_network",
        ]
        name = "cnnTOmulticnn"
    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    if cmdopt["no-gpu"]:
        source_task.append("--no-gpu")
        target_task.append("--no-gpu")

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

    ref_source_dir = Path(json_data_ref["transfer_path"]).parent
    json_data_ref["transfer_path"] = str(
        tmp_out_dir / Path(json_data_ref["transfer_path"]).relative_to(ref_source_dir)
    )
    if cmdopt["no-gpu"] or cmdopt["adapt-base-dir"]:
        json_data_ref = modify_maps(
            maps=json_data_ref,
            base_dir=base_dir,
            no_gpu=cmdopt["no-gpu"],
            adapt_base_dir=cmdopt["adapt-base-dir"],
        )
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
