import os
import shutil
from os.path import join

import pytest

# root="/network/lustre/iss02/aramis/projects/clinicadl/data"
root = "/mnt/data/data_CI"
output_dir = join(root, "transferLearning/out")
target_dir = join(root, "transferLeanrinng/target")

# Everything is tested on roi except for cnn --> multicnn (patch) as multicnn is not implemented for roi.
@pytest.fixture(
    params=[
        "transfer_ae_ae",
        "transfer_ae_cnn",
        "transfer_cnn_cnn",
        "transfer_cnn_multicnn",
    ]
)
def cli_commands(request):
    caps_roi_path = join(root, "transferLearning/in/caps_roi")
    caps_patch_path = join(root, "transferLearning/in/caps_patch")
    extract_roi_str = "t1-linear_mode-roi.json"
    extract_patch_str = "t1-linear_mode-patch.json"
    labels_path = join(root, "transferLearning/in/labels_list")
    config_path = join(root, "transferLearning/in/train_config.toml")
    if request.param == "transfer_ae_ae":
        source_task = [
            "train",
            "reconstruction",
            caps_roi_path,
            extract_roi_str,
            labels_path,
            output_dir,
            "-c",
            config_path,
        ]
        target_task = [
            "train",
            "reconstruction",
            caps_roi_path,
            extract_roi_str,
            labels_path,
            target_dir,
            "-c",
            config_path,
            "--transfer_path",
            output_dir,
        ]
    elif request.param == "transfer_ae_cnn":
        source_task = [
            "train",
            "reconstruction",
            caps_roi_path,
            extract_roi_str,
            labels_path,
            output_dir,
            "-c",
            config_path,
        ]
        target_task = [
            "train",
            "classification",
            caps_roi_path,
            extract_roi_str,
            labels_path,
            target_dir,
            "-c",
            config_path,
            "--transfer_path",
            output_dir,
        ]
    elif request.param == "transfer_cnn_cnn":
        source_task = [
            "train",
            "classification",
            caps_roi_path,
            extract_roi_str,
            labels_path,
            output_dir,
            "-c",
            config_path,
        ]
        target_task = [
            "train",
            "classification",
            caps_roi_path,
            extract_roi_str,
            labels_path,
            target_dir,
            "-c",
            config_path,
            "--transfer_path",
            output_dir,
        ]
    elif request.param == "transfer_cnn_multicnn":
        source_task = [
            "train",
            "classification",
            caps_roi_path,
            extract_roi_str,
            labels_path,
            output_dir,
            "-c",
            config_path,
        ]
        target_task = [
            "train",
            "classification",
            caps_roi_path,
            extract_roi_str,
            labels_path,
            target_dir,
            "-c",
            config_path,
            "--transfer_path",
            output_dir,
            "--multi_network",
        ]
    else:
        raise NotImplementedError(f"Test {request.param} is not implemented.")

    return source_task, target_task


def test_transfer(cli_commands):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    source_task, target_task = cli_commands
    flag_source = not os.system("clinicadl " + " ".join(source_task))
    flag_target = not os.system("clinicadl " + " ".join(target_task))
    assert flag_source
    assert flag_target
    shutil.rmtree(output_dir)
    shutil.rmtree(target_dir)
