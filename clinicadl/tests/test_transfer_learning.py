import os
import shutil

import pytest


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

    if request.param == "transfer_ae_ae":
        source_task = [
            "train",
            "reconstruction",
            "data/dataset/random_example",
            "extract_1629458899.json",
            "data/labels_list",
            "results_source",
            "-c",
            "data/train_config.toml",
        ]
        target_task = [
            "train",
            "reconstruction",
            "data/dataset/random_example",
            "extract_1629458899.json",
            "data/labels_list",
            "results_target",
            "-c",
            "data/train_config.toml",
            "--transfer_path",
            "results_source",
        ]
    elif request.param == "transfer_ae_cnn":
        source_task = [
            "train",
            "reconstruction",
            "data/dataset/random_example",
            "extract_1629458899.json",
            "data/labels_list",
            "results_source",
            "-c",
            "data/train_config.toml",
        ]
        target_task = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629458899.json",
            "data/labels_list",
            "results_target",
            "-c",
            "data/train_config.toml",
            "--transfer_path",
            "results_source",
        ]
    elif request.param == "transfer_cnn_cnn":
        source_task = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629458899.json",
            "data/labels_list",
            "results_source",
            "-c",
            "data/train_config.toml",
        ]
        target_task = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629458899.json",
            "data/labels_list",
            "results_target",
            "-c",
            "data/train_config.toml",
            "--transfer_path",
            "results_source",
        ]
    elif request.param == "transfer_cnn_multicnn":
        source_task = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629458899.json",
            "data/labels_list",
            "results_source",
            "-c",
            "data/train_config.toml",
        ]
        target_task = [
            "train",
            "classification",
            "data/dataset/random_example",
            "extract_1629458899.json",
            "data/labels_list",
            "results_target",
            "-c",
            "data/train_config.toml",
            "--transfer_path",
            "results_source",
            "--multi_network",
        ]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return source_task, target_task


def test_transfer(cli_commands):
    if os.path.exists("results_source"):
        shutil.rmtree("results_source")
    if os.path.exists("results_target"):
        shutil.rmtree("results_target")

    source_task, target_task = cli_commands
    flag_source = not os.system("clinicadl " + " ".join(source_task))
    flag_target = not os.system("clinicadl " + " ".join(target_task))
    assert flag_source
    assert flag_target
    shutil.rmtree("results_source")
    shutil.rmtree("results_target")
