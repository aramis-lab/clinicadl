# coding: utf8

import os
import shutil

import pytest


@pytest.fixture(params=["train_roi_regression"])
def cli_commands(request):

    if request.param == "train_roi_regression":
        command_dict = {
            "network_task": "regression",
            "caps_directory": "data/dataset/random_example",
            "preprocessing": "path/to/preprocessing",
            "tsv_path": "data/labels_list",
            "architecture": "Conv4_FC3",
            "epochs": 1,
            "n_splits": 2,
            "split": [0],
        }
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return command_dict


def test_train(cli_commands):
    import json

    if os.path.exists("results"):
        shutil.rmtree("results")

    json = json.dumps(cli_commands, skipkeys=True, indent=4)
    with open(os.path.join("commandline.json"), "w") as f:
        f.write(json)

    flag_error = not os.system("clinicadl train from_json commandline.json results")
    performances_flag = os.path.exists(
        os.path.join("results", "fold-0", "best-loss", "train")
    )
    assert flag_error
    assert performances_flag
    shutil.rmtree("results")
    os.remove("commandline.json")
