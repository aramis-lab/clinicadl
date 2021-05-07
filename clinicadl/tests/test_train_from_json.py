# coding: utf8

import pytest
import os
import shutil


@pytest.fixture(params=[
    'train_roi_cnn'
])
def cli_commands(request):

    if request.param == 'train_roi_cnn':
        command_dict = {
            "mode": "roi",
            "network_type": "cnn",
            "caps_dir": "data/dataset/random_example",
            "preprocessing": "t1-linear",
            "tsv_path": "data/labels_list",
            "model": "Conv4_FC3",

            "epochs": 1,
            "n_splits": 2,
            "split": [0],
        }
    else:
        raise NotImplementedError(
            "Test %s is not implemented." %
            request.param)

    return command_dict


def test_train(cli_commands):
    import json

    json = json.dumps(cli_commands, skipkeys=True, indent=4)
    with open(os.path.join("commandline.json"), "w") as f:
        f.write(json)

    flag_error = not os.system("clinicadl train from_json commandline.json results")
    performances_flag = os.path.exists(
        os.path.join("results", "fold-0", "cnn_classification"))
    assert flag_error
    assert performances_flag
    shutil.rmtree("results")
    os.remove("commandline.json")
