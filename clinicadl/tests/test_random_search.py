# coding: utf8

import pytest
import os
import shutil
import json

launch_dir = "results"
name_dir = "job-1"


@pytest.fixture(params=[
    'rs_image_cnn',
])
def cli_commands(request):

    if request.param == 'rs_image_cnn':
        arg_dict = {
            'caps_dir': 'data/dataset/random_example',
            'tsv_path': 'data/labels_list',
            'preprocessing': 't1-linear',
            'diagnoses': ["AD", "CN"],

            'mode': 'image',
            'network_type': 'cnn',
            'epochs': 1,
            'patience': 0,
            'tolerance': 0.0,

            'n_convblocks': [3, 5],
            'first_conv_width': [1, 3],
            'n_fcblocks': [1, 2]
        }
        test_input = [
            'random-search',
            launch_dir,
            name_dir,
            '--n_splits', '2',
            '--split', '0',
            '-cpu'
        ]
    else:
        raise NotImplementedError(
            "Test %s is not implemented." %
            request.param)

    return arg_dict, test_input


def test_random_search(cli_commands):
    arg_dict, test_input = cli_commands

    # Write random_search.json file
    os.makedirs(launch_dir, exist_ok=True)
    json_file = json.dumps(arg_dict, skipkeys=True, indent=4)
    f = open(os.path.join(launch_dir, "random_search.json"), "w")
    f.write(json_file)
    f.close()

    flag_error = not os.system("clinicadl " + " ".join(test_input))
    performances_flag = os.path.exists(
        os.path.join(launch_dir, name_dir, "fold-0", "cnn_classification"))
    assert flag_error
    assert performances_flag
    shutil.rmtree(launch_dir)
