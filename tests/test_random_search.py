# coding: utf8

import json
import os
import shutil

import pytest

launch_dir = "results"
name_dir = "job-1"


@pytest.fixture(
    params=[
        "rs_image_cnn",
    ]
)
def cli_commands(request):

    if request.param == "rs_image_cnn":
        toml_path = "data/random_search.toml"
        generate_input = ["random-search", launch_dir, name_dir]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return toml_path, generate_input


def test_random_search(cli_commands):
    toml_path, generate_input = cli_commands

    if os.path.exists(launch_dir):
        shutil.rmtree(launch_dir)

    # Write random_search.toml file
    os.makedirs(launch_dir, exist_ok=True)
    shutil.copy(toml_path, launch_dir)

    flag_error_generate = not os.system("clinicadl " + " ".join(generate_input))
    performances_flag = os.path.exists(
        os.path.join(launch_dir, name_dir, "fold-0", "best-loss", "train")
    )
    assert flag_error_generate
    assert performances_flag
    shutil.rmtree(launch_dir)
