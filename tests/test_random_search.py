# coding: utf8

import json
import os
import shutil
from os.path import join

import pytest

name_dir = "job-1"

# root = "/network/lustre/iss02/aramis/projects/clinicadl/data"
root = "/mnt/data/data_CI"
launch_dir = join(root, "randomSearch/out")


@pytest.fixture(
    params=[
        "rs_image_cnn",
    ]
)
def cli_commands(request):

    if request.param == "rs_image_cnn":
        toml_path = join(root, "randomSearch/in/random_search.toml")
        generate_input = ["random-search", launch_dir, name_dir]
    else:
        raise NotImplementedError(f"Test {request.param} is not implemented.")

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
        os.path.join(launch_dir, name_dir, "split-0", "best-loss", "train")
    )
    assert flag_error_generate
    assert performances_flag
    shutil.rmtree(launch_dir)
