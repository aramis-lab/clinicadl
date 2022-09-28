# coding: utf8
import os
import pathlib
import shutil
from os import system
from os.path import join

import pytest

from clinicadl import MapsManager

# root = "/network/lustre/iss02/aramis/projects/clinicadl/data/resume"
root = "/mnt/data/data_CI"


@pytest.fixture(
    params=[
        join(root, "out/stopped_1"),
        join(root, "out/stopped_2"),
        join(root, "out/stopped_3"),
    ]
)
def input_directory(request):
    return request.param


def clean_folder(path, recreate=True):
    from os import makedirs
    from os.path import abspath, exists
    from shutil import rmtree

    abs_path = abspath(path)
    if exists(abs_path):
        rmtree(abs_path)
    if recreate:
        makedirs(abs_path)


def test_resume(input_directory):

    clean_folder(join(root, "out"))

    shutil.copytree(join(root, "in", "stopped_1"), join(root, "out", "stopped_1"))
    shutil.copytree(join(root, "in", "stopped_2"), join(root, "out", "stopped_2"))
    shutil.copytree(join(root, "in", "stopped_3"), join(root, "out", "stopped_3"))

    print(input_directory)
    flag_error = not system(f"clinicadl -vv train resume {input_directory}")
    assert flag_error

    maps_manager = MapsManager(input_directory)
    split_manager = maps_manager._init_split_manager()
    for split in split_manager.split_iterator():
        performances_flag = pathlib.Path(
            input_directory, f"split-{split}", "best-loss", "train"
        ).exists()
        assert performances_flag
    shutil.rmtree(input_directory)
