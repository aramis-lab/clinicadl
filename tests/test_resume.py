# coding: utf8
import os
import pathlib
import shutil
from os import system
from os.path import join

import pytest

from clinicadl import MapsManager

root = "/network/lustre/iss02/aramis/projects/clinicadl/data/"


@pytest.fixture(
    params=[
        join(root, "resume/in/stopped_1"),
        join(root, "resume/in/stopped_2"),
        join(root, "resume/in/stopped_3"),
        join(root, "resume/in/stopped_4"),
    ]
)
def input_directory(request):
    return request.param


def test_resume(input_directory):
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
