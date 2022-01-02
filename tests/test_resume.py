# coding: utf8
import os
import pathlib
import shutil
from os import system

import pytest

from clinicadl import MapsManager


@pytest.fixture(
    params=[
        "data/stopped_jobs/stopped_1",
        "data/stopped_jobs/stopped_2",
        "data/stopped_jobs/stopped_3",
        "data/stopped_jobs/stopped_4",
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
