# coding: utf8
import pathlib
import shutil
from os import system

import pytest

from clinicadl import MapsManager


@pytest.fixture(
    params=[
        "stopped_jobs/stopped_1",
        "stopped_jobs/stopped_2",
        "stopped_jobs/stopped_3",
        "stopped_jobs/stopped_4",
    ]
)
def input_directory(request):
    return request.param


def test_resume(input_directory):
    flag_error = not system(f"clinicadl train resume {input_directory}")
    assert flag_error

    maps_manager = MapsManager(input_directory)
    split_manager = maps_manager._init_split_manager()
    for split in split_manager.split_iterator():
        performances_flag = pathlib.Path(
            "results", f"split-{split}", "best-loss", "train"
        ).exists()
        assert performances_flag
    shutil.rmtree(input_directory)
