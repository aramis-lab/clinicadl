# coding: utf8
import json
import os
import shutil
from os import system
from os.path import join
from pathlib import Path

import pytest

from clinicadl import MapsManager
from tests.testing_tools import compare_folders


@pytest.fixture(
    params=[
        "stopped_1",
        "stopped_2",
        "stopped_3",
    ]
)
def test_name(request):
    return request.param


def test_resume(cmdopt, tmp_path, test_name):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "resume" / "in"
    ref_dir = base_dir / "resume" / "ref"
    tmp_out_dir = tmp_path / "resume" / "out"
    tmp_out_dir.mkdir(parents=True)

    shutil.copytree(input_dir / test_name, tmp_out_dir / test_name)
    maps_stopped = tmp_out_dir / test_name

    flag_error = not system(f"clinicadl -vv train resume {maps_stopped}")
    assert flag_error

    maps_manager = MapsManager(maps_stopped)
    split_manager = maps_manager._init_split_manager()
    for split in split_manager.split_iterator():
        performances_flag = (
            maps_stopped / f"split-{split}" / "best-loss" / "train"
        ).exists()
        assert performances_flag

        with open(maps_stopped / "maps.json", "r") as out:
            json_data_out = json.load(out)
        with open(ref_dir / "maps_image_cnn" / "maps.json", "r") as ref:
            json_data_ref = json.load(ref)
        assert json_data_ref == json_data_out
