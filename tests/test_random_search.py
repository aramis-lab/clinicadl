# coding: utf8

import json
import os
import shutil
from os.path import join
from pathlib import Path

import pytest

from tests.testing_tools import compare_folders


# random searxh for ROI with CNN
@pytest.fixture(
    params=[
        "rs_roi_cnn",
    ]
)
def test_name(request):
    return request.param


def test_random_search(cmdopt, tmp_path, test_name):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "randomSearch" / "in"
    ref_dir = base_dir / "randomSearch" / "ref"
    tmp_out_dir = tmp_path / "randomSearch" / "out"
    tmp_out_dir.mkdir(parents=True)

    if test_name == "rs_roi_cnn":
        toml_path = join(input_dir / "random_search.toml")
        generate_input = ["random-search", str(tmp_out_dir), "job-1"]
    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    run_test_random_search(toml_path, generate_input, tmp_out_dir, ref_dir)


def run_test_random_search(toml_path, generate_input, tmp_out_dir, ref_dir):
    if os.path.exists(tmp_out_dir):
        shutil.rmtree(tmp_out_dir)

    # Write random_search.toml file
    os.makedirs(tmp_out_dir, exist_ok=True)
    shutil.copy(toml_path, tmp_out_dir)

    flag_error_generate = not os.system("clinicadl " + " ".join(generate_input))
    performances_flag = os.path.exists(
        tmp_out_dir / "job-1" / "split-0" / "best-loss" / "train"
    )
    assert flag_error_generate
    assert performances_flag

    assert compare_folders(
        tmp_out_dir / "job-1" / "groups",
        ref_dir / "job-1" / "groups",
        tmp_out_dir,
    )
    assert compare_folders(
        tmp_out_dir / "job-1" / "split-0" / "best-loss",
        ref_dir / "job-1" / "split-0" / "best-loss",
        tmp_out_dir,
    )
