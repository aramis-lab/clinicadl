# coding: utf8

import os
import shutil
from os.path import join
from pathlib import Path

import pytest

from .testing_tools import compare_folders, modify_toml


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

    if os.path.exists(tmp_out_dir):
        shutil.rmtree(tmp_out_dir)
    tmp_out_dir.mkdir(parents=True)

    if test_name == "rs_roi_cnn":
        toml_path = join(input_dir / "random_search.toml")
        generate_input = ["random-search", str(tmp_out_dir), "job-1"]
    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    # Write random_search.toml file
    shutil.copy(toml_path, tmp_out_dir)

    if cmdopt["no-gpu"] or cmdopt["adapt-base-dir"]:
        modify_toml(
            toml_path=tmp_out_dir / "random_search.toml",
            base_dir=base_dir,
            no_gpu=cmdopt["no-gpu"],
            adapt_base_dir=cmdopt["adapt-base-dir"],
        )

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
