# coding: utf8

import os
from os.path import abspath
from pathlib import Path
from typing import List

import pytest

from tests.testing_tools import clean_folder, compare_folder_with_files, compare_folders


@pytest.fixture(params=["random_example", "shepplogan_example", "trivial_example"])
def test_name(request):
    return request.param


def test_generate(cmdopt, tmp_path, test_name):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "generate" / "in"
    ref_dir = base_dir / "generate" / "ref"
    tmp_out_dir = tmp_path / "generate" / "out"
    tmp_out_dir.mkdir(parents=True)

    clean_folder(tmp_out_dir, recreate=True)
    data_caps_folder = str(input_dir / "caps")

    if test_name == "trivial_example":
        output_folder = str(tmp_out_dir / test_name)
        test_input = [
            "generate",
            "trivial",
            data_caps_folder,
            output_folder,
            "--n_subjects",
            "4",
            "--preprocessing",
            "t1-linear",
        ]

    elif test_name == "random_example":
        output_folder = str(tmp_out_dir / test_name)
        test_input = [
            "generate",
            "random",
            data_caps_folder,
            output_folder,
            "--n_subjects",
            "4",
            "--mean",
            "4000",
            "--sigma",
            "1000",
            "--preprocessing",
            "t1-linear",
        ]

    elif test_name == "shepplogan_example":
        n_subjects = 10
        output_folder = str(tmp_out_dir / test_name)
        test_input = [
            "generate",
            "shepplogan",
            output_folder,
            "--n_subjects",
            f"{n_subjects}",
        ]

    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    flag_error = not os.system("clinicadl " + " ".join(test_input))

    assert flag_error
    assert compare_folders(
        str(output_folder), str(ref_dir / test_name), str(tmp_out_dir)
    )
    # compare_folder_with_files(abspath(output_folder), output_reference)
    clean_folder(tmp_out_dir, recreate=True)
