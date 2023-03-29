# coding: utf8

import os
from os.path import abspath
from pathlib import Path
from typing import List

import pytest

from tests.testing_tools import clean_folder, compare_folders


@pytest.fixture(params=["random_example", "trivial_example", "shepplogan_example"])
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
        output_folder = tmp_out_dir / test_name
        test_input = [
            "generate",
            "trivial",
            data_caps_folder,
            str(output_folder),
            "--n_subjects",
            "4",
            "--preprocessing",
            "t1-linear",
        ]

    elif test_name == "random_example":
        output_folder = tmp_out_dir / test_name
        test_input = [
            "generate",
            "random",
            data_caps_folder,
            str(output_folder),
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
        output_folder = tmp_out_dir / test_name
        test_input = [
            "generate",
            "shepplogan",
            str(output_folder),
            "--n_subjects",
            f"{n_subjects}",
        ]

    else:
        raise NotImplementedError(f"Test {test_name} is not implemented.")

    flag_error = not os.system("clinicadl " + " ".join(test_input))

    assert flag_error

    if test_name == "shepplogan_example":
        file = list((output_folder / "tensor_extraction").iterdir())
        old_name = output_folder / "tensor_extraction" / file[0]
        new_name = output_folder / "tensor_extraction" / "extract_test.json"
        old_name.rename(new_name)

    assert compare_folders(output_folder, ref_dir / test_name, tmp_out_dir)
