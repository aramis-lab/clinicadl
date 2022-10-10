import shutil
from os import system
from os.path import join
from pathlib import Path

import pandas as pd
import pytest

from tests.testing_tools import clean_folder


@pytest.fixture(params=["t1-linear", "t1-volume"])
def test_name(request):
    return request.param


def test_qc(cmdopt, tmp_path, test_name):
    base_dir = Path(cmdopt["input"])
    input_dir = base_dir / "qualityCheck" / "in"
    ref_dir = base_dir / "qualityCheck" / "ref"
    tmp_out_dir = tmp_path / "qualityCheck" / "out"
    tmp_out_dir.mkdir(parents=True)

    clean_folder(tmp_out_dir, recreate=True)

    if test_name == "t1-linear":
        out_tsv = str(tmp_out_dir / "QC.tsv")
        test_input = [
            "t1-linear",
            str(input_dir / "caps"),
            out_tsv,
            "--no-gpu",
        ]

    elif test_name == "t1-volume":
        out_dir = str(tmp_out_dir / "QC_T1V.tsv")
        test_input = [
            "t1-volume",
            str(input_dir / "caps_T1V"),
            out_dir,
            "Ixi549Space",
        ]
    else:
        raise NotImplementedError(
            f"Quality check test on {test_name} is not implemented."
        )

    run_test_qc(test_input)


def run_test_qc(test_input):
    flag_error = not system(f"clinicadl quality-check " + " ".join(test_input))
    assert flag_error
