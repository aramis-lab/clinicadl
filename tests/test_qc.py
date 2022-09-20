import shutil
from os import system

import pytest


@pytest.fixture(params=["t1-linear", "t1-volume"])
def cli_commands(request):
    if request.param == "t1-linear":
        test_input = [
            "t1-linear",
            "data/dataset/caps",
            "QC/quality_check.tsv",
            "--no-gpu",
        ]
    elif request.param == "t1-volume":
        test_input = [
            "t1-volume",
            "data/dataset/caps_T1V",
            "QC/out/quality_check_T1V.tsv",
            "adni2021",
        ]
    else:
        raise NotImplementedError(
            f"Quality check test on {request.param} is not implemented."
        )

    return test_input


def test_qc(cli_commands):
    flag_error = not system(f"clinicadl quality-check " + " ".join(cli_commands))
    assert flag_error
    shutil.rmtree("QC/out")
