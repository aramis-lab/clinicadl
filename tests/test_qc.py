import shutil
from os import system
from os.path import join

import pytest

# root = "/network/lustre/iss02/aramis/projects/clinicadl/data/"


@pytest.fixture(params=["t1-linear", "t1-volume"])
def cli_commands(request):
    if request.param == "t1-linear":
        test_input = [
            "t1-linear",
            "qualityCheck/in/caps",
            "qualityCheck/out/QC.tsv",
            "--no-gpu",
        ]
    elif request.param == "t1-volume":
        test_input = [
            "t1-volume",
            "qualityCheck/in/caps_T1V",
            "qualityCheck/out/QC_T1V.tsv",
            "Ixi549Space",
        ]
    else:
        raise NotImplementedError(
            f"Quality check test on {request.param} is not implemented."
        )

    return test_input


def test_qc(cli_commands):
    flag_error = not system(f"clinicadl quality-check " + " ".join(cli_commands))
    assert flag_error
    shutil.rmtree("qualityCheck/out")
