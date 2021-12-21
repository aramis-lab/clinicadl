import shutil
from os import system

import pytest


@pytest.fixture(
    params=[
        "t1-linear",
        # "t1-volume"
    ]
)
def cli_commands(request):
    if request.param == "t1-linear":
        options = "--no-gpu"
    elif request.param == "t1-volume":
        options = ""
    else:
        raise NotImplementedError(
            f"Quality check test on {request.param} is not implemented."
        )

    return request.param, options


def test_qc(cli_commands):
    input_dir = "data/dataset/OasisCaps_example"
    output_path = "results/quality_check"
    preprocessing, options = cli_commands

    flag_error = not system(
        f"clinicadl quality-check {preprocessing} {input_dir} {output_path} {options}"
    )
    assert flag_error
    shutil.rmtree("results")
