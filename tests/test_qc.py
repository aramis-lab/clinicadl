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
    if request.param not in ["t1-linear", "t1-volume"]:
        raise NotImplementedError(
            f"Quality check test on {request.param} is not implemented."
        )

    return request.param


def test_qc(cli_commands):
    input_dir = "data/dataset/OasisCaps_example"
    output_path = "results/quality_check"
    preprocessing = cli_commands

    flag_error = not system(
        f"clinicadl quality-check {preprocessing} {input_dir} {output_path}"
    )
    assert flag_error
    shutil.rmtree(output_path)
