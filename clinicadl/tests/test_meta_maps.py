# coding: utf8

import os

import pytest


@pytest.fixture(params=["get_loss"])
def cli_commands(request):

    if request.param == "get_loss":
        analysis_input = ["maps-analysis", "data/maps_analysis", "-metric BA"]
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return analysis_input


def test_interpret(cli_commands):
    analysis_input = cli_commands

    analysis_error = not os.system("clinicadl " + " ".join(analysis_input))
    output_path = os.path.join(
        "data", "maps_analysis", "analysis_metric-BA_selection-loss.tsv"
    )
    analysis_flag = os.path.exists(output_path)
    assert analysis_error
    assert analysis_flag

    os.remove(output_path)
