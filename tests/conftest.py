# coding: utf8

"""
    This file contains a set of functional tests designed to check the correct execution of the pipeline and the
    different functions available in ClinicaDL
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--input_data_directory",
        action="store",
        help="Directory for (only-read) inputs for tests",
    )


@pytest.fixture
def cmdopt(request):
    config_param = {}
    config_param["input"] = request.config.getoption("--input_data_directory")
    return config_param
