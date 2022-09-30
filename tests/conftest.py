# coding: utf8

"""
    This file contains a set of functional tests designed to check the correct execution of the pipeline and the
    different functions available in ClinicaDL
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--working_directory", action="store", help="Working directory for tests"
    )


@pytest.fixture
def cmdopt(request):
    config_param = {}
    config_param["wd"] = request.config.getoption("--working_directory")
    return config_param
