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
    parser.addoption(
        "--simulate-gpu",
        action="store_true",
        help="""To simulate the presence of a gpu on a cpu-only device. Default is False.
            To use carefully, only to run tests locally. Should not be used in final CI tests.
            Concretely, the tests won't fail if gpu option if false in the output MAPS whereas
            it should be true.""",
    )
    parser.addoption(
        "--adapt-base-dir",
        action="store_true",
        help="""To virtually change the base directory in the paths stored in MAPS. Default is False.
            To use carefully, only to run tests locally. Should not be used in final CI tests.
            Concretely, the tests won't fail if only the base directory differs in the paths stored
            in the MAPS.""",
    )


@pytest.fixture
def cmdopt(request):
    config_param = {}
    config_param["input"] = request.config.getoption("--input_data_directory")
    config_param["simulate gpu"] = request.config.getoption("--simulate-gpu")
    config_param["adapt base dir"] = request.config.getoption("--adapt-base-dir")
    return config_param
