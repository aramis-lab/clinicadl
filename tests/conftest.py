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
        "--no-gpu",
        action="store_true",
        help="""To run tests on cpu. Default is False.
            To use carefully, only to run tests locally. Should not be used in final CI tests.
            Concretely, the tests won't fail if gpu option is false in the output MAPS whereas
            it is true in the reference MAPS.""",
    )
    parser.addoption(
        "--adapt-base-dir",
        action="store_true",
        help="""To virtually change the base directory in the paths stored in the MAPS of the CI data.
            Default is False.
            To use carefully, only to run tests locally. Should not be used in final CI tests.
            Concretely, the tests won't fail if only the base directories differ in the paths stored
            in the output and reference MAPS.""",
    )


@pytest.fixture
def cmdopt(request):
    config_param = {}
    config_param["input"] = request.config.getoption("--input_data_directory")
    config_param["no-gpu"] = request.config.getoption("--no-gpu")
    config_param["adapt-base-dir"] = request.config.getoption("--adapt-base-dir")
    return config_param
