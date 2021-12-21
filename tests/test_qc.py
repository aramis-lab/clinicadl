import pytest


@pytest.fixture(params=["t1-linear", "t1-volume"])
def cli_commands(request):
    if request.param == "t1-linear":
        input_dir = "data/dataset/OasisCaps_example"
    elif request.param == "t1-volume":
        input_dir = "t1-volume"
    else:
        raise NotImplementedError("Test %s is not implemented." % request.param)

    return input_dir, request.param


def test_qc(cli_commands):
    pass
