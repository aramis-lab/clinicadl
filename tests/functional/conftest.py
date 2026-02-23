from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--ref", action="store")


@pytest.fixture
def ref_data(request) -> Path:
    return Path(request.config.getoption("--ref"))


@pytest.fixture
def caps_dir(ref_data) -> Path:
    return ref_data / "caps"


@pytest.fixture
def split_dir(caps_dir) -> Path:
    return caps_dir / "splits" / "split"


@pytest.fixture
def kfold_dir(split_dir) -> Path:
    return split_dir / "4_fold"
