from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--ref", action="store")


@pytest.fixture
def ref_data(request) -> Path:
    return Path(request.config.getoption("--ref"))


@pytest.fixture
def bids_dir(ref_data) -> Path:
    return ref_data / "bids"


@pytest.fixture
def metadata_tsv(bids_dir) -> Path:
    return bids_dir / "metadata.tsv"


@pytest.fixture
def split_dir(bids_dir) -> Path:
    return bids_dir / "splits" / "split"


@pytest.fixture
def kfold_dir(split_dir) -> Path:
    return split_dir / "4_fold"
