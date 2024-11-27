from pathlib import Path

import pytest
import torchio.transforms as transforms
from pydantic import ValidationError

from clinicadl.dataset.readers import CapsReader
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
    ClinicaDLConfigurationError,
)


def test_caps_reader():
    with pytest.raises(ClinicaDLArgumentError):
        CapsReader(caps_directory=Path("path/that/not/exist/caps"))
    with pytest.raises(ClinicaDLArgumentError):
        CapsReader(Path(__file__).parents[1] / "ressources" / "bids_exemple")

    caps_reader = CapsReader(Path(__file__).parents[1] / "ressources" / "caps_example")

    assert (
        caps_reader.input_directory
        == Path(__file__).parents[1] / "ressources" / "caps_example"
    )


def test_caps_dataset():
    caps_reader = CapsReader(Path(__file__).parents[1] / "ressources" / "caps_example")
