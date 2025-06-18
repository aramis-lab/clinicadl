import pytest

from clinicadl.data.datatypes.modalities import (
    DWI,
    PET,
    Custom,
    Flair,
    T1w,
)


def test_good_custom():
    custom_data = Custom(custom_suffix="example")
    assert custom_data.custom_suffix == "example"
    assert custom_data.modality == "custom"


def test_good_flair():
    flair_data = Flair()
    assert flair_data.modality == "FLAIR"


def test_good_t1():
    t1w_data = T1w()
    assert t1w_data.modality == "T1w"


def test_good_pet():
    pet_data = PET(tracer="18FFBB", reconstruction="nacstat")
    assert pet_data.modality == "pet"
    assert pet_data.tracer == "18FFBB"
    assert pet_data.reconstruction == "nacstat"


def test_good_dwi():
    dwi_data = DWI()
    assert dwi_data.modality == "dwi"


def test_bad_input():
    with pytest.raises(ValueError):
        PET(tracer="abc")  # type: ignore
