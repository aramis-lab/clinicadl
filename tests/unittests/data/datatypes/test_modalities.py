import pytest

from clinicadl.data.datatypes.modalities import (
    PET,
    Flair,
    T1w,
)


def test_good_flair():
    flair_data = Flair()
    assert flair_data._modality == "FLAIR"


def test_good_t1():
    t1w_data = T1w()
    assert t1w_data._modality == "T1w"


def test_good_pet():
    pet_data = PET(tracer="18FFBB", reconstruction="nacstat")
    assert pet_data._modality == "pet"
    assert pet_data.tracer == "18FFBB"
    assert pet_data.reconstruction == "nacstat"


def test_bad_input():
    with pytest.raises(ValueError):
        PET(tracer="abc")  # type: ignore
