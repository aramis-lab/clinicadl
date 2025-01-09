import pytest
from pydantic import ValidationError

from clinicadl.data.datatype.modalities import DWI, PET, Custom, Flair, T1w
from clinicadl.data.datatype.modalities.pet import SUVRReferenceRegions, Tracer
from clinicadl.data.datatype.utils import ImageModality


def test_good_custom():
    custom_data = Custom(custom_suffix="example")
    assert custom_data.custom_suffix == "example"
    assert custom_data.modality == ImageModality.CUSTOM

    custom_data = Custom()
    assert custom_data.custom_suffix == ""
    assert custom_data.modality == ImageModality.CUSTOM


def test_good_flair():
    flair_data = Flair()
    assert flair_data.modality == ImageModality.FLAIR


def test_good_t1():
    t1w_data = T1w()
    assert t1w_data.modality == ImageModality.T1W


def test_good_pet():
    pet_data = PET()
    assert pet_data.modality == ImageModality.PET
    assert pet_data.tracer == Tracer.FFDG
    assert pet_data.suvr_reference_region == SUVRReferenceRegions.CEREBELLUMPONS2


def test_good_dwi():
    dwi_data = DWI()
    assert dwi_data.modality == ImageModality.DWI


def test_bad_filetype():
    with pytest.raises(ValueError):
        PET(tracer="false")  # type: ignore
