import pytest

from clinicadl.data.datatype.modalities.pet import SUVRReferenceRegions, Tracer
from clinicadl.data.datatype.preprocessing.file_type import FileType
from clinicadl.data.datatype.raw import RawCustom, RawDWI, RawFlair, RawPET, RawT1w
from clinicadl.data.datatype.utils import ImageModality, PreprocessingMethod


def test_good_custom():
    custom_data = RawCustom(custom_suffix="example")
    assert custom_data.custom_suffix == "example"
    assert custom_data.modality == ImageModality.CUSTOM
    assert custom_data.file_type.pattern == "*example"
    assert custom_data.file_type.description == "Custom suffix for raw data"
    assert custom_data.file_type.needed_pipeline is None


def test_good_flair():
    flair_data = RawFlair()
    assert flair_data.modality == ImageModality.FLAIR
    assert flair_data.file_type.pattern == "sub-*_ses-*_flair.nii*"
    assert "FLAIR T2w MRI" in flair_data.file_type.description
    assert flair_data.file_type.needed_pipeline is None


def test_good_t1():
    t1w_data = RawT1w()
    assert t1w_data.modality == ImageModality.T1W
    assert t1w_data.file_type.pattern == "anat/sub-*_ses-*_T1w.nii*"
    assert "T1w MRI" in t1w_data.file_type.description
    assert t1w_data.file_type.needed_pipeline is None


def test_good_pet():
    pet_data = RawPET(
        tracer=Tracer.FAV45, suvr_reference_region=SUVRReferenceRegions.PONS2
    )
    assert pet_data.modality == ImageModality.PET
    assert pet_data.tracer == Tracer.FAV45
    assert pet_data.suvr_reference_region == SUVRReferenceRegions.PONS2
    assert pet_data.file_type.pattern == "pet/*_trc-18FAV45_pet.nii*"
    assert "PET data" in pet_data.file_type.description
    assert pet_data.file_type.needed_pipeline is None


def test_good_dwi():
    dwi_data = RawDWI()
    assert dwi_data.modality == ImageModality.DWI
    assert dwi_data.file_type.pattern == "dwi/sub-*_ses-*_dwi.nii*"
    assert "DWI NIfTI" in dwi_data.file_type.description
    assert dwi_data.file_type.needed_pipeline is None
