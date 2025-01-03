import pytest

from clinicadl.data.datatype.file_type import FileType
from clinicadl.data.datatype.modalities.pet import SUVRReferenceRegions, Tracer
from clinicadl.data.datatype.preprocessing import (
    DWIDTI,
    Custom,
    FlairLinear,
    PETLinear,
    T1Linear,
)
from clinicadl.data.datatype.preprocessing.dti import DTIMeasure, DTISpace
from clinicadl.data.datatype.utils import ImageModality, PreprocessingMethod


def test_good_custom():
    custom_data = Custom(custom_suffix="example")
    assert custom_data.custom_suffix == "example"
    assert custom_data.modality == ImageModality.CUSTOM
    assert custom_data.preprocessing == PreprocessingMethod.CUSTOM
    assert custom_data.file_type.pattern == "custom/*example"
    assert custom_data.file_type.description == "Custom suffix"
    assert custom_data.file_type.needed_pipeline is None


def test_good_flair():
    flair_data = FlairLinear()
    assert flair_data.modality == ImageModality.FLAIR
    assert flair_data.preprocessing == PreprocessingMethod.FLAIR_LINEAR
    assert (
        flair_data.file_type.pattern
        == "flair_linear/*space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_flair.nii.gz"
    )
    assert "Image registered in MNI152NLin2009cSym" in flair_data.file_type.description
    assert flair_data.file_type.needed_pipeline == PreprocessingMethod.FLAIR_LINEAR


def test_good_t1():
    t1w_data = T1Linear()
    assert t1w_data.modality == ImageModality.T1W
    assert t1w_data.preprocessing == PreprocessingMethod.T1_LINEAR
    assert (
        t1w_data.file_type.pattern
        == "t1_linear/*space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"
    )
    assert "Image registered in MNI152NLin2009cSym" in t1w_data.file_type.description
    assert t1w_data.file_type.needed_pipeline == PreprocessingMethod.T1_LINEAR


def test_good_pet():
    pet_data = PETLinear()
    assert pet_data.modality == ImageModality.PET
    assert pet_data.tracer == Tracer.FFDG
    assert pet_data.suvr_reference_region == SUVRReferenceRegions.CEREBELLUMPONS2
    assert pet_data.modality == ImageModality.PET
    assert pet_data.preprocessing == PreprocessingMethod.PET_LINEAR
    assert (
        pet_data.file_type.pattern
        == "pet_linear/*_trc-18FFDG_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.nii.gz"
    )
    assert "Image registered in MNI152NLin2009cSym" in pet_data.file_type.description
    assert pet_data.file_type.needed_pipeline == PreprocessingMethod.PET_LINEAR


def test_good_dwi():
    dwi_data = DWIDTI()
    assert dwi_data.modality == ImageModality.DWI
    assert dwi_data.preprocessing == PreprocessingMethod.DWI_DTI
    assert dwi_data.dti_measure == DTIMeasure.FRACTIONAL_ANISOTROPY
    assert dwi_data.dti_space == DTISpace.ALL
    assert (
        dwi_data.file_type.pattern == "dwi/dti_based_processing/*/*_space-*_FA.nii.gz"
    )
    assert "DTI-based" in dwi_data.file_type.description
    assert dwi_data.file_type.needed_pipeline == PreprocessingMethod.DWI_DTI
