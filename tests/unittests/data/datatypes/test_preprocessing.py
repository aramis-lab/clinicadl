import pytest

from clinicadl.data.datatypes.preprocessing import (
    DWIDTI,
    Custom,
    FlairLinear,
    PETLinear,
    T1Linear,
    get_preprocessing_config,
)


def test_good_custom():
    custom_data = Custom(custom_suffix="example")
    assert custom_data.custom_suffix == "example"
    assert custom_data.modality == "custom"
    assert custom_data.name == "custom"
    assert custom_data.file_type.pattern == "example/sub-*_ses-*_example.nii*"
    assert custom_data.file_type.description == "Custom images with suffix 'example'"
    assert custom_data.file_type.needed_pipeline is None
    assert str(custom_data) == "Custom images with suffix 'example'"
    assert custom_data.tsv_filename == "overview_example.tsv"


def test_good_flair():
    flair_data = FlairLinear(use_uncropped_image=True)
    assert flair_data.modality == "FLAIR"
    assert flair_data.name == "flair-linear"
    assert (
        flair_data.file_type.pattern
        == "flair_linear/sub-*_ses-*_space-MNI152NLin2009cSym_res-1x1x1_FLAIR.nii*"
    )
    assert (
        flair_data.file_type.description
        == "FLAIR images registered to MNI152NLin2009cSym space using Clinica's 'flair-linear' pipeline"
    )
    assert flair_data.file_type.needed_pipeline == "flair-linear"
    assert flair_data.tsv_filename == "overview_flair-linear.tsv"


def test_good_t1():
    t1w_data = T1Linear()
    assert t1w_data.modality == "T1w"
    assert t1w_data.name == "t1-linear"
    assert (
        t1w_data.file_type.pattern
        == "t1_linear/sub-*_ses-*_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii*"
    )
    assert t1w_data.file_type.description == (
        "T1w images registered to MNI152NLin2009cSym space using Clinica's 't1-linear' pipeline, "
        "and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
    )
    assert t1w_data.file_type.needed_pipeline == "t1-linear"
    assert t1w_data.tsv_filename == "overview_t1-linear_cropped.tsv"


def test_good_pet():
    pet_data = PETLinear(tracer="18FFDG", suvr_reference_region="cerebellumPons2")
    assert pet_data.modality == "pet"
    assert pet_data.tracer == "18FFDG"
    assert pet_data.suvr_reference_region == "cerebellumPons2"
    assert pet_data.modality == "pet"
    assert pet_data.name == "pet-linear"
    assert (
        pet_data.file_type.pattern
        == "pet_linear/sub-*_ses-*_trc-18FFDG_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.nii*"
    )
    assert pet_data.file_type.description == (
        "PET images with tracer '18FFDG', registered to MNI152NLin2009cSym space using Clinica's "
        "'pet-linear' pipeline with SUVR reference region 'cerebellumPons2', and cropped "
        "(matrix size 169×208×179, 1 mm isotropic voxels)"
    )
    assert pet_data.file_type.needed_pipeline == "pet-linear"
    assert (
        pet_data.tsv_filename
        == "overview_pet-linear_18FFDG_cerebellumPons2_cropped.tsv"
    )

    pet_data.use_uncropped_image = True
    pet_data.suvr_reference_region = "pons2"
    pet_data.reconstruction = "nacstat"
    assert (
        pet_data.file_type.pattern
        == "pet_linear/sub-*_ses-*_trc-18FFDG_rec-nacstat_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii*"
    )
    assert pet_data.file_type.description == (
        "PET images with tracer '18FFDG' and reconstruction method 'nacstat', registered to MNI152NLin2009cSym space "
        "using Clinica's 'pet-linear' pipeline with SUVR reference region 'pons2'"
    )
    assert pet_data.tsv_filename == "overview_pet-linear_18FFDG_pons2_nacstat.tsv"


def test_good_dwi():
    dwi_data = DWIDTI(measure="FA", space="normalized")
    assert dwi_data.modality == "dwi"
    assert dwi_data.name == "dwi-dti"
    assert dwi_data.measure == "FA"
    assert dwi_data.space == "normalized"
    assert (
        dwi_data.file_type.pattern
        == "dwi/dti_based_processing/normalized_space/sub-*_ses-*_space-MNI152Lin_FA.nii*"
    )
    assert (
        dwi_data.file_type.description
        == "DTI FA images in normalized space, preprocessed with Clinica's 'dwi-dti' pipeline"
    )
    assert dwi_data.file_type.needed_pipeline == "dwi-dti"
    assert dwi_data.tsv_filename == "overview_dwi-dti_FA_normalized.tsv"

    dwi_data.measure = "MD"
    dwi_data.space = "native"
    assert (
        dwi_data.file_type.pattern
        == "dwi/dti_based_processing/native_space/sub-*_ses-*_space-*_MD.nii*"
    )
    assert (
        dwi_data.file_type.description
        == "DTI MD images in native space, preprocessed with Clinica's 'dwi-dti' pipeline"
    )
    assert dwi_data.tsv_filename == "overview_dwi-dti_MD_native.tsv"


@pytest.mark.parametrize(
    "preprocessing,config",
    [
        ("t1-linear", T1Linear),
        ("pet-linear", PETLinear),
        ("custom", Custom),
        ("flair-linear", FlairLinear),
        ("dwi-dti", DWIDTI),
    ],
)
def test_factory(preprocessing, config):
    mandatory_args = {
        "custom_suffix": "abc",
        "measure": "FA",
        "space": "normalized",
        "tracer": "18FFDG",
        "suvr_reference_region": "cerebellumPons2",
    }
    assert isinstance(get_preprocessing_config(preprocessing, **mandatory_args), config)
