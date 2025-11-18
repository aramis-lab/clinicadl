import os
import re
import sys

import pytest

from clinicadl.data.datatypes.preprocessing import (
    DWIDTI,
    FlairLinear,
    PETLinear,
    T1Linear,
)


def test_flair():
    flair_data = FlairLinear(use_uncropped_image=True)
    assert flair_data.key == "flair-linear"
    assert flair_data.name == "FlairLinear"
    assert flair_data.pattern == re.compile(
        os.path.join(
            "flair_linear",
            "sub-.*_ses-.*_space-MNI152NLin2009cSym_res-1x1x1_FLAIR.nii.*",
        )
    )
    assert (
        flair_data.description
        == "FLAIR images registered to MNI152NLin2009cSym space using Clinica's 'flair-linear' pipeline"
    )
    assert flair_data.tsv_filename == "overview_flair-linear.tsv"


def test_t1():
    t1w_data = T1Linear()
    assert t1w_data.key == "t1-linear"
    assert t1w_data.pattern == re.compile(
        os.path.join(
            "t1_linear",
            "sub-.*_ses-.*_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.*",
        )
    )
    assert t1w_data.description == (
        "T1w images registered to MNI152NLin2009cSym space using Clinica's 't1-linear' pipeline, "
        "and cropped (matrix size 169×208×179, 1 mm isotropic voxels)"
    )
    assert t1w_data.tsv_filename == "overview_t1-linear_cropped.tsv"


def test_pet():
    pet_data = PETLinear(tracer="18FFDG", suvr_reference_region="cerebellumPons2")
    assert pet_data.tracer == "18FFDG"
    assert pet_data.suvr_reference_region == "cerebellumPons2"
    assert pet_data.key == "pet-linear"
    assert pet_data.pattern == re.compile(
        os.path.join(
            "pet_linear",
            "sub-.*_ses-.*_trc-18FFDG_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.nii.*",
        )
    )
    assert pet_data.description == (
        "PET images with tracer '18FFDG', registered to MNI152NLin2009cSym space using Clinica's "
        "'pet-linear' pipeline with SUVR reference region 'cerebellumPons2', and cropped "
        "(matrix size 169×208×179, 1 mm isotropic voxels)"
    )
    assert (
        pet_data.tsv_filename
        == "overview_pet-linear_18FFDG_cerebellumPons2_cropped.tsv"
    )

    pet_data.use_uncropped_image = True
    pet_data.suvr_reference_region = "pons2"
    pet_data.reconstruction = "nacstat"
    assert pet_data.pattern == re.compile(
        os.path.join(
            "pet_linear",
            "sub-.*_ses-.*_trc-18FFDG_rec-nacstat_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.*",
        )
    )
    assert pet_data.description == (
        "PET images with tracer '18FFDG' and reconstruction method 'nacstat', registered to MNI152NLin2009cSym space "
        "using Clinica's 'pet-linear' pipeline with SUVR reference region 'pons2'"
    )
    assert pet_data.tsv_filename == "overview_pet-linear_18FFDG_pons2_nacstat.tsv"


def test_dwi():
    dwi_data = DWIDTI(measure="FA", space="normalized")
    assert dwi_data.key == "dwi-dti"
    assert dwi_data.measure == "FA"
    assert dwi_data.space == "normalized"
    assert dwi_data.pattern == re.compile(
        os.path.join(
            "dwi",
            "dti_based_processing",
            "normalized_space",
            "sub-.*_ses-.*_space-MNI152Lin_FA.nii.*",
        )
    )
    assert (
        dwi_data.description
        == "DTI FA images in normalized space, preprocessed with Clinica's 'dwi-dti' pipeline"
    )
    assert dwi_data.tsv_filename == "overview_dwi-dti_FA_normalized.tsv"

    dwi_data.measure = "MD"
    dwi_data.space = "native"
    assert dwi_data.pattern == re.compile(
        os.path.join(
            "dwi",
            "dti_based_processing",
            "native_space",
            "sub-.*_ses-.*_space-.*_MD.nii.*",
        )
    )
    assert (
        dwi_data.description
        == "DTI MD images in native space, preprocessed with Clinica's 'dwi-dti' pipeline"
    )
    assert dwi_data.tsv_filename == "overview_dwi-dti_MD_native.tsv"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="To test with raw path.")
@pytest.mark.parametrize(
    "patterns,preprocessing",
    [
        (
            {
                "t1_linear/sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii": True,
                "t1_linear/sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz": True,
                "t1_linear/sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii": False,
                "sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz": False,
            },
            T1Linear(use_uncropped_image=False),
        ),
        (
            {
                "flair_linear/sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_FLAIR.nii": True,
                "flair_linear/sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_FLAIR.nii.gz": True,
                "flair_linear/sub-000_ses-M000_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_FLAIR.nii": False,
                "flair_linear/sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz": False,
            },
            FlairLinear(use_uncropped_image=True),
        ),
        (
            {
                "pet_linear/sub-000_ses-M000_trc-18FFDG_space-MNI152NLin2009cSym_res-1x1x1_suvr-cerebellumPons2_pet.nii": True,
                "pet_linear/sub-000_ses-M000_trc-18FFDG_space-MNI152NLin2009cSym_res-1x1x1_suvr-cerebellumPons2_pet.nii.gz": True,
                "pet_linear/sub-000_ses-M000_trc-18FFMM_space-MNI152NLin2009cSym_res-1x1x1_suvr-cerebellumPons2_pet.nii": False,
                "pet_linear/sub-000_ses-M000_trc-18FFDG_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-cerebellumPons2_pet.nii.gz": False,
                "pet_linear/sub-000_ses-M000_trc-18FFDG_space-MNI152NLin2009cSym_res-1x1x1_suvr-pons2_pet.nii.gz": False,
            },
            PETLinear(
                tracer="18FFDG",
                suvr_reference_region="cerebellumPons2",
                use_uncropped_image=True,
            ),
        ),
        (
            {
                "dwi/dti_based_processing/native_space/sub-000_ses-M000_space-MNI152Lin_FA.nii": True,
                "dwi/dti_based_processing/native_space/sub-000_ses-M000_space-abc_FA.nii.gz": True,
                "dwi/dti_based_processing/sub-000_ses-M000_space-MNI152Lin_FA.nii": False,
                "dwi/dti_based_processing/native_space/sub-000_ses-M000_space-MNI152Lin_MD.nii.gz": False,
            },
            DWIDTI(space="native", measure="FA"),
        ),
        (
            {
                "dwi/dti_based_processing/normalized_space/sub-000_ses-M000_space-MNI152Lin_FA.nii": True,
                "dwi/dti_based_processing/normalized_space/sub-000_ses-M000_space-abc_FA.nii.gz": False,
            },
            DWIDTI(space="normalized", measure="FA"),
        ),
    ],
)
def test_patterns(patterns, preprocessing):
    for pattern, match in patterns.items():
        assert (preprocessing.pattern.match(pattern) is not None) == match
