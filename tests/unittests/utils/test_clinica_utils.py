import pytest


@pytest.mark.parametrize(
    "tracer,suvr_reference_region,uncropped_image,expected_pattern",
    [
        (
            "18FFDG",
            "cerebellumPons2",
            True,
            "pet_linear/*_trc-18FFDG_space-MNI152NLin2009cSym_res-1x1x1_suvr-cerebellumPons2_pet.nii.gz",
        ),
        (
            "18FAV45",
            "pons",
            False,
            "pet_linear/*_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons_pet.nii.gz",
        ),
    ],
)
def test_pet_linear_nii(
    tracer, suvr_reference_region, uncropped_image, expected_pattern
):
    from clinicadl.caps_dataset.preprocessing.config import PETPreprocessingConfig
    from clinicadl.caps_dataset.preprocessing.utils import pet_linear_nii
    from clinicadl.utils.clinica_utils import FileType

    config = PETPreprocessingConfig(
        tracer=tracer,
        suvr_reference_region=suvr_reference_region,
        use_uncropped_image=uncropped_image,
    )
    assert pet_linear_nii(config) == FileType(
        description="",
        needed_pipeline="pet-linear",
        pattern=expected_pattern,
    )
