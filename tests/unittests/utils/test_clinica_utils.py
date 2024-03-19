import pytest


@pytest.mark.parametrize(
    "acq_label,suvr_reference_region,uncropped_image,expected_pattern",
    [
        (
            "18FFDG", 
            "cerebellumPons2", 
            True, 
            "pet_linear/*_trc-18FFDG_space-MNI152NLin2009cSym_res-1x1x1_suvr-cerebellumPons2_pet.nii.gz"
        ),
        (
            "18FAV45", 
            "pons", 
            False, 
            "pet_linear/*_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons_pet.nii.gz"
        )
    ] 

)
def test_pet_linear_nii(
    acq_label, suvr_reference_region, uncropped_image, expected_pattern
):
    from clinicadl.utils.clinica_utils import pet_linear_nii

    assert(
        pet_linear_nii(acq_label, suvr_reference_region, uncropped_image)["pattern"] == expected_pattern
    )