from pathlib import Path

import pytest

from clinicadl.data.datatype.preprocessing import PETLinear, T1Linear
from clinicadl.data.readers import CapsReader
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
)

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"
caps_dir = Path(__file__).parents[2] / "resources" / "caps_example"
bids_dir = Path(__file__).parents[2] / "resources" / "bids_example"


def test_good_caps_reader():
    caps_reader = CapsReader(caps_dir)
    subject_dir = caps_dir / "subjects" / "sub-000" / "ses-M000" / "t1_linear"

    assert caps_reader.input_directory == caps_dir
    assert caps_reader.subject_directory == caps_dir / "subjects"

    # get_preprocessing_folder
    assert (
        caps_reader.get_preprocessing_folder("sub-000", "ses-M000", "t1-linear")
        == subject_dir
    )

    # get_participant_path
    assert (
        caps_reader.get_participant_path("sub-000") == caps_dir / "subjects" / "sub-000"
    )

    # path_to_tensor
    assert (
        caps_reader.path_to_tensor(
            subject_dir
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1W.nii.gz"
        )
        == subject_dir
        / "tensors"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1W.pt"
    )
    assert (
        caps_reader.path_to_tensor(caps_dir / "masks" / "leftHippocampus.nii")
        == caps_dir / "masks" / "tensors" / "leftHippocampus.pt"
    )

    # get_tensor_path
    assert (
        caps_reader.get_tensor_path(
            "sub-000", "ses-M000", T1Linear(use_uncropped_image=True)
        )
        == caps_dir
        / subject_dir
        / "tensors"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    )

    # get_image_path
    assert (
        caps_reader.get_image_path(
            "sub-999",
            "ses-M099",
            PETLinear(tracer="18FAV45", suvr_reference_region="pons2"),
        )
        == caps_dir
        / "subjects"
        / "sub-999"
        / "ses-M099"
        / "pet_linear"
        / "sub-999_ses-M099_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_pet.nii.gz"
    )

    # get_common_mask_path
    assert (
        caps_reader.get_common_mask_path("leftHippocampus.nii.gz")
        == caps_dir / "masks" / "leftHippocampus.nii.gz"
    )
    assert (
        caps_reader.get_common_mask_path("leftHippocampus.pt")
        == caps_dir / "masks" / "tensors" / "leftHippocampus.pt"
    )
    assert caps_reader.tensor_conversion_json_dir == caps_dir / "tensor_conversion"

    # check_preprocessing
    with pytest.raises(ClinicaDLCAPSError):
        caps_reader.check_preprocessing(
            [("sub-000", "ses-M003")],
            PETLinear(
                tracer="18FFDG", suvr_reference_region="pons2", use_uncropped_image=True
            ),
        )
    caps_reader.check_preprocessing(
        [("sub-000", "ses-M003"), ("sub-000", "ses-M000")],
        PETLinear(
            tracer="18FAV45", suvr_reference_region="pons2", use_uncropped_image=True
        ),
    )


def test_bad_caps_reader():
    with pytest.raises(ClinicaDLArgumentError):
        CapsReader("ddd")

    assert bids_dir.is_dir()
    with pytest.raises(ClinicaDLCAPSError):
        CapsReader(bids_dir)
