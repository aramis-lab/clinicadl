from pathlib import Path

import pytest

from clinicadl.data.datatype.preprocessing import PreprocessingMethod, T1Linear
from clinicadl.data.readers import CapsReader
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
)

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"
caps_dir = Path(__file__).parents[1] / "ressources" / "caps_example"
bids_dir = Path(__file__).parents[1] / "ressources" / "bids_example"


def test_good_caps_reader():
    caps_reader = CapsReader(caps_dir)

    assert caps_reader.input_directory == caps_dir
    assert caps_reader.subject_directory == caps_dir / "subjects"
    assert str(caps_reader) == f"CAPS reader for {caps_dir}"
    assert (
        caps_reader.get_preprocessing_folder(
            "sub-000", "ses-M000", PreprocessingMethod.T1_LINEAR
        )
        == caps_dir / "subjects" / "sub-000" / "ses-M000" / "t1_linear"
    )
    assert (
        caps_reader.get_participant_path("sub-000") == caps_dir / "subjects" / "sub-000"
    )
    assert (
        caps_reader.get_tensor_dir("sub-000", "ses-M000", T1Linear())
        == caps_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "deeplearning_prepare_data"
        / "image_based"
        / "t1_linear"
    )
    assert (
        caps_reader.get_tensor_path("sub-000", "ses-M000", T1Linear())
        == caps_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "deeplearning_prepare_data"
        / "image_based"
        / "t1_linear"
        / "sub-000_ses-M000_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.pt"
    )
    assert (
        caps_reader.get_image_path("sub-000", "ses-M000", T1Linear())
        == caps_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "sub-000_ses-M000_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"
    )


def test_bad_caps_reader():
    with pytest.raises(ClinicaDLArgumentError):
        CapsReader("ddd")

    with pytest.raises(ClinicaDLCAPSError):
        CapsReader(bids_dir)
