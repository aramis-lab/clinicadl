from pathlib import Path

import pandas as pd
import pytest

from clinicadl.data.datatypes.preprocessing import FlairLinear, PETLinear, T1Linear
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

    # get_participant_path
    assert (
        caps_reader.get_participant_path("sub-000") == caps_dir / "subjects" / "sub-000"
    )

    # path_to_tensor
    assert (
        caps_reader.path_to_tensor(
            subject_dir
            / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1W.nii.gz",
            conversion_name="default_t1-linear",
        )
        == subject_dir
        / "tensors"
        / "default_t1-linear"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1W.pt"
    )
    assert (
        caps_reader.path_to_tensor(
            caps_dir / "masks" / "leftHippocampus.nii",
            conversion_name="default_t1-linear",
        )
        == caps_dir / "masks" / "tensors" / "default_t1-linear" / "leftHippocampus.pt"
    )

    # get_tensor_path
    assert (
        caps_reader.get_tensor_path(
            "sub-000",
            "ses-M000",
            T1Linear(use_uncropped_image=True),
            conversion_name="default_t1-linear",
        )
        == caps_dir
        / subject_dir
        / "tensors"
        / "default_t1-linear"
        / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_T1w.pt"
    )

    # get_image_path
    assert (
        caps_reader.get_image_path(
            "sub-100",
            "ses-M000",
            PETLinear(tracer="18FAV45", suvr_reference_region="pons2"),
        )
        == caps_dir
        / "subjects"
        / "sub-100"
        / "ses-M000"
        / "pet_linear"
        / "sub-100_ses-M000_trc-18FAV45_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_suvr-pons2_pet.nii.gz"
    )

    # get_common_mask_path
    assert (
        caps_reader.get_common_mask_path("leftHippocampus.nii.gz")
        == caps_dir / "masks" / "leftHippocampus.nii.gz"
    )
    assert (
        caps_reader.get_common_mask_tensor_path(
            "leftHippocampus.nii.gz", conversion_name="default_t1-linear"
        )
        == caps_dir / "masks" / "tensors" / "default_t1-linear" / "leftHippocampus.pt"
    )
    assert caps_reader.tensor_conversion_json_dir == caps_dir / "tensor_conversion"

    # check_preprocessing
    with pytest.raises(RuntimeError):
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
    with pytest.raises(RuntimeError):
        caps_reader.check_preprocessing(
            [("sub-666", "ses-M666")],
            FlairLinear(
                use_uncropped_image=True,
            ),
        )

    # get_participants_sessions
    true_df = pd.DataFrame.from_dict(
        {
            "participant_id": ["sub-000", "sub-000", "sub-010", "sub-010"],
            "session_id": ["ses-M000", "ses-M003", "ses-M003", "ses-M012"],
        }
    )
    participants_sessions = caps_reader.get_participants_sessions(
        T1Linear(use_uncropped_image=True)
    )
    assert (participants_sessions == true_df).all().all()

    # create_subjects_sessions_tsv
    tsv_path = caps_reader.create_subjects_sessions_tsv(
        T1Linear(use_uncropped_image=True)
    )
    tsv = pd.read_csv(caps_dir / "overview_t1-linear.tsv", sep="\t")
    assert (tsv == true_df).all().all()
    (caps_dir / "overview_t1-linear.tsv").unlink()
    assert tsv_path == str(caps_dir / "overview_t1-linear.tsv")


def test_bad_caps_reader():
    with pytest.raises(ClinicaDLArgumentError):
        CapsReader("ddd")

    assert bids_dir.is_dir()
    with pytest.raises(ClinicaDLCAPSError):
        CapsReader(bids_dir)
