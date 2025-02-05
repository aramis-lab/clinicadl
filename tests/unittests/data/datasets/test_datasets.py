from pathlib import Path

import pytest
import torchio as tio

from clinicadl.data.datasets import CapsDataset
from clinicadl.data.datatype.preprocessing import PETLinear, T1Linear
from clinicadl.transforms import Transforms
from clinicadl.utils.exceptions import (
    ClinicaDLArgumentError,
    ClinicaDLCAPSError,
    ClinicaDLConfigurationError,
    ClinicaDLTSVError,
)

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"
caps_dir = Path(__file__).parents[1] / "ressources" / "caps_example"
bids_dir = Path(__file__).parents[1] / "ressources" / "bids_example"


def test_good_caps_dataset():
    preprocessing = T1Linear()

    transforms = Transforms(image_transforms=[tio.RescaleIntensity()])

    caps_dataset = CapsDataset(
        caps_directory=caps_dir, preprocessing=preprocessing, transforms=transforms
    )
    assert caps_dataset.caps_reader.input_directory == caps_dir
    assert caps_dataset.caps_reader.subject_directory == caps_dir / "subjects"

    assert caps_dataset.eval_mode is False
    assert caps_dataset.preprocessing == preprocessing
    assert caps_dataset.extraction == transforms.extraction
    assert isinstance(caps_dataset.image_transform.transforms[0], tio.RescaleIntensity)
    assert caps_dataset.image_augmentation.transforms == []
    assert caps_dataset.sample_transform.transforms == []
    assert caps_dataset.sample_augmentation.transforms == []
    assert caps_dataset.samples_per_image == 1
    assert {PARTICIPANT_ID, SESSION_ID}.issubset(set(caps_dataset.df.columns.values))
    assert len(caps_dataset.df) == 4
    assert len(caps_dataset) == 4
    assert caps_dataset._get_session(1) == "ses-M006"
    assert caps_dataset._get_meta_data(1) == ("sub-000", "ses-M006", 1, 0)
    assert caps_dataset._get_participant(1) == "sub-000"
    assert caps_dataset._get_session(2) == "ses-M000"

    image_sample = caps_dataset[0]

    assert image_sample.participant_id == "sub-000"
    assert image_sample.session_id == "ses-M000"
    assert image_sample.sample.shape == (1, 169, 208, 179)
    assert str(image_sample.image_path) == str(
        caps_dir
        / "subjects"
        / "sub-000"
        / "ses-M000"
        / "t1_linear"
        / "sub-000_ses-M000_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz"
    )
    assert image_sample.label is None

    caps_dataset.eval()
    assert caps_dataset.eval_mode is True


def test_bad_caps_dataset():
    preprocessing = T1Linear()
    transforms = Transforms()

    with pytest.raises(ClinicaDLArgumentError):
        CapsDataset(
            caps_directory="./cpas", preprocessing=preprocessing, transforms=transforms
        )

    with pytest.raises(ClinicaDLCAPSError):
        CapsDataset(
            caps_directory=bids_dir, preprocessing=preprocessing, transforms=transforms
        )

    with pytest.raises(ClinicaDLTSVError):
        CapsDataset(
            caps_directory=caps_dir,
            preprocessing=preprocessing,
            transforms=transforms,
            data="test.tsv",
        )

    with pytest.raises(ClinicaDLConfigurationError):
        CapsDataset(
            caps_directory=caps_dir,
            preprocessing=PETLinear(),
            transforms=transforms,
        )

    caps_dataset = CapsDataset(
        caps_directory=caps_dir, preprocessing=preprocessing, transforms=transforms
    )
    with pytest.raises(ValueError):
        caps_dataset[-1]

    with pytest.raises(IndexError):
        caps_dataset[10]
