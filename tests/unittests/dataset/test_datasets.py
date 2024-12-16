from pathlib import Path

import pytest
import torchio as tio

from clinicadl.dataset.datasets.caps_dataset import CapsDataset
from clinicadl.dataset.preprocessing import PreprocessingPET, PreprocessingT1
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
    preprocessing = PreprocessingT1()
    transforms = Transforms(image_transforms=[tio.RescaleIntensity()])

    caps_dataset = CapsDataset(
        caps_directory=caps_dir, preprocessing=preprocessing, transforms=transforms
    )
    assert caps_dataset.caps_reader.input_directory == caps_dir
    assert caps_dataset.caps_reader.subject_directory == caps_dir / "subjects"

    assert caps_dataset.eval_mode is False
    assert caps_dataset.preprocessing == preprocessing
    assert caps_dataset.extraction == transforms.extraction
    assert caps_dataset.image_transform == tio.Compose([tio.RescaleIntensity()])
    assert caps_dataset.sample_transform == tio.Compose([])
    assert caps_dataset.image_augmentation == tio.Compose([])
    assert caps_dataset.sample_augmentation == tio.Compose([])
    assert caps_dataset.samples_per_image == 1
    assert {PARTICIPANT_ID, SESSION_ID}.issubset(set(caps_dataset.df.columns.values))
    assert len(caps_dataset.df) == 4
    assert len(caps_dataset) == 4
    assert caps_dataset._get_session(1) == "ses-M006"
    assert caps_dataset._get_meta_data(1) == ("sub-000", "ses-M006", 1, 0)
    assert caps_dataset._get_participant(1) == "sub-000"
    assert caps_dataset._get_session(2) == "ses-M000"

    image_sample = caps_dataset[0]

    assert image_sample.participant_id == caps_dataset._get_participant(0)
    assert image_sample.session_id == caps_dataset._get_session(0)
    assert image_sample.sample.shape == caps_dataset._get_full_image(0)[0].shape
    assert image_sample.image_path == caps_dataset._get_meta_data(0)[2]
    assert image_sample.label is None

    caps_dataset.eval()
    assert caps_dataset.eval_mode is True


def test_bad_caps_dataset():
    preprocessing = PreprocessingT1()
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
        preprocessing_pet = PreprocessingPET()
        CapsDataset(
            caps_directory=caps_dir,
            preprocessing=PreprocessingPET(),
            transforms=transforms,
        )

    caps_dataset = CapsDataset(
        caps_directory=caps_dir, preprocessing=preprocessing, transforms=transforms
    )
    with pytest.raises(ValueError):
        caps_dataset[-1]

    with pytest.raises(IndexError):
        caps_dataset[10]
