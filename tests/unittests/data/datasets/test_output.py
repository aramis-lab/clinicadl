from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.datasets.output import Sample, Sample2D
from clinicadl.data.datatypes import DataType

CAPS_DIR = Path(__file__).parents[2] / "resources" / "caps_example"


AFFINE = np.diag([1.3, 1.2, 1.1, 1])
IMAGE = tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3), affine=AFFINE)
AGE = 1
LABEL = (
    CAPS_DIR
    / "subjects"
    / "sub-000"
    / "ses-M000"
    / "t1_linear"
    / "sub-000_ses-M000_space-MNI152NLin2009cSym_res-1x1x1_brain.nii.gz"
)
DATATYPE = DataType.from_folder_and_suffix(folder="abc", suffix="abc")
PATH = Path("abc")
PARTICIPANT = "sub-000"
SESSION = "ses-000"


def test_sample():
    sample = Sample(
        image=IMAGE,
        participant=PARTICIPANT,
        session=SESSION,
        label=LABEL,
        datatype=DATATYPE,
        image_path=PATH,
    )
    assert sample.image is IMAGE
    assert sample["image"] is IMAGE
    assert sample.participant is PARTICIPANT
    assert sample.session is SESSION
    assert isinstance(sample.label, tio.LabelMap)
    assert sample.datatype is DATATYPE
    assert sample.image_path is PATH
    assert sample.sample_type == "image"
    assert sample.sample_position is None

    sample = Sample(
        image=IMAGE,
        participant=PARTICIPANT,
        session=SESSION,
        label=LABEL,
        datatype=DATATYPE,
        image_path=PATH,
        sample_type="patch",
        sample_position=(0, 0, 0),
    )
    assert sample.sample_type == "patch"
    assert sample.sample_position == (0, 0, 0)

    with pytest.raises(ValidationError):
        Sample(
            image=IMAGE,
            participant=PARTICIPANT,
            session=SESSION,
            label=LABEL,
            datatype=DATATYPE,
            image_path=PATH,
            sample_type="patch",
            sample_position=0,
        )

    with pytest.raises(ValidationError):
        Sample(
            image=IMAGE,
            participant=PARTICIPANT,
            session=SESSION,
            label=LABEL,
            datatype=DATATYPE,
            image_path=PATH,
            sample_type="slice",
            sample_position=(0, 0, 0),
        )

    with pytest.raises(ValidationError):
        Sample(
            image=IMAGE,
            participant=PARTICIPANT,
            session=SESSION,
            label=LABEL,
            datatype=DATATYPE,
            image_path=PATH,
            sample_type="image",
            sample_position=0,
        )


def test_sample_2d():
    sample = Sample2D(
        image=IMAGE,
        participant=PARTICIPANT,
        session=SESSION,
        label=LABEL,
        datatype=DATATYPE,
        image_path=PATH,
        sample_position=1,
        squeeze=False,
        slice_direction=0,
    )
    assert sample.session is SESSION
    assert sample.sample_position == 1
    assert not sample.squeeze
    assert sample.slice_direction == 0
    assert sample.sample_type == "slice"
