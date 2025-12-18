from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.datatypes import DataType
from clinicadl.data.structures import Sample, Sample2D

CAPS_DIR = Path(__file__).parents[2] / "resources" / "caps_example"


AFFINE = np.diag([1.3, 1.2, 1.1, 1])
IMAGE = tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3), affine=AFFINE)
ISO_IMAGE = tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3), affine=np.eye(4))
DOUBLE_IMAGE = tio.ScalarImage(tensor=torch.randn(2, 3, 3, 3), affine=AFFINE)
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
    assert sample.datatype[0] is DATATYPE
    assert str(sample.image_path[0]) == str(PATH)
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

    sample = Sample(
        image=DOUBLE_IMAGE,
        participant=PARTICIPANT,
        session=SESSION,
        label=LABEL,
        datatype=DATATYPE,
        image_path=PATH,
    )
    assert len(sample.datatype) == 2
    assert len(sample.image_path) == 2

    sample = Sample(
        image=DOUBLE_IMAGE,
        participant=PARTICIPANT,
        session=SESSION,
        label=LABEL,
        datatype=(DATATYPE, DATATYPE),
        image_path=(PATH, PATH),
    )

    with pytest.raises(
        ValidationError,
        match=r"'datatype' has 2 value\(s\) but there are 1 channel\(s\) in the image.",
    ):
        Sample(
            image=IMAGE,
            participant=PARTICIPANT,
            session=SESSION,
            label=LABEL,
            datatype=(DATATYPE, DATATYPE),
            image_path=(PATH,),
        )

    with pytest.raises(
        ValidationError,
        match=r"'image_path' has 2 value\(s\) but there are 1 channel\(s\) in the image.",
    ):
        Sample(
            image=IMAGE,
            participant=PARTICIPANT,
            session=SESSION,
            label=LABEL,
            datatype=(DATATYPE,),
            image_path=(PATH, PATH),
        )

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

    with pytest.raises(RuntimeError):
        Sample(
            image=IMAGE,
            participant=PARTICIPANT,
            session=SESSION,
            label=LABEL,
            datatype=DATATYPE,
            image_path=PATH,
            other_image=ISO_IMAGE,
        )

    sample = Sample(
        image=IMAGE,
        participant=PARTICIPANT,
        session=SESSION,
        label=LABEL,
        datatype=DATATYPE,
        image_path=PATH,
        other_image=ISO_IMAGE,
        check_consistency=False,
    )
    assert sample["other_image"].spatial_shape == (3, 3, 3)


def test_sample_2d():
    with pytest.raises(RuntimeError):
        Sample2D(
            image=tio.ScalarImage(tensor=torch.randn(1, 1, 3, 3), affine=AFFINE),
            participant=PARTICIPANT,
            session=SESSION,
            label=LABEL,
            datatype=DATATYPE,
            image_path=PATH,
            sample_position=1,
            squeeze=False,
            slice_direction=0,
        )
    sample = Sample2D(
        image=tio.ScalarImage(tensor=torch.randn(1, 1, 3, 3), affine=AFFINE),
        participant=PARTICIPANT,
        session=SESSION,
        label=LABEL,
        datatype=DATATYPE,
        image_path=PATH,
        sample_position=1,
        squeeze=False,
        slice_direction=0,
        check_consistency=False,
    )
    assert sample.session is SESSION
    assert sample.sample_position == 1
    assert not sample.squeeze
    assert sample.slice_direction == 0
    assert sample.sample_type == "slice"
    assert sample.get_image_tensor("image").shape == (1, 1, 3, 3)
    sample.squeeze = True
    assert sample.get_image_tensor("image").shape == (1, 3, 3)

    with pytest.raises(
        ValidationError,
        match=r"The dimension along 'slice_direction' should be 1. But here got slice_direction=0 and spatial_shape of \(3, 1, 3\)",
    ):
        Sample2D(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 1, 3), affine=AFFINE),
            participant=PARTICIPANT,
            session=SESSION,
            label=LABEL,
            datatype=DATATYPE,
            image_path=PATH,
            sample_position=2,
            squeeze=False,
            slice_direction=0,
            check_consistency=False,
        )
