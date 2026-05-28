import re
from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.structures import Sample, Sample2D
from clinicadl.io.bids import BidsFileType

AFFINE = np.diag([1.3, 1.2, 1.1, 1])
IMAGE = tio.ScalarImage(tensor=torch.randn(1, 3, 3, 3), affine=AFFINE)
DOUBLE_IMAGE = tio.ScalarImage(tensor=torch.randn(2, 3, 3, 3), affine=AFFINE)
AGE = 1
MASK = tio.LabelMap(tensor=torch.randn(2, 3, 3, 3), affine=AFFINE)
FILE_TYPE = BidsFileType(data_type="abc", suffix="abc")
PATH = Path("abc")
PARTICIPANT = "sub-000"
SESSION = "ses-000"


def test_sample():
    sample = Sample(
        image=IMAGE,
        participant=PARTICIPANT,
        session=SESSION,
        file_type=FILE_TYPE,
        image_path=PATH,
        age=AGE,
        mask=MASK,
    )
    assert sample.image is IMAGE
    assert sample["image"] is IMAGE
    assert sample.participant is PARTICIPANT
    assert sample.session is SESSION
    assert sample["age"] == AGE
    assert sample["mask"] is MASK
    assert sample.file_type[0] is FILE_TYPE
    assert str(sample.image_path[0]) == str(PATH)
    assert sample.sample_type == "image"
    assert sample.sample_position is None

    sample = Sample(
        image=IMAGE,
        participant=PARTICIPANT,
        session=SESSION,
        file_type=FILE_TYPE,
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
        file_type=FILE_TYPE,
        image_path=PATH,
    )
    assert len(sample.file_type) == 2
    assert len(sample.image_path) == 2

    sample = Sample(
        image=DOUBLE_IMAGE,
        participant=PARTICIPANT,
        session=SESSION,
        file_type=(FILE_TYPE, FILE_TYPE),
        image_path=(PATH, PATH),
    )

    with pytest.raises(
        ValidationError,
        match=r"'file_type' has 2 value\(s\) but there are 1 channel\(s\) in the image.",
    ):
        Sample(
            image=IMAGE,
            participant=PARTICIPANT,
            session=SESSION,
            file_type=(FILE_TYPE, FILE_TYPE),
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
            file_type=(FILE_TYPE,),
            image_path=(PATH, PATH),
        )

    with pytest.raises(ValidationError):
        Sample(
            image=IMAGE,
            participant=PARTICIPANT,
            session=SESSION,
            file_type=FILE_TYPE,
            image_path=PATH,
            sample_type="patch",
            sample_position=0,
        )

    with pytest.raises(ValidationError):
        Sample(
            image=IMAGE,
            participant=PARTICIPANT,
            session=SESSION,
            file_type=FILE_TYPE,
            image_path=PATH,
            sample_type="slice",
            sample_position=(0, 0, 0),
        )

    with pytest.raises(ValidationError):
        Sample(
            image=IMAGE,
            participant=PARTICIPANT,
            session=SESSION,
            file_type=FILE_TYPE,
            image_path=PATH,
            sample_type="image",
            sample_position=0,
        )


def test_sample_2d():
    with pytest.raises(
        ValidationError,
        match=r"The dimension along 'slice_direction' should be 1. But here got slice_direction=0 and spatial_shape of \(3, 1, 3\)",
    ):
        Sample2D(
            image=tio.ScalarImage(tensor=torch.randn(1, 3, 1, 3), affine=AFFINE),
            participant=PARTICIPANT,
            session=SESSION,
            file_type=FILE_TYPE,
            image_path=PATH,
            sample_position=2,
            squeeze=False,
            slice_direction=0,
        )

    sample = Sample2D(
        image=tio.ScalarImage(tensor=torch.randn(1, 3, 1, 3), affine=AFFINE),
        participant=PARTICIPANT,
        session=SESSION,
        mask=tio.LabelMap(tensor=torch.randn(2, 3, 1, 3), affine=AFFINE),
        file_type=FILE_TYPE,
        image_path=PATH,
        sample_position=1,
        squeeze=False,
        slice_direction=1,
    )
    assert sample.session is SESSION
    assert sample.sample_position == 1
    assert not sample.squeeze
    assert sample.slice_direction == 1
    assert sample.sample_type == "slice"

    assert sample.get_image_tensor("image").shape == (1, 3, 1, 3)
    assert sample.get_image_tensor("mask").shape == (2, 3, 1, 3)
    sample["squeeze"] = True
    assert sample.get_image_tensor("image").shape == (1, 3, 3)
    assert sample.get_image_tensor("mask").shape == (2, 3, 3)

    # add images
    sample.add_image(sample.get_image_tensor("image"), "image_1")
    sample.add_mask(sample.get_image_tensor("image"), "mask_1")
    assert isinstance(sample["image_1"], tio.ScalarImage)
    assert sample["image_1"].shape == (1, 3, 1, 3)
    assert isinstance(sample["mask_1"], tio.LabelMap)
    assert sample["mask_1"].shape == (1, 3, 1, 3)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "If squeeze=True, a 3D tensor is expected (including one channel dimension). Got: torch.Size([1, 3, 1, 3])"
        ),
    ):
        sample.add_image(sample.image.tensor, "image_2")

    sample["squeeze"] = False
    sample.add_image(sample.image.tensor, "image_1")
    sample.add_mask(sample.image.tensor, "mask_1")

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "If squeeze=False, a 4D tensor is expected (including one channel dimension). Got: torch.Size([1, 3, 3])"
        ),
    ):
        sample.add_mask(sample.image.tensor.squeeze(2), "mask_2")
