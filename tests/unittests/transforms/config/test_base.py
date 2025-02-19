import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.transforms.config.base import OneOfConfig
from clinicadl.transforms.config.intensity_augmentations import RandomMotionConfig
from clinicadl.transforms.config.spatial_augmentations import RandomFlipConfig


def test_one_of():
    x = tio.Subject(
        image=tio.ScalarImage(tensor=torch.randn(1, 16, 17, 18)),
        label=tio.LabelMap(tensor=torch.ones(1, 16, 17, 18)),
    )

    one_of = OneOfConfig(
        transforms=[
            RandomMotionConfig(degrees=1),
            RandomFlipConfig(axes=1),
        ],
        probabilities=[9, 1],
    )
    assert [type(transform) for transform in one_of.transforms] == [
        RandomMotionConfig,
        RandomFlipConfig,
    ]
    assert one_of.transforms[0].degrees == 1
    assert one_of.transforms[1].axes == 1
    assert one_of.probabilities == [9, 1]

    transform = one_of.get_object()
    assert isinstance(transform, tio.OneOf)
    transform_dict = transform.transforms_dict
    assert [type(t) for t in transform_dict.keys()] == [
        tio.RandomMotion,
        tio.RandomFlip,
    ]
    assert list(transform_dict.values()) == [0.9, 0.1]
    transform(x)

    one_of = OneOfConfig(
        transforms=[
            RandomMotionConfig(degrees=1),
            RandomFlipConfig(axes=1),
        ],
    )
    assert one_of.probabilities == [1 / 2, 1 / 2]

    with pytest.raises(ValidationError):
        OneOfConfig(
            transforms=[
                RandomMotionConfig(degrees=1),
                RandomFlipConfig(axes=1),
            ],
            probabilities=[10, -2],
        )

    with pytest.raises(ValidationError):
        OneOfConfig(
            transforms=[
                RandomMotionConfig(degrees=1),
                RandomFlipConfig(axes=1),
            ],
            probabilities=[10],
        )
