import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.transforms.config.base import OneOfConfig
from clinicadl.transforms.config.intensity_augmentations import (
    RandomGhostingConfig,
    RandomMotionConfig,
)
from clinicadl.transforms.config.spatial_augmentations import RandomFlipConfig

X = tio.Subject(
    image=tio.ScalarImage(tensor=torch.randn(1, 16, 17, 18)),
    label=tio.LabelMap(tensor=torch.ones(1, 16, 17, 18)),
)


def test_include_exlude():
    with pytest.raises(ValidationError):
        RandomMotionConfig(include=["a"], exclude=["b"])
    c = RandomMotionConfig(exclude=["b"])
    assert c.get_object().exclude == ["b"]
    c = RandomMotionConfig(include=["a"])
    assert c.get_object().include == ["a"]


def test_one_of():
    one_of = OneOfConfig(
        transforms=[
            RandomMotionConfig(degrees=1),
            [RandomFlipConfig(axes=1), RandomGhostingConfig(axes=0)],
            [],
        ],
        probabilities=[8, 1, 1],
    )
    assert one_of.transforms[0].degrees == 1
    assert one_of.transforms[1][0].axes == 1
    assert one_of.transforms[1][1].axes == 0
    assert one_of.probabilities == [8.0, 1.0, 1.0]

    transform = one_of.get_object()
    assert isinstance(transform, tio.OneOf)
    transform_dict = transform.transforms_dict
    assert [type(t) for t in transform_dict.keys()] == [
        tio.RandomMotion,
        tio.Compose,
        tio.Compose,
    ]
    assert list(transform_dict.values()) == [0.8, 0.1, 0.1]
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
