import pytest
import torch
import torchio as tio

from clinicadl.transforms.config import ImplementedTransform, create_transform_config
from clinicadl.transforms.config.intensity_augmentations import (
    RandomBlurConfig,
    RandomNoiseConfig,
)
from clinicadl.transforms.factory import get_transform_config, get_transform_from_config

MANDATORY_ARGS = {
    "masking_method": 1,
    "target_shape": 4,
    "out_max": 1.0,
    "target_multiple": 2,
    "cropping": 1,
    "padding": 1,
    "remapping": {0: 1},
}


def test_get_transform_from_config():
    x = tio.Subject(
        image=tio.ScalarImage(tensor=torch.randn(1, 16, 17, 18)),
        label=tio.LabelMap(tensor=torch.ones(1, 16, 17, 18)),
    )

    # test all transforms
    for transform in ImplementedTransform:
        if transform != ImplementedTransform.ONE_OF:
            config = create_transform_config(transform=transform)(**MANDATORY_ARGS)
            transform, _ = get_transform_from_config(config=config)
            transform(x)

    # test arguments
    config = create_transform_config("CropOrPad")(
        target_shape=4,
        padding_mode=1,
        mask_name=None,
    )
    transform, updated_config = get_transform_from_config(config=config)
    out = transform(x)
    assert isinstance(transform, tio.CropOrPad)
    assert out.label.tensor.shape == (1, 4, 4, 4)
    assert out.label.tensor.shape == (1, 4, 4, 4)

    assert updated_config.name == "CropOrPad"
    assert updated_config.target_shape == 4
    assert updated_config.padding_mode == 1
    assert updated_config.mask_name is None
    assert updated_config.labels is None


def test_get_one_of():
    x = tio.Subject(
        image=tio.ScalarImage(tensor=torch.randn(1, 16, 17, 18)),
        label=tio.LabelMap(tensor=torch.ones(1, 16, 17, 18)),
    )

    config = create_transform_config("OneOf")(
        transforms=[
            get_transform_config("RandomBlur"),
            get_transform_config("RandomNoise"),
        ],
        probabilities=[1, 9],
    )
    transform, _ = get_transform_from_config(config=config)
    transform_dict = transform.transforms_dict
    assert [type(t) for t in transform_dict.keys()] == [tio.RandomBlur, tio.RandomNoise]
    assert list(transform_dict.values()) == [0.1, 0.9]
    transform(x)


def test_get_transform_config():
    config = get_transform_config("CropOrPad", target_shape=4, padding_mode=1)
    assert config.name == "CropOrPad"
    assert config.target_shape == 4
    assert config.padding_mode == 1
    assert config.mask_name is None
    assert config.labels is None

    with pytest.raises(ValueError):
        get_transform_config("abc", **MANDATORY_ARGS)

    config = get_transform_config("NanRemoval", nan=1)
    assert config.name == "NanRemoval"
    assert config.nan == 1
    assert config.posinf is None
    assert config.neginf is None
