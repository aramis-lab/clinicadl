import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.transforms.config.spatial_augmentations import (
    RandomAffineConfig,
    RandomAnisotropyConfig,
    RandomElasticDeformationConfig,
    RandomFlipConfig,
)

BAD_INPUTS = [
    ({"axes": 3}, [RandomFlipConfig, RandomAnisotropyConfig]),
    ({"axes": (1, 3)}, [RandomFlipConfig, RandomAnisotropyConfig]),
    ({"flip_probability": 1.1}, RandomFlipConfig),
    ({"scales": -0.5}, RandomAffineConfig),
    ({"scales": (0.5, -0.5)}, RandomAffineConfig),
    ({"scales": (0.5, -0.5, 1, 2, 1, 2)}, RandomAffineConfig),
    ({"degrees": -0.5}, RandomAffineConfig),
    ({"degrees": (0.5, -0.5)}, RandomAffineConfig),
    ({"degrees": (0.5, -0.5, 1, 2, 1, 2)}, RandomAffineConfig),
    ({"translation": -0.5}, RandomAffineConfig),
    ({"translation": (0.5, -0.5)}, RandomAffineConfig),
    ({"translation": (0.5, -0.5, 1, 2, 1, 2)}, RandomAffineConfig),
    ({"isotropic": None}, RandomAffineConfig),
    ({"center": None}, RandomAffineConfig),
    ({"default_pad_value": "abc"}, RandomAffineConfig),
    (
        {"image_interpolation": "abc"},
        [RandomAffineConfig, RandomElasticDeformationConfig, RandomAnisotropyConfig],
    ),
    (
        {"label_interpolation": "abc"},
        [RandomAffineConfig, RandomElasticDeformationConfig],
    ),
    ({"check_shape": None}, RandomAffineConfig),
    ({"num_control_points": 3}, RandomElasticDeformationConfig),
    ({"num_control_points": (3, 5, 6)}, RandomElasticDeformationConfig),
    ({"max_displacement": -0.1}, RandomElasticDeformationConfig),
    ({"max_displacement": (-0.1, 1.0, 3.0)}, RandomElasticDeformationConfig),
    ({"locked_borders": 3}, RandomElasticDeformationConfig),
    ({"downsampling": 0.9}, RandomAnisotropyConfig),
    ({"downsampling": (0.9, 2.0)}, RandomAnisotropyConfig),
]

GOOD_INPUTS = [
    ({"axes": 2, "flip_probability": 0.5}, RandomFlipConfig),
    ({"axes": (0, 1, 2)}, RandomFlipConfig),
    (
        {
            "scales": 0.5,
            "degrees": 0.5,
            "translation": 0.5,
            "isotropic": True,
            "center": "image",
            "default_pad_value": 1.0,
            "check_shape": False,
        },
        RandomAffineConfig,
    ),
    (
        {
            "scales": (-0.5, 0.5),
            "degrees": (-0.5, 0.5),
            "translation": (-0.5, 0.5),
            "isotropic": True,
            "center": "origin",
            "default_pad_value": "minimum",
        },
        RandomAffineConfig,
    ),
    (
        {
            "scales": (-0.5, 0.5, 1, 2, 1, 2),
            "degrees": (-0.5, 0.5, 1, 2, 1, 2),
            "translation": (-0.5, 0.5, 1, 2, 1, 2),
            "default_pad_value": "mean",
        },
        RandomAffineConfig,
    ),
    ({"default_pad_value": "otsu"}, RandomAffineConfig),
    (
        {"num_control_points": 4, "max_displacement": 0, "locked_borders": 0},
        RandomElasticDeformationConfig,
    ),
    (
        {
            "num_control_points": (4, 5, 6),
            "max_displacement": (0, 1.0, 3.0),
            "locked_borders": 1,
        },
        RandomElasticDeformationConfig,
    ),
    ({"locked_borders": 2}, RandomElasticDeformationConfig),
    ({"axes": 2, "downsampling": 1.1}, RandomAnisotropyConfig),
    ({"axes": (0, 1, 2), "downsampling": (1.0, 1.1)}, RandomAnisotropyConfig),
]

X = tio.Subject(
    image=tio.ScalarImage(tensor=torch.randn(1, 16, 17, 18)),
    label=tio.LabelMap(tensor=torch.ones(1, 16, 17, 18)),
)


@pytest.mark.parametrize("args,configs", BAD_INPUTS)
def test_bad_inputs(args, configs):
    if not isinstance(configs, list):
        configs = [configs]
    for config in configs:
        with pytest.raises(ValidationError):
            config(**args)


@pytest.mark.parametrize("args,config", GOOD_INPUTS)
def test_good_inputs(args: dict, config):
    c = config(**args)
    for arg, value in args.items():
        assert getattr(c, arg) == value


@pytest.mark.parametrize(
    "config,transform",
    [
        (RandomAffineConfig, tio.RandomAffine),
        (RandomAnisotropyConfig, tio.RandomAnisotropy),
        (RandomElasticDeformationConfig, tio.RandomElasticDeformation),
        (RandomFlipConfig, tio.RandomFlip),
    ],
)
def test_get_object(config, transform):
    c = config()
    transform_from_config = c.get_object()
    assert isinstance(transform_from_config, transform)
    assert isinstance(transform_from_config(X), tio.Subject)


def test_interpolation():
    modes = [
        "blackman",
        "bspline",
        "cosine",
        "cubic",
        "gaussian",
        "hamming",
        "label_gaussian",
        "lanczos",
        "linear",
        "nearest",
        "welch",
    ]
    for mode in modes:
        for config in [RandomAffineConfig, RandomElasticDeformationConfig]:
            c = config(image_interpolation=mode, label_interpolation=mode)
            assert c.image_interpolation == mode
            assert c.label_interpolation == mode

        c = config(image_interpolation=mode)
        assert c.image_interpolation == mode


def test_axes():
    axes = [0, 1, 2, (0, 1), "LR", "PA", "IS", ("LR", "PA", "IS")]
    for ax in axes:
        c = RandomFlipConfig(axes=ax)
        assert c.axes == ax
