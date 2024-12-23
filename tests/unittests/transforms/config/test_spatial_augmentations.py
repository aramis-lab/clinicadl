import pytest
from pydantic import ValidationError

from clinicadl.transforms.config import create_transform_config

BAD_INPUTS = [
    ({"axes": 3}, ["RandomFlip", "RandomAnisotropy"]),
    ({"axes": (1, 3)}, ["RandomFlip", "RandomAnisotropy"]),
    ({"flip_probability": 1.1}, "RandomFlip"),
    ({"scales": -0.5}, "RandomAffine"),
    ({"scales": (0.5, -0.5)}, "RandomAffine"),
    ({"scales": (0.5, -0.5, 1, 2, 1, 2)}, "RandomAffine"),
    ({"degrees": -0.5}, "RandomAffine"),
    ({"degrees": (0.5, -0.5)}, "RandomAffine"),
    ({"degrees": (0.5, -0.5, 1, 2, 1, 2)}, "RandomAffine"),
    ({"translation": -0.5}, "RandomAffine"),
    ({"translation": (0.5, -0.5)}, "RandomAffine"),
    ({"translation": (0.5, -0.5, 1, 2, 1, 2)}, "RandomAffine"),
    ({"isotropic": None}, "RandomAffine"),
    ({"center": None}, "RandomAffine"),
    ({"default_pad_value": "abc"}, "RandomAffine"),
    (
        {"image_interpolation": "abc"},
        ["RandomAffine", "RandomElasticDeformation", "RandomAnisotropy"],
    ),
    ({"label_interpolation": "abc"}, ["RandomAffine", "RandomElasticDeformation"]),
    ({"check_shape": None}, "RandomAffine"),
    ({"num_control_points": 3}, "RandomElasticDeformation"),
    ({"num_control_points": (3, 5, 6)}, "RandomElasticDeformation"),
    ({"max_displacement": -0.1}, "RandomElasticDeformation"),
    ({"max_displacement": (-0.1, 1.0, 3.0)}, "RandomElasticDeformation"),
    ({"locked_borders": 3}, "RandomElasticDeformation"),
    ({"downsampling": 0.9}, "RandomAnisotropy"),
    ({"downsampling": (0.9, 2.0)}, "RandomAnisotropy"),
]

GOOD_INPUTS = [
    ({"axes": 2, "flip_probability": 0.5}, "RandomFlip"),
    ({"axes": (0, 1, 2)}, "RandomFlip"),
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
        "RandomAffine",
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
        "RandomAffine",
    ),
    (
        {
            "scales": (-0.5, 0.5, 1, 2, 1, 2),
            "degrees": (-0.5, 0.5, 1, 2, 1, 2),
            "translation": (-0.5, 0.5, 1, 2, 1, 2),
            "default_pad_value": "mean",
        },
        "RandomAffine",
    ),
    ({"default_pad_value": "otsu"}, "RandomAffine"),
    (
        {"num_control_points": 4, "max_displacement": 0, "locked_borders": 0},
        "RandomElasticDeformation",
    ),
    (
        {
            "num_control_points": (4, 5, 6),
            "max_displacement": (0, 1.0, 3.0),
            "locked_borders": 1,
        },
        "RandomElasticDeformation",
    ),
    ({"locked_borders": 2}, "RandomElasticDeformation"),
    ({"axes": 2, "downsampling": 1.1}, "RandomAnisotropy"),
    ({"axes": (0, 1, 2), "downsampling": (1.0, 1.1)}, "RandomAnisotropy"),
]


@pytest.mark.parametrize("args,transform", BAD_INPUTS)
def test_bad_inputs(args, transform):
    if not isinstance(transform, list):
        transform = [transform]
    for trans in transform:
        config = create_transform_config(trans)
        with pytest.raises(ValidationError):
            config(**args)


@pytest.mark.parametrize("args,transform", GOOD_INPUTS)
def test_good_inputs(args: dict, transform):
    config = create_transform_config(transform)
    c = config(**args)
    for arg, value in args.items():
        assert getattr(c, arg) == value


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
        for transform in ["RandomAffine", "RandomElasticDeformation"]:
            c = create_transform_config(transform)(
                target_shape=1, image_interpolation=mode, label_interpolation=mode
            )
            assert c.image_interpolation == mode
            assert c.label_interpolation == mode

        c = create_transform_config("RandomAnisotropy")(image_interpolation=mode)
        assert c.image_interpolation == mode


def test_axes():
    axes = [0, 1, 2, (0, 1), "AP", "LR", "IS", ("AP", "LR", "IS")]
    for ax in axes:
        c = create_transform_config("RandomFlip")(axes=ax)
        assert c.axes == ax
