import pytest
from pydantic import ValidationError

from clinicadl.transforms.config import create_transform_config

BAD_INPUTS = [
    ({"target_shape": (0, 2, 3)}, ["CropOrPad", "Resize"]),
    ({"target_shape": (1, 2)}, ["CropOrPad", "Resize"]),
    ({"target_shape": (1, 2, 3), "padding_mode": "abc"}, "CropOrPad"),
    ({"target_shape": (1, 2, 3), "mask_name": "mask", "labels": 0}, "CropOrPad"),
    ({"target_shape": (1, 2, 3), "mask_name": "mask", "labels": [0.5]}, "CropOrPad"),
    ({"target_shape": (1, 2, 3), "labels": (1,)}, "CropOrPad"),
    ({"mask_name": None, "labels": None}, "CropOrPad"),
    ({"target_shape": (1, 2, 3), "image_interpolation": "abc"}, "Resize"),
    ({"target_shape": (1, 2, 3), "label_interpolation": "abc"}, "Resize"),
    ({"target_mutiple": (1, 0, 3)}, "EnsureShapeMultiple"),
    ({"target_mutiple": (1, 2, 3), "method": "abc"}, "EnsureShapeMultiple"),
    ({"cropping": -1}, "Crop"),
    ({"cropping": (1, -1, 3)}, "Crop"),
    ({"cropping": (1, 2, 3, 4, -1, 6)}, "Crop"),
    ({"padding": -1}, "Pad"),
    ({"padding": (1, -1, 3)}, "Pad"),
    ({"padding": (1, 2, 3, 4, -1, 6)}, "Pad"),
    ({"padding": 1, "padding_mode": "abc"}, "Pad"),
]

GOOD_INPUTS = [
    (
        {
            "target_shape": (1, 2, 3),
            "padding_mode": 1,
            "mask_name": None,
            "labels": None,
        },
        "CropOrPad",
    ),
    ({"target_shape": (1, 2, 3), "mask_name": "mask", "labels": (1,)}, "CropOrPad"),
    ({"mask_name": "mask", "labels": None}, "CropOrPad"),
    ({"target_shape": (-1, 2, 3)}, "Resize"),
    ({"target_multiple": (1, 2, 3), "method": "crop"}, "EnsureShapeMultiple"),
    ({"target_multiple": (1, 2, 3), "method": "pad"}, "EnsureShapeMultiple"),
    ({"cropping": 1}, "Crop"),
    ({"cropping": (1, 0, 3)}, "Crop"),
    ({"cropping": (1, 2, 3, 4, 5, 6)}, "Crop"),
    ({"padding": 1}, "Pad"),
    ({"padding": (1, 0, 3), "padding_mode": 1}, "Pad"),
    ({"padding": (1, 2, 3, 4, 5, 6)}, "Pad"),
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
        c = create_transform_config("Resize")(
            target_shape=1, image_interpolation=mode, label_interpolation=mode
        )
        assert c.image_interpolation == mode
        assert c.label_interpolation == mode


def test_padding_mode():
    modes = [
        "edge",
        "linear_ramp",
        "maximum",
        "mean",
        "median",
        "minimum",
        "reflect",
        "symmetric",
        "wrap",
    ]
    for mode in modes:
        c = create_transform_config("Pad")(padding=1, padding_mode=mode)
        assert c.padding_mode == mode
        c = create_transform_config("CropOrPad")(target_shape=1, padding_mode=mode)
        assert c.padding_mode == mode
