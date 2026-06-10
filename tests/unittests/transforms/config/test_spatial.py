from pathlib import Path

import numpy as np
import pytest
import torch
import torchio as tio
from pydantic import ValidationError
from torchio.datasets import Colin27

from clinicadl.transforms.config.spatial import (
    CropConfig,
    CropOrPadConfig,
    EnsureShapeMultipleConfig,
    PadConfig,
    ResampleConfig,
    ResizeConfig,
)

mask_path = (
    Path(__file__).parents[2]
    / "resources"
    / "bids"
    / "derivatives"
    / "caps"
    / "space-MNI152NLin2009cSym_res-1d3x1d2x1d1_label-leftHippocampus_mask.nii"
)

BAD_INPUTS = [
    ({"target_shape": (0, 2, 3)}, [CropOrPadConfig, ResizeConfig]),
    ({"target_shape": (1, 2)}, [CropOrPadConfig, ResizeConfig]),
    ({"target_shape": (1, 2, 3), "padding_mode": "abc"}, CropOrPadConfig),
    ({"target_shape": (1, 2, 3), "mask_name": "mask", "labels": 0}, CropOrPadConfig),
    (
        {"target_shape": (1, 2, 3), "mask_name": "mask", "labels": [0.5]},
        CropOrPadConfig,
    ),
    ({"target_shape": (1, 2, 3), "labels": (1,)}, CropOrPadConfig),
    ({"mask_name": None, "labels": None}, CropOrPadConfig),
    ({"target_shape": (1, 2, 3), "image_interpolation": "abc"}, ResizeConfig),
    ({"target_shape": (1, 2, 3), "label_interpolation": "abc"}, ResizeConfig),
    ({"target": 0}, ResampleConfig),
    ({"target": (0, 1.2, 1)}, ResampleConfig),
    ({"target": ((0, 3, 2), np.eye(4, 4))}, ResampleConfig),
    ({"target": ((1.2, 3, 2), np.eye(4, 4))}, ResampleConfig),
    ({"target": ((1, 3, 2), np.eye(4, 2))}, ResampleConfig),
    ({"target": ((1, 3, 2), [[0, 1], [1, 0]])}, ResampleConfig),
    (
        {
            "target": Colin27().t1,
        },
        ResampleConfig,
    ),
    (
        {
            "target": Path("abc.nii.gz"),
        },
        ResampleConfig,
    ),
    (
        {
            "pre_affine_name": "t1",
        },
        ResampleConfig,
    ),
    (
        {
            "image_interpolation": "abc",
        },
        ResampleConfig,
    ),
    (
        {
            "label_interpolation": "abc",
        },
        ResampleConfig,
    ),
    ({"target_multiple": (1, 0, 3)}, EnsureShapeMultipleConfig),
    ({"target_multiple": (1, 2, 3), "method": "abc"}, EnsureShapeMultipleConfig),
    ({"cropping": -1}, CropConfig),
    ({"cropping": (1, -1, 3)}, CropConfig),
    ({"cropping": (1, 2, 3, 4, -1, 6)}, CropConfig),
    ({"padding": -1}, PadConfig),
    ({"padding": (1, -1, 3)}, PadConfig),
    ({"padding": (1, 2, 3, 4, -1, 6)}, PadConfig),
    ({"padding": 1, "padding_mode": "abc"}, PadConfig),
]

GOOD_INPUTS = [
    (
        {
            "target_shape": (1, 2, 3),
            "padding_mode": 1,
            "mask_name": None,
            "labels": None,
        },
        CropOrPadConfig,
    ),
    ({"target_shape": (1, 2, 3), "mask_name": "mask", "labels": (1,)}, CropOrPadConfig),
    ({"mask_name": "mask", "labels": None}, CropOrPadConfig),
    ({"target_shape": (-1, 2, 3)}, ResizeConfig),
    ({"target": 1, "pre_affine_name": None}, ResampleConfig),
    ({"target": (1, 2.0, 2.1), "scalars_only": True}, ResampleConfig),
    ({"target": "t1", "scalars_only": False}, ResampleConfig),
    ({"target": mask_path}, ResampleConfig),
    ({"target": ((1, 3, 2), np.eye(4, 4))}, ResampleConfig),
    ({"target_multiple": (1, 2, 3), "method": "crop"}, EnsureShapeMultipleConfig),
    ({"target_multiple": (1, 2, 3), "method": "pad"}, EnsureShapeMultipleConfig),
    ({"cropping": 1}, CropConfig),
    ({"cropping": (1, 0, 3)}, CropConfig),
    ({"cropping": (1, 2, 3, 4, 5, 6)}, CropConfig),
    ({"padding": 1}, PadConfig),
    ({"padding": (1, 0, 3), "padding_mode": 1}, PadConfig),
    ({"padding": (1, 2, 3, 4, 5, 6)}, PadConfig),
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
    "args,config,transform",
    [
        ({"cropping": 1}, CropConfig, tio.Crop),
        ({"padding": 1}, PadConfig, tio.Pad),
        ({"target_shape": 3}, CropOrPadConfig, tio.CropOrPad),
        (
            {"target_multiple": (1, 2, 3)},
            EnsureShapeMultipleConfig,
            tio.EnsureShapeMultiple,
        ),
        ({}, ResampleConfig, tio.Resample),
        ({"target_shape": 3}, ResizeConfig, tio.Resize),
    ],
)
def test_get_object(args, config, transform):
    c = config(**args)
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
        c = ResizeConfig(
            target_shape=1, image_interpolation=mode, label_interpolation=mode
        )
        assert c.image_interpolation == mode
        assert c.label_interpolation == mode

        c = ResampleConfig(image_interpolation=mode, label_interpolation=mode)
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
        c = PadConfig(padding=1, padding_mode=mode)
        assert c.padding_mode == mode
        c = CropOrPadConfig(target_shape=1, padding_mode=mode)
        assert c.padding_mode == mode
