import monai.transforms as transforms
import pytest
import torch
from pydantic import ValidationError

from clinicadl.data import structures
from clinicadl.transforms.config import (
    ActivationsConfig,
    AsDiscreteConfig,
    DistanceTransformEDTConfig,
    FillHolesConfig,
    KeepLargestConnectedComponentConfig,
    LabelFilterConfig,
    RemoveSmallObjectsConfig,
    SobelGradientsConfig,
)
from clinicadl.transforms.monai_wrapper import MonaiTransformWrapper

BAD_INPUTS = [
    ({"to_onehot": 0, "rounding": "abc"}, AsDiscreteConfig),
    ({"rounding": "abc"}, AsDiscreteConfig),
    ({"applied_labels": 1.4}, KeepLargestConnectedComponentConfig),
    (
        {"connectivity": 0},
        [
            KeepLargestConnectedComponentConfig,
            RemoveSmallObjectsConfig,
            FillHolesConfig,
        ],
    ),
    ({"num_components": 0}, KeepLargestConnectedComponentConfig),
    ({"min_size": 0}, RemoveSmallObjectsConfig),
    ({"applied_labels": 1.1}, [LabelFilterConfig, FillHolesConfig]),
    ({"kernel_size": 1}, SobelGradientsConfig),
    ({"kernel_size": 4}, SobelGradientsConfig),
    ({"spatial_axes": -1}, SobelGradientsConfig),
    ({"padding_mode": "abc"}, SobelGradientsConfig),
    ({"dtype": int}, SobelGradientsConfig),
]
GOOD_INPUTS = [
    (
        {"include": ["abc"]},
        [
            ActivationsConfig,
            AsDiscreteConfig,
            KeepLargestConnectedComponentConfig,
            DistanceTransformEDTConfig,
            RemoveSmallObjectsConfig,
            SobelGradientsConfig,
        ],
    ),
    (
        {"exclude": ["abc"]},
        [
            ActivationsConfig,
            AsDiscreteConfig,
            KeepLargestConnectedComponentConfig,
            DistanceTransformEDTConfig,
            RemoveSmallObjectsConfig,
            SobelGradientsConfig,
        ],
    ),
    ({"sigmoid": True, "softmax": False, "other": lambda x: x}, ActivationsConfig),
    ({"other": None}, ActivationsConfig),
    (
        {
            "argmax": True,
            "to_onehot": 1,
            "threshold": -1.5,
            "rounding": "torchrounding",
        },
        AsDiscreteConfig,
    ),
    (
        {
            "to_onehot": None,
            "threshold": None,
            "rounding": None,
        },
        AsDiscreteConfig,
    ),
    (
        {
            "applied_labels": 1,
            "is_onehot": True,
            "independent": False,
            "connectivity": 1,
            "num_components": 1,
        },
        KeepLargestConnectedComponentConfig,
    ),
    (
        {
            "applied_labels": [1, 2],
            "is_onehot": None,
            "connectivity": None,
            "num_components": None,
        },
        KeepLargestConnectedComponentConfig,
    ),
    (
        {
            "applied_labels": None,
        },
        KeepLargestConnectedComponentConfig,
    ),
    (
        {
            "sampling": None,
        },
        DistanceTransformEDTConfig,
    ),
    (
        {
            "sampling": -0.1,
        },
        DistanceTransformEDTConfig,
    ),
    (
        {
            "sampling": [-0.1, 0.2],
        },
        DistanceTransformEDTConfig,
    ),
    (
        {
            "min_size": 1,
            "connectivity": 1,
            "independent_channels": True,
            "by_measure": False,
        },
        RemoveSmallObjectsConfig,
    ),
    (
        {
            "connectivity": None,
        },
        RemoveSmallObjectsConfig,
    ),
    (
        {"applied_labels": [1, 2], "include": ["abc"]},
        [LabelFilterConfig, FillHolesConfig],
    ),
    (
        {"applied_labels": [1, 2], "exclude": ["abc"]},
        [LabelFilterConfig, FillHolesConfig],
    ),
    ({"connectivity": 1}, FillHolesConfig),
    (
        {
            "kernel_size": 3,
            "spatial_axes": 0,
            "normalize_kernels": 0,
            "normalize_gradients": True,
            "padding_mode": "zeros",
            "dtype": torch.float32,
        },
        SobelGradientsConfig,
    ),
    (
        {
            "spatial_axes": [0, 1],
            "padding_mode": "reflect",
        },
        SobelGradientsConfig,
    ),
    (
        {
            "spatial_axes": None,
            "padding_mode": "replicate",
        },
        SobelGradientsConfig,
    ),
    (
        {
            "padding_mode": "circular",
        },
        SobelGradientsConfig,
    ),
]


X = structures.ColinDataPoint()


@pytest.mark.parametrize("args,configs", BAD_INPUTS)
def test_bad_inputs(args, configs):
    if not isinstance(configs, list):
        configs = [configs]
    for config in configs:
        with pytest.raises(ValidationError):
            config(**args)


@pytest.mark.parametrize("args,configs", GOOD_INPUTS)
def test_good_inputs(args: dict, configs):
    if not isinstance(configs, list):
        configs = [configs]
    for config in configs:
        c = config(**args)
        for arg, value in args.items():
            assert getattr(c, arg) == value


@pytest.mark.parametrize(
    "args,config,transform",
    [
        ({"include": ["label"]}, ActivationsConfig, transforms.Activations),
        ({"include": ["label"]}, AsDiscreteConfig, transforms.AsDiscrete),
        (
            {"include": ["label"]},
            KeepLargestConnectedComponentConfig,
            transforms.KeepLargestConnectedComponent,
        ),
        (
            {"include": ["label"]},
            DistanceTransformEDTConfig,
            transforms.DistanceTransformEDT,
        ),
        (
            {"include": ["label"]},
            RemoveSmallObjectsConfig,
            transforms.RemoveSmallObjects,
        ),
        (
            {"include": ["label"], "applied_labels": 1},
            LabelFilterConfig,
            transforms.LabelFilter,
        ),
        ({"include": ["label"]}, FillHolesConfig, transforms.FillHoles),
        ({"include": ["label"]}, SobelGradientsConfig, transforms.SobelGradients),
    ],
)
def test_get_object(args, config, transform):
    c = config(**args)
    transform_from_config = c.get_object()
    assert isinstance(transform_from_config, MonaiTransformWrapper)
    assert isinstance(transform_from_config.transform, transform)
    output = transform_from_config(X)
    assert isinstance(output, structures.DataPoint)
