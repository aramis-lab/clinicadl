import monai.transforms as transforms
import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.data.structures import DataPoint
from clinicadl.transforms.config import (
    ActivationsConfig,
    AsDiscreteConfig,
    DistanceTransformEDTConfig,
    FillHolesConfig,
    FormatConfig,
    KeepLargestConnectedComponentConfig,
    LabelFilterConfig,
    RemoveSmallObjectsConfig,
    SobelGradientsConfig,
)
from clinicadl.transforms.homemade import Format
from clinicadl.transforms.monai_wrapper import MonaiTransformWrapper

BAD_INPUTS = [
    ({}, ActivationsConfig),
    ({"softmax": True, "sigmoid": True}, ActivationsConfig),
    ({"to_onehot": 0}, AsDiscreteConfig),
    ({"rounding": "abc"}, AsDiscreteConfig),
    ({}, AsDiscreteConfig),
    ({"dtype": int, "argxmax": True}, SobelGradientsConfig),
    ({"to_onehot": 2, "threshold": 0.5}, AsDiscreteConfig),
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
    ({"dtype": int}, [SobelGradientsConfig, FormatConfig]),
    ({"unsqueeze": -1}, FormatConfig),
    ({"squeeze": [0, -1]}, FormatConfig),
]
GOOD_INPUTS = [
    (
        {"include": ["abc"]},
        [
            KeepLargestConnectedComponentConfig,
            DistanceTransformEDTConfig,
            RemoveSmallObjectsConfig,
            SobelGradientsConfig,
            FormatConfig,
        ],
    ),
    (
        {"exclude": ["abc"]},
        [
            KeepLargestConnectedComponentConfig,
            DistanceTransformEDTConfig,
            RemoveSmallObjectsConfig,
            SobelGradientsConfig,
            FormatConfig,
        ],
    ),
    (
        {"sigmoid": False, "softmax": False, "other": lambda x: x, "exclude": ["abc"]},
        ActivationsConfig,
    ),
    ({"softmax": True, "other": None, "include": ["abc"]}, ActivationsConfig),
    (
        {
            "argmax": False,
            "to_onehot": 1,
            "threshold": None,
            "rounding": None,
            "include": ["abc"],
            "dtype": torch.int,
        },
        AsDiscreteConfig,
    ),
    (
        {
            "rounding": "torchrounding",
            "exclude": ["abc"],
        },
        AsDiscreteConfig,
    ),
    (
        {
            "threshold": 0.5,
        },
        AsDiscreteConfig,
    ),
    (
        {
            "argmax": True,
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
    ({"unsqueeze": 0, "squeeze": True, "dtype": torch.int16}, FormatConfig),
    ({"unsqueeze": None, "squeeze": 1, "dtype": None}, FormatConfig),
    ({"squeeze": False}, FormatConfig),
    ({"squeeze": [0, 1]}, FormatConfig),
]


X = DataPoint(
    image=tio.ScalarImage(tensor=torch.randn(1, 2, 2, 2)),
    label=tio.LabelMap(tensor=torch.randint(0, 2, (1, 2, 2, 2))),
    participant="abc",
    session="abc",
)


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
        (
            {"softmax": True},
            ActivationsConfig,
            transforms.Activations,
        ),
        ({"argmax": True}, AsDiscreteConfig, transforms.AsDiscrete),
        (
            {},
            KeepLargestConnectedComponentConfig,
            transforms.KeepLargestConnectedComponent,
        ),
        (
            {},
            DistanceTransformEDTConfig,
            transforms.DistanceTransformEDT,
        ),
        (
            {"include": ["label"]},
            RemoveSmallObjectsConfig,
            transforms.RemoveSmallObjects,
        ),
        (
            {"applied_labels": 1},
            LabelFilterConfig,
            transforms.LabelFilter,
        ),
        ({}, FillHolesConfig, transforms.FillHoles),
        ({}, SobelGradientsConfig, transforms.SobelGradients),
        ({}, FormatConfig, Format),
    ],
)
def test_get_object(args, config, transform):
    c = config(**args)
    transform_from_config = c.get_object()
    assert isinstance(transform_from_config, MonaiTransformWrapper)
    assert isinstance(transform_from_config.transform, transform)
    output = transform_from_config(X)
    assert isinstance(output, DataPoint)
