import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.transforms.config.intensity import (
    ClampConfig,
    MaskConfig,
    RescaleIntensityConfig,
    ZNormalizationConfig,
)

BAD_INPUTS = [
    ({"out_min_max": -0.5}, RescaleIntensityConfig),
    ({"out_min_max": (0.5, -0.5)}, RescaleIntensityConfig),
    ({"percentiles": 101}, RescaleIntensityConfig),
    ({"percentiles": (0, 101.1)}, RescaleIntensityConfig),
    ({"in_min_max": -0.5}, RescaleIntensityConfig),
    ({"in_min_max": (0.5, -0.5)}, RescaleIntensityConfig),
    ({"masking_method": None, "labels": 0}, MaskConfig),
    ({"masking_method": None, "labels": [0.5]}, MaskConfig),
    ({"out_min": 1.0, "out_max": 0.5}, ClampConfig),
    ({"out_min": None, "out_max": None}, ClampConfig),
    ({}, ClampConfig),
]

GOOD_INPUTS = [
    ({"out_min_max": 0.5}, RescaleIntensityConfig),
    ({"out_min_max": (-0.5, 0.5)}, RescaleIntensityConfig),
    ({"percentiles": 100}, RescaleIntensityConfig),
    ({"percentiles": (0.2, 99.2)}, RescaleIntensityConfig),
    ({"in_min_max": 0.5}, RescaleIntensityConfig),
    ({"in_min_max": (-0.5, 0.5)}, RescaleIntensityConfig),
    ({"masking_method": None, "outside_value": 1.0}, MaskConfig),
    ({"masking_method": None, "labels": (1,)}, MaskConfig),
    ({"masking_method": None, "labels": None}, MaskConfig),
    ({"out_min": 0.5, "out_max": 1.0}, ClampConfig),
    ({"out_min": 0.5, "out_max": None}, ClampConfig),
    ({"out_min": None, "out_max": 1.0}, ClampConfig),
]

X = tio.Subject(
    image=tio.ScalarImage(tensor=torch.randn(1, 16, 17, 18)),
    label=tio.LabelMap(tensor=torch.ones(1, 16, 17, 18)),
)


@pytest.mark.parametrize("args,config", BAD_INPUTS)
def test_bad_inputs(args, config):
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
        ({}, RescaleIntensityConfig, tio.RescaleIntensity),
        ({"masking_method": None}, MaskConfig, tio.Mask),
        ({"out_max": 1.0}, ClampConfig, tio.Clamp),
        ({}, ZNormalizationConfig, tio.ZNormalization),
    ],
)
def test_get_object(args, config, transform):
    c = config(**args)
    transform_from_config = c.get_object()
    assert isinstance(transform_from_config, transform)
    assert isinstance(transform_from_config(X), tio.Subject)


def test_masking_method():
    methods = [
        "mask",
        1,
        (1, 2, 3),
        (1, 2, 3, 4, 5, 0),
        "Left",
        "Right",
        "Anterior",
        "Posterior",
        "Inferior",
        "Superior",
    ]
    configs = [MaskConfig, RescaleIntensityConfig, ZNormalizationConfig]
    for config in configs:
        for method in methods:
            c = config(masking_method=method)
            assert c.masking_method == method
    with pytest.raises(ValidationError):
        config(masking_method=lambda x: x > 1)
