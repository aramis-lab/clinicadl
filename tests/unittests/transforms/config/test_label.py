import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.transforms.config.label import OneHotConfig, RemapLabelsConfig

BAD_INPUTS = [
    ({"remapping": {1: 1.5}}, RemapLabelsConfig),
    ({"num_classes": 0}, OneHotConfig),
]

GOOD_INPUTS = [
    ({"remapping": {-1: 2, 0: 3}}, RemapLabelsConfig),
    ({"num_classes": 1}, OneHotConfig),
]

X = tio.Subject(
    image=tio.ScalarImage(tensor=torch.randn(1, 16, 17, 18)),
    label=tio.LabelMap(tensor=torch.randint(0, 2, (1, 16, 17, 18))),
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
        ({"remapping": {1: 2}}, RemapLabelsConfig, tio.RemapLabels),
        ({"num_classes": 2}, OneHotConfig, tio.OneHot),
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
    for method in methods:
        c = RemapLabelsConfig(remapping={0: 1}, masking_method=method)
        assert c.masking_method == method
    with pytest.raises(ValidationError):
        RemapLabelsConfig(remapping={0: 1}, masking_method=lambda x: x > 1)
