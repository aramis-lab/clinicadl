import pytest
import torch
import torchio as tio

from clinicadl.transforms.zoo import NanRemoval
from clinicadl.transforms.zoo.config.factory import NanRemovalConfig

GOOD_INPUTS = [
    ({"posinf": 1.2, "neginf": 0.1}, NanRemovalConfig),
    ({"posinf": None, "neginf": None}, NanRemovalConfig),
]

X = tio.Subject(
    image=tio.ScalarImage(tensor=torch.randn(1, 16, 17, 18)),
    label=tio.LabelMap(tensor=torch.ones(1, 16, 17, 18)),
)


@pytest.mark.parametrize("args,config", GOOD_INPUTS)
def test_good_inputs(args: dict, config):
    c = config(**args)
    for arg, value in args.items():
        assert getattr(c, arg) == value


@pytest.mark.parametrize(
    "args,config,transform",
    [
        ({}, NanRemovalConfig, NanRemoval),
    ],
)
def test_get_object(args, config, transform):
    c = config(**args)
    transform_from_config = c.get_object()
    assert isinstance(transform_from_config, transform)
    assert isinstance(transform_from_config(X), tio.Subject)
