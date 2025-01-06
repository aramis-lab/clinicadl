import pytest
from pydantic import ValidationError

from clinicadl.transforms.config import create_transform_config

BAD_INPUTS = [
    ({"remapping": {1: 1.5}}, "RemapLabels"),
    ({"num_classes": 0}, "OneHot"),
]

GOOD_INPUTS = [
    ({"remapping": {-1: 2, 0: 3}}, "RemapLabels"),
    ({"num_classes": 1}, "OneHot"),
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
        config = create_transform_config("RemapLabels")
        c = config(remapping={0: 1}, masking_method=method)
        assert c.masking_method == method
    with pytest.raises(ValidationError):
        config(masking_method=lambda x: x > 1)
