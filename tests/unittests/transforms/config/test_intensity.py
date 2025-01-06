import pytest
from pydantic import ValidationError

from clinicadl.transforms.config import create_transform_config

BAD_INPUTS = [
    ({"out_min_max": -0.5}, "RescaleIntensity"),
    ({"out_min_max": (0.5, -0.5)}, "RescaleIntensity"),
    ({"percentiles": 101}, "RescaleIntensity"),
    ({"percentiles": (0, 101.1)}, "RescaleIntensity"),
    ({"in_min_max": -0.5}, "RescaleIntensity"),
    ({"in_min_max": (0.5, -0.5)}, "RescaleIntensity"),
    ({"masking_method": None, "labels": 0}, "Mask"),
    ({"masking_method": None, "labels": [0.5]}, "Mask"),
    ({"out_min": 1.0, "out_max": 0.5}, "Clamp"),
    ({"out_min": None, "out_max": None}, "Clamp"),
    ({}, "Clamp"),
]

GOOD_INPUTS = [
    ({"out_min_max": 0.5}, "RescaleIntensity"),
    ({"out_min_max": (-0.5, 0.5)}, "RescaleIntensity"),
    ({"percentiles": 100}, "RescaleIntensity"),
    ({"percentiles": (0.2, 99.2)}, "RescaleIntensity"),
    ({"in_min_max": 0.5}, "RescaleIntensity"),
    ({"in_min_max": (-0.5, 0.5)}, "RescaleIntensity"),
    ({"masking_method": None, "outside_value": 1.0}, "Mask"),
    ({"masking_method": None, "labels": (1,)}, "Mask"),
    ({"masking_method": None, "labels": None}, "Mask"),
    ({"out_min": 0.5, "out_max": 1.0}, "Clamp"),
    ({"out_min": 0.5, "out_max": None}, "Clamp"),
    ({"out_min": None, "out_max": 1.0}, "Clamp"),
    ({"posinf": 1.2, "neginf": 0.1}, "NanRemoval"),
    ({"posinf": None, "neginf": None}, "NanRemoval"),
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
    transforms = ["Mask", "RescaleIntensity", "ZNormalization"]
    for transform in transforms:
        for method in methods:
            config = create_transform_config(transform)
            c = config(masking_method=method)
            assert c.masking_method == method
    with pytest.raises(ValidationError):
        config(masking_method=lambda x: x > 1)
