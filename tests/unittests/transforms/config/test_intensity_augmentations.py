import pytest
from pydantic import ValidationError

from clinicadl.transforms.config import create_transform_config

BAD_INPUTS = [
    ({"degrees": -0.5}, "RandomMotion"),
    ({"translation": -0.5}, "RandomMotion"),
    ({"degrees": (0.5, -0.5)}, "RandomMotion"),
    ({"translation": (0.5, -0.5)}, "RandomMotion"),
    ({"num_transforms": 0}, "RandomMotion"),
    ({"image_interpolation": "abc"}, "RandomMotion"),
    ({"num_ghosts": 1.5}, "RandomGhosting"),
    ({"num_ghosts": (-1, 1)}, "RandomGhosting"),
    ({"intensity": -0.1}, "RandomGhosting"),
    ({"intensity": (-0.1, 0.1)}, "RandomGhosting"),
    ({"axes": "R"}, "RandomGhosting"),
    ({"axes": 3}, "RandomGhosting"),
    ({"restore": 1.1}, "RandomGhosting"),
    ({"num_spikes": 1.1}, "RandomSpike"),
    ({"num_spikes": -1}, "RandomSpike"),
    ({"num_spikes": (-1, 1)}, "RandomSpike"),
    ({"intensity": -1}, "RandomSpike"),
    ({"coefficients": -1}, "RandomBiasField"),
    ({"order": -1}, "RandomBiasField"),
    ({"std": -1}, ["RandomBlur", "RandomNoise"]),
    ({"std": (-1, 1)}, ["RandomBlur", "RandomNoise"]),
    ({"mean": -1}, "RandomNoise"),
    ({"patch_size": -1}, "RandomSwap"),
    ({"patch_size": (-1, 2, 3)}, "RandomSwap"),
    ({"patch_size": (2, 3)}, "RandomSwap"),
    ({"num_iterations": -1}, "RandomSwap"),
    ({"log_gamma": -1}, "RandomGamma"),
]

GOOD_INPUTS = [
    ({"degrees": 0.5, "translation": 0.5, "num_transforms": 1}, "RandomMotion"),
    ({"degrees": (-0.5, 0.5), "translation": (-0.5, 0.5)}, "RandomMotion"),
    ({"num_ghosts": 0, "axes": 0, "intensity": 0.1, "restore": 0.5}, "RandomGhosting"),
    (
        {"num_ghosts": (1, 5), "axes": (0, 2), "intensity": (0.1, 0.2), "restore": 0},
        "RandomGhosting",
    ),
    ({"num_spikes": (0, 1), "intensity": 1.0}, "RandomSpike"),
    ({"num_spikes": 1, "intensity": (-1.0, 1.0)}, "RandomSpike"),
    ({"coefficients": 0, "order": 0}, "RandomBiasField"),
    ({"coefficients": (-1, 1.2)}, "RandomBiasField"),
    ({"std": 0.1}, "RandomBlur"),
    ({"std": (0, 1)}, "RandomBlur"),
    ({"mean": 1, "std": 0.1}, "RandomNoise"),
    ({"mean": (-1.0, 1.0), "std": (0, 1)}, "RandomNoise"),
    ({"patch_size": 1, "num_iterations": 0}, "RandomSwap"),
    ({"patch_size": (1, 1, 1)}, "RandomSwap"),
    ({"log_gamma": 0}, "RandomGamma"),
    ({"log_gamma": (-1, 1.2)}, "RandomGamma"),
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
        for transform in ["RandomMotion"]:
            c = create_transform_config(transform)(image_interpolation=mode)
            assert c.image_interpolation == mode
