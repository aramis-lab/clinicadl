import pytest
import torch
import torchio as tio
from pydantic import ValidationError

from clinicadl.transforms.config.intensity_augmentations import (
    RandomBiasFieldConfig,
    RandomBlurConfig,
    RandomGammaConfig,
    RandomGhostingConfig,
    RandomMotionConfig,
    RandomNoiseConfig,
    RandomSpikeConfig,
    RandomSwapConfig,
)

BAD_INPUTS = [
    ({"degrees": -0.5}, RandomMotionConfig),
    ({"translation": -0.5}, RandomMotionConfig),
    ({"degrees": (0.5, -0.5)}, RandomMotionConfig),
    ({"translation": (0.5, -0.5)}, RandomMotionConfig),
    ({"num_transforms": 0}, RandomMotionConfig),
    ({"image_interpolation": "abc"}, RandomMotionConfig),
    ({"num_ghosts": 1.5}, RandomGhostingConfig),
    ({"num_ghosts": (-1, 1)}, RandomGhostingConfig),
    ({"intensity": -0.1}, RandomGhostingConfig),
    ({"intensity": (-0.1, 0.1)}, RandomGhostingConfig),
    ({"axes": "R"}, RandomGhostingConfig),
    ({"axes": 3}, RandomGhostingConfig),
    ({"restore": (0, 1.1)}, RandomGhostingConfig),
    ({"num_spikes": 1.1}, RandomSpikeConfig),
    ({"num_spikes": -1}, RandomSpikeConfig),
    ({"num_spikes": (-1, 1)}, RandomSpikeConfig),
    ({"intensity": -1}, RandomSpikeConfig),
    ({"coefficients": -1}, RandomBiasFieldConfig),
    ({"order": -1}, RandomBiasFieldConfig),
    ({"std": -1}, [RandomBlurConfig, RandomNoiseConfig]),
    ({"std": (-1, 1)}, [RandomBlurConfig, RandomNoiseConfig]),
    ({"mean": -1}, RandomNoiseConfig),
    ({"patch_size": -1}, RandomSwapConfig),
    ({"patch_size": (-1, 2, 3)}, RandomSwapConfig),
    ({"patch_size": (2, 3)}, RandomSwapConfig),
    ({"num_iterations": -1}, RandomSwapConfig),
    ({"log_gamma": -1}, RandomGammaConfig),
]

GOOD_INPUTS = [
    ({"degrees": 0.5, "translation": 0.5, "num_transforms": 1}, RandomMotionConfig),
    ({"degrees": (-0.5, 0.5), "translation": (-0.5, 0.5)}, RandomMotionConfig),
    (
        {"num_ghosts": 0, "axes": 0, "intensity": 0.1, "restore": 0.5},
        RandomGhostingConfig,
    ),
    (
        {
            "num_ghosts": (1, 5),
            "axes": (0, 2),
            "intensity": (0.1, 0.2),
            "restore": (0, 0.1),
        },
        RandomGhostingConfig,
    ),
    ({"num_spikes": (0, 1), "intensity": 1.0}, RandomSpikeConfig),
    ({"num_spikes": 1, "intensity": (-1.0, 1.0)}, RandomSpikeConfig),
    ({"coefficients": 0, "order": 0}, RandomBiasFieldConfig),
    ({"coefficients": (-1, 1.2)}, RandomBiasFieldConfig),
    ({"std": 0.1}, RandomBlurConfig),
    ({"std": (0, 1)}, RandomBlurConfig),
    ({"mean": 1, "std": 0.1}, RandomNoiseConfig),
    ({"mean": (-1.0, 1.0), "std": (0, 1)}, RandomNoiseConfig),
    ({"patch_size": 1, "num_iterations": 0}, RandomSwapConfig),
    ({"patch_size": (1, 1, 1)}, RandomSwapConfig),
    ({"log_gamma": 0}, RandomGammaConfig),
    ({"log_gamma": (-1, 1.2)}, RandomGammaConfig),
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
    "config,transform",
    [
        (RandomMotionConfig, tio.RandomMotion),
        (RandomBiasFieldConfig, tio.RandomBiasField),
        (RandomBlurConfig, tio.RandomBlur),
        (RandomGammaConfig, tio.RandomGamma),
        (RandomGhostingConfig, tio.RandomGhosting),
        (RandomNoiseConfig, tio.RandomNoise),
        (RandomSwapConfig, tio.RandomSwap),
        (RandomSpikeConfig, tio.RandomSpike),
    ],
)
def test_get_object(config, transform):
    c = config()
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
        c = RandomMotionConfig(image_interpolation=mode)
        assert c.image_interpolation == mode
