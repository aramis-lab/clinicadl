import pytest

from clinicadl.transforms.config.intensity_augmentations import (
    RandomBlurConfig,
    RandomNoiseConfig,
)
from clinicadl.transforms.utils import get_transform_config


def test_get_transform_config():
    config = get_transform_config(
        "OneOf",
        transforms=[
            get_transform_config("RandomBlur"),
            get_transform_config("RandomNoise"),
        ],
        probabilities=[1, 9],
    )
    assert config.name == "OneOf"
    assert config.transforms == [RandomBlurConfig(), RandomNoiseConfig()]
    assert config.probabilities == [1, 9]

    with pytest.raises(ValueError):
        get_transform_config("abc")
