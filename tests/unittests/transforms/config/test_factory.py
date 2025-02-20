import pytest

from clinicadl.transforms.config.factory import get_transform_config
from clinicadl.transforms.config.intensity_augmentations import (
    RandomBlurConfig,
    RandomNoiseConfig,
)


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

    config = get_transform_config("NanRemoval", nan=1)
    assert config.name == "NanRemoval"
    assert config.nan == 1
    assert config.posinf is None
    assert config.neginf is None
