import pytest
from pydantic import ValidationError

from clinicadl.transforms.config import *
from clinicadl.transforms.config.intensity_augmentations import (
    RandomBlurConfig,
    RandomNoiseConfig,
)

MANDATORY_ARGS = {
    "masking_method": "mask",
    "remapping": {0: 1},
    "target_shape": 1,
    "target_multiple": 1,
    "cropping": 1,
    "padding": 1,
    "out_min": 0,
    "applied_labels": [0],
}


@pytest.mark.parametrize(
    "name,config",
    [
        ("RandomMotion", RandomMotionConfig),
        ("RandomGhosting", RandomGhostingConfig),
        ("RandomSpike", RandomSpikeConfig),
        ("RandomBiasField", RandomBiasFieldConfig),
        ("RandomBlur", RandomBlurConfig),
        ("RandomNoise", RandomNoiseConfig),
        ("RandomSwap", RandomSwapConfig),
        ("RandomGamma", RandomGammaConfig),
        ("RescaleIntensity", RescaleIntensityConfig),
        ("ZNormalization", ZNormalizationConfig),
        ("Mask", MaskConfig),
        ("Clamp", ClampConfig),
        ("RemapLabels", RemapLabelsConfig),
        ("OneHot", OneHotConfig),
        ("RandomFlip", RandomFlipConfig),
        ("RandomAffine", RandomAffineConfig),
        ("RandomElasticDeformation", RandomElasticDeformationConfig),
        ("RandomAnisotropy", RandomAnisotropyConfig),
        ("CropOrPad", CropOrPadConfig),
        ("ToCanonical", ToCanonicalConfig),
        ("Resize", ResizeConfig),
        ("Resample", ResampleConfig),
        ("EnsureShapeMultiple", EnsureShapeMultipleConfig),
        ("Crop", CropConfig),
        ("Pad", PadConfig),
        ("OneOf", OneHotConfig),
        ("Activations", ActivationsConfig),
        ("AsDiscrete", AsDiscreteConfig),
        ("KeepLargestConnectedComponent", KeepLargestConnectedComponentConfig),
        ("DistanceTransformEDT", DistanceTransformEDTConfig),
        ("RemoveSmallObjects", RemoveSmallObjectsConfig),
        ("LabelFilter", LabelFilterConfig),
        ("FillHoles", FillHolesConfig),
        ("SobelGradients", SobelGradientsConfig),
    ],
)
def test_get_transform_config(name, config):
    if name == "OneOf":
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
    else:
        try:
            c = get_transform_config(name)
        except (TypeError, ValidationError):
            for arg, value in MANDATORY_ARGS.items():
                try:
                    c = get_transform_config(name, **{arg: value})
                except (TypeError, ValidationError):
                    continue

        assert c.name == name
        assert isinstance(c, config)

    if name == "RandomNoise":
        config = get_transform_config("RandomNoise", mean=1)
        assert config.name == "RandomNoise"
        assert config.mean == 1
        assert config.std == (0, 0.25)
