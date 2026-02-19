import pytest

from clinicadl.transforms.config import *
from clinicadl.transforms.config.intensity_augmentations import (
    RandomBlurConfig,
    RandomNoiseConfig,
)
from clinicadl.transforms.factory import get_transform_from_dict
from clinicadl.utils.json import read_json

MANDATORY_ARGS = {
    "masking_method": "mask",
    "remapping": {0: 1},
    "target_shape": 1,
    "target_multiple": 1,
    "cropping": 1,
    "padding": 1,
    "out_min": 0,
    "applied_labels": [0],
    "threshold": 0.5,
    "softmax": True,
}


@pytest.mark.parametrize(
    "config",
    [
        RandomMotionConfig,
        RandomGhostingConfig,
        RandomSpikeConfig,
        RandomBiasFieldConfig,
        RandomBlurConfig,
        RandomNoiseConfig,
        RandomSwapConfig,
        RandomGammaConfig,
        RescaleIntensityConfig,
        ZNormalizationConfig,
        MaskConfig,
        ClampConfig,
        RemapLabelsConfig,
        OneHotConfig,
        RandomFlipConfig,
        RandomAffineConfig,
        RandomElasticDeformationConfig,
        RandomAnisotropyConfig,
        CropOrPadConfig,
        ToCanonicalConfig,
        ResizeConfig,
        ResampleConfig,
        EnsureShapeMultipleConfig,
        CropConfig,
        PadConfig,
        OneHotConfig,
        ActivationsConfig,
        AsDiscreteConfig,
        KeepLargestConnectedComponentConfig,
        DistanceTransformEDTConfig,
        RemoveSmallObjectsConfig,
        LabelFilterConfig,
        FillHolesConfig,
        SobelGradientsConfig,
    ],
)
def test_get_transform_from_dict(config, tmp_path):
    c = config(**MANDATORY_ARGS)
    c.to_json(tmp_path / "config.json")
    config_dict = read_json(tmp_path / "config.json")
    c = get_transform_from_dict(config_dict)
    assert isinstance(c, config)

    if config is RandomMotionConfig:
        c = RandomMotionConfig(translation=[1, 2])
        assert get_transform_from_dict(c.to_dict()).translation == (1.0, 2.0)
