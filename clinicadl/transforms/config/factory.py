from typing import Type, Union

# pylint: disable=unused-import
from .base import OneOfConfig, TransformConfig
from .enum import ImplementedTransform
from .intensity import (
    ClampConfig,
    MaskConfig,
    NanRemovalConfig,
    RescaleIntensityConfig,
    ZNormalizationConfig,
)
from .intensity_augmentations import (
    RandomBiasFieldConfig,
    RandomBlurConfig,
    RandomGammaConfig,
    RandomGhostingConfig,
    RandomMotionConfig,
    RandomNoiseConfig,
    RandomSpikeConfig,
    RandomSwapConfig,
)
from .label import OneHotConfig, RemapLabelsConfig
from .spatial import (
    CropConfig,
    CropOrPadConfig,
    EnsureShapeMultipleConfig,
    PadConfig,
    ResampleConfig,
    ResizeConfig,
    ToCanonicalConfig,
)
from .spatial_augmentations import (
    RandomAffineConfig,
    RandomAnisotropyConfig,
    RandomElasticDeformationConfig,
    RandomFlipConfig,
)


def create_transform_config(
    transform: Union[str, ImplementedTransform],
) -> Type[TransformConfig]:
    """
    A factory function to create a config class suited for the transform.

    Parameters
    ----------
    transform : Union[str, ImplementedTransform]
        The name of the transform.

    Returns
    -------
    Type[TransformConfig]
        The config class.

    Raises
    ------
    ValueError
        If `transform` is not supported.
    """
    transform = ImplementedTransform(transform)
    config_name = "".join([transform, "Config"])
    config = globals()[config_name]

    return config
