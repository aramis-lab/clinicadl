from typing import Any, Union

# pylint: disable=unused-import
from .base import OneOfConfig, TransformConfig
from .enum import ImplementedTransform
from .intensity import (
    ClampConfig,
    MaskConfig,
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


def get_transform_config(
    name: Union[str, ImplementedTransform], **kwargs: Any
) -> TransformConfig:
    """
    Factory function to get a transform configuration object from its name
    and parameters.

    Parameters
    ----------
    name : Union[str, ImplementedTransform]
        the name of the transform. Check our documentation to know
        supported transforms.
    **kwargs : Any
        any parameter of the transform. Check our documentation on transforms to
        know these parameters.

    Returns
    -------
    TransformConfig
        the config object. Default values will be returned for the parameters
        not passed by the user.
    """
    transform = ImplementedTransform(name)
    config_name = f"{transform.value}Config"
    try:
        config = globals()[config_name]
    except KeyError:
        raise ValueError(
            f"Can't find the transform {name}. "
            f"Please check if you have imported the transform in the config factory."
        )

    return config(**kwargs)
