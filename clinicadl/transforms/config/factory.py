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
from .post_processing import (
    ActivationsConfig,
    AsDiscreteConfig,
    DistanceTransformEDTConfig,
    FillHolesConfig,
    KeepLargestConnectedComponentConfig,
    LabelFilterConfig,
    RemoveSmallObjectsConfig,
    SobelGradientsConfig,
)
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
    transform = ImplementedTransform(name).value
    config_name = f"{transform}Config"
    config = globals()[config_name]

    return config(**kwargs)
