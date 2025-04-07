from typing import Any, Union

# pylint: disable=unused-import
from ...config.base import TransformConfig
from .enum import ZooTransform

# factory of custom transforms
from .nan_removal import NanRemovalConfig


def get_transform_config(
    name: Union[str, ZooTransform], **kwargs: Any
) -> TransformConfig:
    """
    Factory function to get a transform configuration object from its name
    and parameters.

    Parameters
    ----------
    name : Union[str, ZooTransform]
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
    transform = ZooTransform(name)
    config_name = f"{transform.value}Config"
    try:
        config = globals()[config_name]
    except KeyError:
        raise ValueError(
            f"Can't find the transform {name}. "
            f"Please check if you have imported the transform in the config factory."
        )

    return config(**kwargs)
