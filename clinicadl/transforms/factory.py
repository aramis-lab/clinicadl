from copy import deepcopy
from typing import Any, Tuple, Union

import torchio.transforms as tio_transforms

import clinicadl.transforms.homemade_transforms as homemade_transforms
from clinicadl.utils.factories import update_config_with_defaults

from .config import (
    ImplementedTransform,
    TransformConfig,
    TransformType,
    create_transform_config,
)
from .config.base import OneOfConfig
from .utils import Transform


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
        available transforms.
    **kwargs : Any
        any parameter of the transform. Check our documentation on transforms to
        know these parameters.

    Returns
    -------
    TransformConfig
        the config object. Default values will be returned for the parameters
        not passed by the user.
    """
    config = create_transform_config(name)(**kwargs)
    if config._type == TransformType.TORCHIO:  # pylint: disable=protected-access
        transform_class = getattr(tio_transforms, config.name)
    elif config._type == TransformType.HOMEMADE:  # pylint: disable=protected-access
        transform_class = getattr(homemade_transforms, config.name)

    update_config_with_defaults(config, function=transform_class.__init__)  # pylint: disable=possibly-used-before-assignment

    return config


def get_transform_from_config(
    config: TransformConfig,
) -> Tuple[Transform, TransformConfig]:
    """
    Factory function to get a TorchIO transform from a TransformConfig instance.

    Parameters
    ----------
    config : TransformConfig
        the configuration object.

    Returns
    -------
    Transform
        the transform.
    TransformConfig
        the updated config object: the arguments set to default will be updated
        with their effective values (the default values from the library).
        Useful for reproducibility.
    """
    config = deepcopy(config)
    if config._type == TransformType.TORCHIO:  # pylint: disable=protected-access
        transform_class = getattr(tio_transforms, config.name)
    elif config._type == TransformType.HOMEMADE:  # pylint: disable=protected-access
        transform_class = getattr(homemade_transforms, config.name)

    update_config_with_defaults(config, function=transform_class.__init__)  # pylint: disable=possibly-used-before-assignment

    if config.name == ImplementedTransform.ONE_OF:
        config: OneOfConfig
        config_dict = {
            get_transform_from_config(transform)[0]: proba
            for transform, proba in zip(config.transforms, config.probabilities)
        }
        transform = transform_class(config_dict)
    else:
        config_dict = config.model_dump(exclude={"name", "_type"})
        transform = transform_class(**config_dict)

    return transform, config
