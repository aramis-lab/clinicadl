from typing import Any, Callable, Union

from clinicadl.data.structures import DataPoint

from .config.base import TransformConfig
from .config.enum import ImplementedTransform
from .config.factory import get_transform_config as gtc_implemented
from .zoo.config.enum import ZooTransform
from .zoo.config.factory import get_transform_config as gtc_zoo

Transform = Callable[[DataPoint], DataPoint]
AllTransfromsType = Union[ImplementedTransform, ZooTransform]


def get_transform_config(
    name: Union[str, AllTransfromsType], **kwargs: Any
) -> TransformConfig:
    """
    Factory function to get a transform configuration object from its name
    and parameters.

    Parameters
    ----------
    name : Union[str, Union[ImplementedTransform, ZooTransform]]
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
    if isinstance(name, ZooTransform) or name in ZooTransform.values():
        return gtc_zoo(name, **kwargs)

    elif (
        isinstance(name, ImplementedTransform) or name in ImplementedTransform.values()
    ):
        return gtc_implemented(name, **kwargs)

    else:
        raise ValueError(
            f"Transform {name} is not implemented. "
            f"Please check the documentation for available transforms."
        )
