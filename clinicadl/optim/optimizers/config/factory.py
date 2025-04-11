from typing import Any, Union

from .base import ImplementedOptimizer, OptimizerConfig
from .configs import *


def get_optimizer_config(
    name: Union[str, ImplementedOptimizer],
    **kwargs: Any,
) -> OptimizerConfig:
    """
    Factory function to get an optimizer configuration object from its name
    and parameters.

    Parameters
    ----------
    name : Union[str, ImplementedOptimizer]
        the name of the optimizer. Check our documentation to know
        available optimizers.
    **kwargs : Any
        any parameter of the optimizer. Check our documentation on optimizers to
        know these parameters.

    Returns
    -------
    OptimizerConfig
        the configuration object. Default values will be returned for the parameters
        not passed by the user.
    """
    optimizer = ImplementedOptimizer(name).value
    config_name = f"{optimizer}Config"
    config = globals()[config_name]

    return config(**kwargs)
