from copy import deepcopy
from typing import Any, Tuple, Union

import torch

from clinicadl.utils.factories import DefaultFromLibrary, get_args_and_defaults

from .config import LossConfig, create_loss_config
from .enum import ImplementedLoss
from .utils import Loss


def get_loss_function(
    name: Union[str, ImplementedLoss], return_config: bool = False, **kwargs: Any
) -> Union[Loss, Tuple[Loss, LossConfig]]:
    """
    Factory function to get a loss function from its name and parameters.

    Parameters
    ----------
    name : Union[str, ImplementedLoss]
        the name of the loss function. Check our documentation to know
        available losses.
    return_config : bool (optional, default=False)
        if the function should return the config class regrouping the parameters of the
        loss function. Useful to keep track of the hyperparameters.
    **kwargs : Any
        the parameters of the loss function. Check our documentation on losses to
        know these parameters.

    Returns
    -------
    nnn.Module
        the loss function.
    LossConfig
        the associated config class. Only returned if `return_config` is True.
    """
    config = create_loss_config(name)(**kwargs)
    loss, updated_config = get_loss_function_from_config(config)

    return loss if not return_config else (loss, updated_config)


def get_loss_function_from_config(
    config: LossConfig,
) -> Tuple[Loss, LossConfig]:
    """
    Factory function to get a loss function from a LossConfig instance.

    Parameters
    ----------
    loss : LossConfig
        the configuration object.

    Returns
    -------
    nn.Module
        the loss function.
    LossConfig
        the updated config class: the arguments set to default will be updated
        with their effective values (the default values from the library).
        Useful for reproducibility.
    """
    config = deepcopy(config)
    loss_class = getattr(torch.nn, config.name)

    # update config with defaults
    _, defaults = get_args_and_defaults(loss_class.__init__)
    for arg, value in config:
        if value == DefaultFromLibrary.YES and arg in defaults:
            setattr(config, arg, defaults[arg])

    config_dict = config.model_dump(exclude={"name"})
    if "weight" in config_dict and config_dict["weight"] is not None:
        config_dict["weight"] = torch.Tensor(config_dict["weight"])
    if "pos_weight" in config_dict and config_dict["pos_weight"] is not None:
        config_dict["pos_weight"] = torch.Tensor(config_dict["pos_weight"])

    loss = loss_class(**config_dict)

    return loss, config
