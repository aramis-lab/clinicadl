from copy import deepcopy
from typing import Any, Tuple, Union

import torch

from clinicadl.utils.factories import update_config_with_defaults

from .config import LossConfig, create_loss_function_config
from .enum import ImplementedLoss
from .utils import Loss


def get_loss_function_config(
    name: Union[str, ImplementedLoss], **kwargs: Any
) -> LossConfig:
    """
    Factory function to get a loss function configuration object from its name
    and parameters.

    Parameters
    ----------
    name : Union[str, ImplementedLoss]
        the name of the loss function. Check our documentation to know
        available losses.
    **kwargs : Any
        any parameter of the loss function. Check our documentation on losses to
        know these parameters.

    Returns
    -------
    LossConfig
        the config object. Default values will be returned for the parameters
        not passed by the user.
    """
    config = create_loss_function_config(name)(**kwargs)
    loss_class = getattr(torch.nn, config.name)

    update_config_with_defaults(config, function=loss_class.__init__)

    return config


def get_loss_function_from_config(
    config: LossConfig,
) -> Tuple[Loss, LossConfig]:
    """
    Factory function to get a PyTorch loss function from a LossConfig instance.

    Parameters
    ----------
    config : LossConfig
        the configuration object.

    Returns
    -------
    Loss
        the loss function.
    LossConfig
        the updated config object: the arguments set to default will be updated
        with their effective values (the default values from the library).
        Useful for reproducibility.
    """
    config = deepcopy(config)
    loss_class = getattr(torch.nn, config.name)

    update_config_with_defaults(config, function=loss_class.__init__)
    config_dict = config.model_dump(exclude={"name"})

    # change list to tensors
    if "weight" in config_dict and config_dict["weight"] is not None:
        config_dict["weight"] = torch.Tensor(config_dict["weight"])
    if "pos_weight" in config_dict and config_dict["pos_weight"] is not None:
        config_dict["pos_weight"] = torch.Tensor(config_dict["pos_weight"])

    loss = loss_class(**config_dict)

    return loss, config
