from copy import deepcopy
from typing import Tuple

import torch

from clinicadl.utils.factories import DefaultFromLibrary, get_args_and_defaults

from .config import LossConfig


def get_loss_function(config: LossConfig) -> Tuple[torch.nn.Module, LossConfig]:
    """
    Factory function to get a loss function from PyTorch.

    Parameters
    ----------
    loss : LossConfig
        The config class with the parameters of the loss function.

    Returns
    -------
    nn.Module
        The loss function.
    LossConfig
        The updated config class: the arguments set to default will be updated
        with their effective values (the default values from the library).
        Useful for reproducibility.
    """
    loss_class = getattr(torch.nn, config.loss)
    expected_args, config_dict = get_args_and_defaults(loss_class.__init__)
    for arg, value in config.model_dump().items():
        if arg in expected_args and value != DefaultFromLibrary.YES:
            config_dict[arg] = value

    config_dict_ = deepcopy(config_dict)
    if "weight" in config_dict and config_dict["weight"] is not None:
        config_dict_["weight"] = torch.Tensor(config_dict_["weight"])
    if "pos_weight" in config_dict and config_dict["pos_weight"] is not None:
        config_dict_["pos_weight"] = torch.Tensor(config_dict_["pos_weight"])
    loss = loss_class(**config_dict_)

    updated_config = config.model_copy(update=config_dict)

    return loss, updated_config
