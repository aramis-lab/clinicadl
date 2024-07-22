import inspect
from copy import deepcopy
from typing import Any, Dict, Tuple

import torch

from .config import LossConfig


def get_loss_function(config: LossConfig) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Factory function to get a loss function.

    Parameters
    ----------
    loss : LossConfig
        The config class with the parameters of the loss function.

    Returns
    -------
    nn.Module
        The loss function.
    Dict[str, Any]
        The config dict with only the parameters relevant to the selected
        loss function.
    """
    loss_class = getattr(torch.nn, config.loss)
    expected_args = inspect.getfullargspec(loss_class).args
    config_dict = {
        arg: v for arg, v in config.model_dump().items() if arg in expected_args
    }

    config_dict_ = deepcopy(config_dict)
    if config.weight is not None:
        config_dict_["weight"] = torch.Tensor(config_dict_["weight"])
    loss = loss_class(**config_dict_)

    config_dict["loss"] = config.loss

    return loss, config_dict


# TODO : what about them?
# "KLDivLoss",
# "BCEWithLogitsLoss",
# "VAEGaussianLoss",
# "VAEBernoulliLoss",
# "VAEContinuousBernoulliLoss",
