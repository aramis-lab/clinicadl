import inspect

import torch.nn as nn

from .config import LossConfig


def get_loss_function(config: LossConfig) -> nn.Module:
    """
    Factory function to get a loss function from its name.

    Parameters
    ----------
    loss : LossConfig
        The config class with the parameters of the loss function.

    Returns
    -------
    nn.Module
        The loss function.
    """
    loss_class = getattr(nn, config.loss)
    expected_args = inspect.getfullargspec(loss_class).args
    config = {arg: v for arg, v in config.model_dump().items() if arg in expected_args}
    loss = loss_class(**config)

    return loss


# TODO : what about them?
# "KLDivLoss",
# "BCEWithLogitsLoss",
# "VAEGaussianLoss",
# "VAEBernoulliLoss",
# "VAEContinuousBernoulliLoss",
