from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Type, Union

if TYPE_CHECKING:
    import torch.nn as nn


class ClassificationLoss(str, Enum):
    """Losses that can be used only for classification."""

    CrossENTROPY = "CrossEntropyLoss"
    MultiMargin = "MultiMarginLoss"


class ImplementedLoss(str, Enum):
    """Implemented losses in ClinicaDL."""

    CrossENTROPY = "CrossEntropyLoss"
    MultiMargin = "MultiMarginLoss"
    L1 = "L1Loss"
    MSE = "MSELoss"
    HUBER = "HuberLoss"
    SmoothL1 = "SmoothL1Loss"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not implemented. Implemented losses are: "
            + ", ".join([repr(m.value) for m in cls])
        )


def get_loss_function(loss: Union[str, ImplementedLoss]) -> Type[nn.Module]:
    """
    Factory function to get a loss function from its name.

    Parameters
    ----------
    loss : Union[str, ImplementedLoss]
        The name of the loss.

    Returns
    -------
    Type[nn.Module]
        The loss function object.
    """
    import torch.nn as nn

    loss = ImplementedLoss(loss)
    return getattr(nn, loss.value)


# TODO : what about them?
# "KLDivLoss",
# "BCEWithLogitsLoss",
# "VAEGaussianLoss",
# "VAEBernoulliLoss",
# "VAEContinuousBernoulliLoss",
