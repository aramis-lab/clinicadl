from __future__ import annotations

import torch
from monai.metrics import LossMetric as MonaiLossMetric
from torch.nn.modules.loss import _Loss

from clinicadl.losses.config.enum import Reduction


class LossMetric(MonaiLossMetric):
    """
    A wrapper of :py:class:`monai.metrics.LossMetric` to fix
    aggregation issue.

    MONAI's LossMetric only aggregates along batch and/or channel
    dimension, but not along pixels.
    """

    def __init__(self, loss_fn: _Loss, reduction: str | Reduction):
        super().__init__(
            loss_fn=loss_fn, reduction=Reduction(reduction).value, get_not_nans=False
        )

    def aggregate(self) -> torch.Tensor:
        data = self.get_buffer()
        if self.reduction == Reduction.MEAN:
            return data.mean()
        elif self.reduction == Reduction.SUM:
            return data.sum()
