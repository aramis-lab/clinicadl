"""
We chose to overwrite MONAI's CumulativeIterationMetric because here
we wanted to be able to compute the metric for each element of the batch
individually.

Besides, we think our implementation facilitates the creation fo custom
transforms by the user.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Union

import torch
from monai.metrics.metric import CumulativeIterationMetric

TensorOrList = Union[torch.Tensor, Sequence[torch.Tensor]]


class Metric(CumulativeIterationMetric, ABC):
    """
    Transforms must inherit from this class to work with ``ClinicaDL``.

    The user must override :py:meth:`_aggregate` and :py:meth:`_accumulate`.
    """

    @abstractmethod
    def _aggregate(self, data: TensorOrList, **kwargs: Any) -> float:
        """
        Aggregation logic.

        This function tells how to aggregate the data returned by :py:meth:`_accumulate`
        to compute the metric.

        Parameters
        ----------
        data : TensorOrList
            Data useful to compute the metric, as returned by :py:meth:`_accumulate`.

        Returns
        -------
        float
            The aggregated metric.
        """

    @abstractmethod
    def _accumulate(
        self, y_pred: torch.Tensor, y: torch.Tensor | None = None, **kwargs: Any
    ) -> TensorOrList:
        """
        To accumulate data useful for the final metric computation.

        For example, for segmentation, to compute the
        accuracy, this function would just return the confusion matrix.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predictions, as a "batch-first" tensor.
        y : torch.Tensor | None, default=None
            The potential ground truths, as a "batch-first" tensor.

        Returns
        -------
        TensorOrList
            Useful results for the final aggregation, as a "batch-first" tensor, or a sequence
            of "batch-first" tensors.
        """

    def aggregate(self, **kwargs: Any) -> float:
        """
        See :py:meth:`monai.metrics.metric.Cumulative.aggregate`.
        """
        data = self.get_buffer()
        return self._aggregate(data, **kwargs)

    def _compute_tensor(
        self, y_pred: torch.Tensor, y: torch.Tensor | None = None, **kwargs: Any
    ) -> TensorOrList:
        """
        See :py:meth:`monai.metrics.metric.IterationMetric._compute_tensor`.
        Note: :py:meth:`_accumulate` is defined just to have a name more explicit.
        """
        return self._accumulate(y_pred=y_pred, y=y, **kwargs)

    def __call__(
        self, y_pred: torch.Tensor, y: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        """
        See :py:meth:`monai.metrics.metric.CumulativeIterationMetric.__call__`.
        It is modified to get the metric for each element of the batch, whereas the
        original method only accumulates.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predictions, as a "batch-first" tensor.
        y : torch.Tensor | None, default=None
            The potential ground truths, as a "batch-first" tensor.

        Returns
        -------
        torch.Tensor
            The metric value for each element of the batch.
        """
        data = super().__call__(y_pred=y_pred, y=y, **kwargs)

        results = []
        if isinstance(data, torch.Tensor):
            for elem in data:
                res = self._aggregate(elem.unsqueeze(0))
                results.append(res)
        elif isinstance(data, Sequence):
            for elems in zip(*data):
                res = self._aggregate(list(elem.unsqueeze(0) for elem in elems))
                results.append(res)

        return torch.tensor(results)
