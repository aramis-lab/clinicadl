"""
We chose to overwrite MONAI's CumulativeIterationMetric because here
we wanted to work with ``DataPoints``, and to be able to compute the metric
for each element of the batch individually.

Besides, we think our implementation facilitates the creation fo custom
transforms by the user.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal, Union

import torch
from monai.metrics.metric import CumulativeIterationMetric

from clinicadl.data.dataloader import Batch

from .enum import Optimum

TensorOrList = Union[torch.Tensor, Sequence[torch.Tensor]]


class Metric(CumulativeIterationMetric, ABC):
    """
    Transforms must inherit from this class to work with ``ClinicaDL``.

    The user must override :py:meth:`_aggregate`, and :py:meth:`_accumulate`.

    The user must also define the attribute ``_optimum``:

    - use "min" when a lower metric value indicates better performance.
    - use "max" when a higher metric value indicates better performance.

    Finally, ``__init__`` can be overwrite, but don't forget to call
    ``super().__init__()`` inside.
    """

    _optimum: Literal["min", "max"]

    @abstractmethod
    def _aggregate(self, data: TensorOrList) -> float:
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
    def _accumulate(self, batch: Batch) -> TensorOrList:
        """
        To accumulate data useful for the final metric computation.

        For example, for segmentation, to compute the
        accuracy, this function would just return the confusion matrix
        for each element of the batch.

        Parameters
        ----------
        batch : Batch
            The batch of :py:class:`~clinicadl.data.structures.DataPoint`,
            passed via a :py:class:`~clinicadl.data.dataloader.Batch`.

        Returns
        -------
        TensorOrList
            Useful results for the final aggregation, as a "batch-first" tensor, or a sequence
            of "batch-first" tensors.
        """

    @property
    def optimum(self) -> Optimum:
        """Optimization criterion for the metric."""
        return Optimum(self._optimum)

    # pylint: disable=arguments-differ
    def aggregate(self) -> float:
        """
        See :py:meth:`monai.metrics.metric.Cumulative.aggregate`.
        """
        data = self.get_buffer()
        return self._aggregate(data)

    # pylint: disable=signature-differs
    def __call__(self, batch: Batch) -> torch.Tensor:
        """
        See :py:meth:`monai.metrics.metric.CumulativeIterationMetric.__call__`.

        It is modified to accept a batch of :py:class:`~clinicadl.data.structures.DataPoint`,
        and to get the metric for each element of the batch, whereas the
        original method only accumulates.

        Parameters
        ----------
        batch : Batch
            The batch of :py:class:`~clinicadl.data.structures.DataPoint`,
            passed via a :py:class:`~clinicadl.data.dataloader.Batch`.

        Returns
        -------
        torch.Tensor
            The metric value for each element of the batch.
        """
        # get the data for metric computation
        data = self._accumulate(batch)

        # store the data in the buffers
        if isinstance(data, Sequence):
            self.extend(*data)
        else:
            self.extend(data)

        # compute the metric for each element of the batch
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

    def _compute_tensor(self, batch: Batch) -> TensorOrList:
        """
        See :py:meth:`monai.metrics.metric.IterationMetric._compute_tensor`.

        Note: :py:meth:`_accumulate` is defined just to have a name more explicit.

        ``_compute_tensor`` is actually not used, but it is mandatory to override it
        (see :py:class:`monai.metrics.metric.IterationMetric`).
        """
        return self._accumulate(batch)
