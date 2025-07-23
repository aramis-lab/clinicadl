from collections.abc import Sequence
from typing import Any

import torch
from monai.metrics import CumulativeIterationMetric

from .base import Metric, TensorOrList


class MonaiMetricWrapper(Metric):
    """
    Converts a metric from ``MONAI`` to a metric compatible with ``ClinicaDL``.
    """

    def __init__(self, metric: CumulativeIterationMetric) -> None:
        super().__init__()
        self.metric = metric
        self.metric.reset()

    def __repr__(self):
        return f"{self.__class__.__name__}(metric={repr(self.metric)})"

    def _aggregate(self, data: TensorOrList, **kwargs: Any) -> float:
        """
        See :py:meth:`clinicadl.metrics.Metric._aggregate`.

        :py:meth:`clinicadl.metrics.Metric._aggregate` is for the computation
        on a single local batch, whereas ``aggregate`` methods from MONAI metrics
        compute the metric on all the batches from all the devices.
        So, to adapt the latter, we need to fake synchronization.
        """
        # empty the local buffer
        self.metric.reset()

        # add the data to the local buffer
        if isinstance(data, Sequence):
            self.metric.extend(*data)
        else:
            self.metric.extend(data)

        # make sure the computation is local by faking synchronization
        self.metric._synced = True
        self.metric._synced_tensors = [
            torch.stack(b, dim=0) for b in self.metric._buffers
        ]

        # now we can call 'aggregate' to compute the metric. self.metric._synced = True, so it won't do synchronization
        res = self.metric.aggregate(**kwargs)

        # make sure to return a float
        if isinstance(res, Sequence) and len(res) == 1:
            res = res[0]
        if isinstance(res, torch.Tensor):
            try:
                return res.item()
            except RuntimeError:
                pass

        return res

    def _accumulate(
        self, y_pred: torch.Tensor, y: torch.Tensor | None = None, **kwargs: Any
    ) -> TensorOrList:
        """
        See :py:meth:`clinicadl.metrics.Metric._accumulate`.
        """
        return self.metric._compute_tensor(y_pred=y_pred, y=y, **kwargs)
