from collections.abc import Sequence
from typing import Optional

import torch
from monai.metrics import CumulativeIterationMetric

from clinicadl.data.dataloader import Batch
from clinicadl.transforms.handlers import Postprocessing
from clinicadl.transforms.types import TransformOrConfig

from .base import Metric, TensorOrList
from .enum import Optimum


class MonaiMetricWrapper(Metric):
    """
    Converts a metric from ``MONAI`` to a metric compatible with ``ClinicaDL``.

    Parameters
    ----------
    metric : CumulativeIterationMetric
        The metric to wrap.
    optimum : Optimum
        Either ``"max"`` or ``"min"``:

        - use "min" when a lower metric value indicates better performance.
        - use "max" when a higher metric value indicates better performance.

    pred_key : str
        The key corresponding to the prediction in the input :py:class:`~clinicadl.data.structures.DataPoint`.
        The value associated to the key must be a :py:class:`torchio.Image`, a :py:class:`torch.torch.Tensor`,
        a :py:class:`numpy.ndarray`, or a numeric value.
    label_key : Optional[str] = None
        The key corresponding to the ground truth label in the input :py:class:`~clinicadl.data.structures.DataPoint`.
        The value associated to the key must be a :py:class:`torchio.Image`, a :py:class:`torch.torch.Tensor`,
        a :py:class:`numpy.ndarray`, or a numeric value.

        Leave to ``None`` if no ground truth is used to compute the metric.
    """

    def __init__(
        self,
        metric: CumulativeIterationMetric,
        optimum: Optimum,
        pred_key: str,
        label_key: Optional[str] = None,
        postprocessing: Optional[list[TransformOrConfig]] = None,
    ) -> None:
        super().__init__()
        self.pred_key = pred_key
        self.label_key = label_key
        self.metric = metric
        self.metric.reset()
        self._optimum = optimum
        self.postprocessing = (
            Postprocessing(transforms=postprocessing) if postprocessing else None
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(metric={repr(self.metric)}, "
            f"optimum='{self.optimum.value}', "
            f"pred_key='{self.pred_key}', "
            f"label_key='{self.label_key}')"
        )

    def _aggregate(self, data: TensorOrList) -> float:
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
        res = self.metric.aggregate()

        # make sure to return a float
        if isinstance(res, Sequence) and len(res) == 1:
            res = res[0]
        if isinstance(res, torch.Tensor):
            try:
                return res.item()
            except RuntimeError:
                pass

        return res

    def _accumulate(self, batch: Batch) -> TensorOrList:
        """
        See :py:meth:`clinicadl.metrics.Metric._accumulate`.
        """
        if self.postprocessing:
            batch = self.postprocessing.batch_apply(batch)

        y_pred = batch.get_field(self.pred_key, ensure_channel_dim=True)
        if self.label_key:
            y = batch.get_field(self.label_key, ensure_channel_dim=True)
        else:
            y = None

        return self.metric._compute_tensor(y_pred=y_pred, y=y)
